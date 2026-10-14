"""방식 C / D 학습: 이미지만(CNN) 또는 이미지 + 수치(융합)

프로젝트 루트(~/dev_ws/mediaPipe)에서 실행:
  python -m learning.image_train --kind cnn               # → models/cnn.pt
  python -m learning.image_train --kind fusion            # → models/fusion.pt
  python -m learning.image_train --kind cnn --train-fraction 0.25 --out models/cnn_25.pt
  python -m learning.image_train --kind cnn --labels fine   # 거북목·구부정을 나눈 5개 라벨로 학습
  python -m learning.image_train --kind cnn --image-dir images_crop   # 사람 영역 crop 이미지로 학습
                                                          # (먼저 python -m learning.crop_images)
  python -m learning.image_train --kind cnn --image-dir images_cropmask --color gray   # 흑백으로 학습

분할은 posture_dataset.load_splits(images_only=True)로 다른 방식과 동일하게 한다.
GPU가 있으면 자동으로 사용한다 (CPU로도 가능하지만 느림).
"""
import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from evaluation.eval_metrics import balanced_acc, confusion, format_cm
from learning.image_model import COLOR_MODES, FrameDataset, build, image_probs, save_image_model
from learning.labels import DEFAULT, SCHEMES, classes
from learning.posture_dataset import load_splits


def parse_args():
    root = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser(description="이미지 / 융합 모델 학습")
    ap.add_argument("--kind", choices=["cnn", "fusion"], required=True)
    ap.add_argument("--data", default=str(root / "data"))
    ap.add_argument("--out", default="")
    ap.add_argument("--labels", choices=list(SCHEMES), default=DEFAULT,
                    help="라벨 체계: merged = 거북목·구부정 합침 (기본), fine = 5개 그대로")
    ap.add_argument("--image-dir", default="images",
                    help="이미지 폴더: images = 원본 (기본), images_crop = 사람 영역 crop")
    ap.add_argument("--color", choices=COLOR_MODES, default="basic",
                    help="색 처리: basic (기존), jitter = 색조 변경·흑백 섞기, gray = 항상 흑백")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--val-ratio", type=float, default=0.2)
    ap.add_argument("--train-fraction", type=float, default=1.0)
    ap.add_argument("--no-pretrained", action="store_true", help="사전학습 가중치 없이 처음부터 학습")
    args = ap.parse_args()
    args.out = args.out or str(root / "models" / f"{args.kind}.pt")
    return args


def num_stats(X):
    mean, std = X.mean(0), X.std(0)
    std[std < 1e-6] = 1.0
    return mean.astype(np.float32), std.astype(np.float32)


def class_weight_tensor(y, n):
    counts = np.bincount(y, minlength=n).astype(np.float32)
    w = np.where(counts > 0, counts.sum() / (n * np.maximum(counts, 1)), 0.0)
    return torch.tensor(w, dtype=torch.float32)


def run_epoch(model, loader, loss_fn, opt, device):
    model.train()
    total, n = 0.0, 0
    for img, num, y in loader:
        img, num, y = img.to(device), num.to(device), y.to(device)
        opt.zero_grad()
        loss = loss_fn(model(img, num), y)
        loss.backward()
        opt.step()
        total += loss.item() * len(y)
        n += len(y)
    return total / max(n, 1)


def main():
    args = parse_args()
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    names = classes(args.labels)
    n_cls = len(names)
    train_split, val_split, _ = load_splits(args.data, args.val_ratio, images_only=True, scheme=args.labels)
    train_split = train_split.fraction(args.train_fraction).with_image_dir(args.image_dir)
    val_split = val_split.with_image_dir(args.image_dir)
    if len(train_split) == 0:
        raise SystemExit("이미지가 있는 학습 프레임이 없습니다. "
                         "collect.posture_collect에서 --img-every 0이 아닌지 확인하세요.")
    missing = train_split.missing_images() + val_split.missing_images()
    if missing:
        raise SystemExit(f"{args.image_dir} 이미지 {len(missing)}장이 없습니다 (예: {missing[0]}). "
                         f"python -m learning.crop_images를 먼저 실행하세요.")
    print(f"[{args.kind}] 학습 {len(train_split)}장 (비율 {args.train_fraction:.0%}) / "
          f"검증 {len(val_split)}장, 이미지 {args.image_dir}, 색 {args.color}, device={device}")

    mean, std = num_stats(train_split.X)
    loader = lambda s, train: DataLoader(                           # noqa: E731
        FrameDataset(s, mean, std, train, args.color), batch_size=args.batch, shuffle=train,
        num_workers=args.workers, pin_memory=(device == "cuda"))
    train_loader, val_loader = loader(train_split, True), loader(val_split, False)

    model = build(args.kind, pretrained=not args.no_pretrained, n_cls=n_cls,
                  gray=args.color == "gray").to(device)
    loss_fn = nn.CrossEntropyLoss(weight=class_weight_tensor(train_split.y, n_cls).to(device))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    best, best_state = -1.0, None
    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        loss = run_epoch(model, train_loader, loss_fn, opt, device)
        sched.step()
        pred = image_probs(model, val_loader, device).argmax(1)
        score = balanced_acc(confusion(val_split.y, pred, n_cls))
        if score > best:
            best, best_state = score, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(f"epoch {ep:3d}  loss {loss:.4f}  검증 균형 정확도 {score:.3f}  ({time.time() - t0:.0f}초)")

    model.load_state_dict(best_state)
    pred = image_probs(model, val_loader, device).argmax(1)
    print("\n" + format_cm(confusion(val_split.y, pred, n_cls), f"검증 · {args.kind}", names))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    save_image_model(args.out, args.kind, model.cpu(), mean, std, names, args.image_dir, args.color)
    print(f"\n저장: {args.out}  (최종 비교는 evaluation.compare로 테스트 세트에서)")


if __name__ == "__main__":
    main()