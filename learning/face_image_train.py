"""얼굴 방식 C / D 학습: 얼굴 crop 이미지만(CNN) 또는 이미지 + 수치(융합)

프로젝트 루트(~/dev_ws/mediaPipe)에서 실행:
  python -m learning.face_image_train --kind cnn --val-sessions g0 --out models/face/check/cnn.pt
  python -m learning.face_image_train --kind fusion --val-sessions g0 --out models/face/check/fusion.pt

모델은 자세와 같은 MobileNetV3-Small (learning/image_model.py의 PostureCNN, FusionNet)을 그대로 쓰고,
입력만 얼굴 crop 128×128 (collect.face_recorder가 원본 해상도에서 잘라 저장)과 얼굴 수치 특징으로 바꾼다.
좌우 반전은 쓰지 않는다 (고개 방향 정보가 바뀌기 때문). 밝기·대비 변형만 사용.
외우기 억제 옵션(--freeze, --label-smoothing, --erase)은 자세의 image_train과 같다.
"""
import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

from collect.face_guide import FACE_CLASSES, FACE_KO
from evaluation.eval_metrics import balanced_acc, confusion, format_cm
from learning.face_dataset import NUM_INPUTS, load_face_splits
from learning.image_model import MEAN, STD, FusionNet, PostureCNN, image_probs
from learning.image_train import class_weight_tensor, freeze_blocks, num_stats, run_epoch

FACE_HW = (128, 128)


def parse_args():
    root = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser(description="얼굴 이미지 / 융합 모델 학습")
    ap.add_argument("--kind", choices=["cnn", "fusion"], required=True)
    ap.add_argument("--data", default=str(root / "data" / "face"))
    ap.add_argument("--out", default="")
    ap.add_argument("--val-sessions", nargs="+", required=True, help="통째로 검증에 쓸 세션 (파일 이름 일부)")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--freeze", type=int, default=0, help="백본 앞쪽 N블록 고정 (0~13)")
    ap.add_argument("--label-smoothing", type=float, default=0.0)
    ap.add_argument("--erase", type=float, default=0.0, help="무작위 가리기 확률")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    args.out = args.out or str(root / "models" / "face" / f"{args.kind}.pt")
    return args


def face_transform(train, erase=0.0):
    aug = [T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2)] if train else []
    post = [T.RandomErasing(p=erase, scale=(0.02, 0.15), value=0)] if (train and erase > 0) else []
    return T.Compose([T.Resize(FACE_HW), *aug, T.ToTensor(), T.Normalize(MEAN, STD), *post])


class FaceImageDataset(Dataset):
    """FaceSplit(이미지 있는 프레임) → (얼굴 이미지 텐서, 정규화된 수치, 라벨)"""

    def __init__(self, split, num_mean, num_std, train, erase=0.0):
        self.split = split
        self.num = ((split.X - num_mean) / num_std).astype(np.float32)
        self.tf = face_transform(train, erase)

    def __len__(self):
        return len(self.split)

    def __getitem__(self, i):
        img = Image.open(self.split.images[i]).convert("RGB")
        return self.tf(img), torch.from_numpy(self.num[i]), int(self.split.y[i])


def build_face(kind, pretrained=True):
    n_cls = len(FACE_CLASSES)
    if kind == "cnn":
        return PostureCNN(pretrained, n_cls=n_cls)
    return FusionNet(pretrained, n_num=len(NUM_INPUTS), n_cls=n_cls)


def save_face_image_model(path, kind, model, mean, std):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"kind": kind, "state": model.state_dict(), "classes": list(FACE_CLASSES),
                "inputs": list(NUM_INPUTS), "num_mean": mean.tolist(), "num_std": std.tolist()}, path)


def load_face_image_model(path, device="cpu"):
    """→ (model, kind, num_mean, num_std). 사전학습 가중치는 다시 받지 않는다"""
    ck = torch.load(path, map_location=device)
    if tuple(ck["inputs"]) != NUM_INPUTS or tuple(ck["classes"]) != FACE_CLASSES:
        raise ValueError(f"{path}: 입력·클래스가 현재 코드와 다릅니다. 다시 학습하세요.")
    model = build_face(ck["kind"], pretrained=False)
    model.load_state_dict(ck["state"])
    model.to(device).eval()
    return (model, ck["kind"], np.array(ck["num_mean"], dtype=np.float32),
            np.array(ck["num_std"], dtype=np.float32))


def face_loader(split, mean, std, train, batch, workers, erase=0.0, device="cpu"):
    return DataLoader(FaceImageDataset(split, mean, std, train, erase), batch_size=batch, shuffle=train,
                      num_workers=workers, pin_memory=(device == "cuda"))


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    names = [FACE_KO[c] for c in FACE_CLASSES]
    n_cls = len(names)
    tr, va, _ = load_face_splits(args.data, args.val_sessions)
    tr, va = tr.with_images(), va.with_images()
    if len(tr) == 0 or len(va) == 0:
        raise SystemExit("이미지가 있는 학습·검증 프레임이 없습니다. --val-sessions를 확인하세요.")
    missing = tr.missing_images() + va.missing_images()
    if missing:
        raise SystemExit(f"얼굴 이미지 {len(missing)}장이 없습니다 (예: {missing[0]})")
    print(f"[{args.kind}] 학습 {len(tr)}장 / 검증 {len(va)}장, device={device}")
    print(f"외우기 억제: 고정 {args.freeze}블록, 라벨 스무딩 {args.label_smoothing}, 가리기 {args.erase}")

    mean, std = num_stats(tr.X)
    train_loader = face_loader(tr, mean, std, True, args.batch, args.workers, args.erase, device)
    val_loader = face_loader(va, mean, std, False, args.batch, args.workers, device=device)
    model = build_face(args.kind).to(device)
    frozen = freeze_blocks(model, args.freeze)
    loss_fn = nn.CrossEntropyLoss(weight=class_weight_tensor(tr.y, n_cls).to(device),
                                  label_smoothing=args.label_smoothing)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    best, best_state, best_ep = -1.0, None, 0
    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        loss = run_epoch(model, train_loader, loss_fn, opt, device, frozen)
        sched.step()
        score = balanced_acc(confusion(va.y, image_probs(model, val_loader, device).argmax(1), n_cls))
        if score > best:
            best, best_ep = score, ep
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(f"epoch {ep:3d}  loss {loss:.4f}  검증 균형 정확도 {score:.3f}  ({time.time() - t0:.0f}초)")

    model.load_state_dict(best_state)
    print(f"선택: epoch {best_ep} (검증 균형 정확도 {best:.3f})")
    pred = image_probs(model, val_loader, device).argmax(1)
    print("\n" + format_cm(confusion(va.y, pred, n_cls), f"검증 · {args.kind}", names))
    save_face_image_model(args.out, args.kind, model.cpu(), mean, std)
    print(f"\n저장: {args.out}  (다른 방식과의 비교는 evaluation.face_compare)")


if __name__ == "__main__":
    main()