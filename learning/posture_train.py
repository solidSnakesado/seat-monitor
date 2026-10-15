"""방식 B: 수치 데이터(기준 대비 변화량 5개) → MLP 학습

프로젝트 루트(~/dev_ws/mediaPipe)에서 실행:
  python -m learning.posture_train                       # 모든 프레임 사용
  python -m learning.posture_train --images-only         # 네 방식 비교용: 이미지 있는 프레임만 (같은 데이터)
  python -m learning.posture_train --train-fraction 0.25 --out models/mlp_25.pt   # 데이터 효율 실험
  python -m learning.posture_train --labels fine         # 거북목·구부정을 나눈 5개 라벨로 학습
  python -m learning.posture_train --offset-aug 1.0      # 기준 자세 오차에 둔감하게 (오프셋 증강)

오프셋 증강: 기준 자세를 저장할 때 몸이 조금 기울어 있으면 모든 수치가 일정하게 어긋난다.
학습 때 수치에 무작위 오프셋을 더해, 작은 어긋남은 자세 변화로 보지 않도록 만든다.
분할은 posture_dataset.load_splits()로 다른 방식과 동일하게 한다.
최종 비교(테스트 세트)는 evaluation/compare.py에서 한다. 여기서는 검증 세트로 모델만 고른다.
"""
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from evaluation.eval_metrics import balanced_acc, confusion, format_cm, rule_predict
from learning.labels import DEFAULT, SCHEMES, classes
from learning.posture_dataset import load_splits
from learning.posture_model import INPUTS, PostureMLP, mlp_probs, save_model

# --offset-aug 1.0일 때 더할 오프셋의 최대 크기 (실제로 관찰된 어긋남: 어깨 기울기 3.2°, 어깨 높이 0.031)
OFFSET = {"ear_shoulder": 0.03, "head_ratio": 0.03, "shoulder_y": 0.03, "shoulder_tilt": 3.0, "ear_tilt": 3.0}


def parse_args():
    root = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser(description="자세 분류 MLP 학습 (수치 데이터)")
    ap.add_argument("--data", default=str(root / "data"))
    ap.add_argument("--out", default=str(root / "models" / "posture_mlp.pt"))
    ap.add_argument("--labels", choices=list(SCHEMES), default=DEFAULT,
                    help="라벨 체계: merged = 거북목·구부정 합침 (기본), fine = 5개 그대로")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val-ratio", type=float, default=0.2)
    ap.add_argument("--val-sessions", nargs="+", default=None,
                    help="통째로 검증에 쓸 세션 (파일 이름 일부, 예: round_stripe). 지정하면 --val-ratio 무시")
    ap.add_argument("--images-only", action="store_true", help="이미지가 있는 프레임만 사용")
    ap.add_argument("--train-fraction", type=float, default=1.0, help="학습 데이터 사용 비율")
    ap.add_argument("--offset-aug", type=float, default=0.0,
                    help="오프셋 증강 배율 (0 = 사용 안 함, 1.0 = 기울기 ±3°, 비율·어깨 높이 ±0.03)")
    return ap.parse_args()


def class_weight_tensor(y, n):
    counts = np.bincount(y, minlength=n).astype(np.float32)
    w = np.where(counts > 0, counts.sum() / (n * np.maximum(counts, 1)), 0.0)
    return torch.tensor(w, dtype=torch.float32)


def train(train_split, val_split, epochs, lr, n_cls, offset_aug=0.0):
    Xtr, ytr = train_split.X, train_split.y
    mean, std = Xtr.mean(0), Xtr.std(0)
    std[std < 1e-6] = 1.0
    xt = torch.tensor((Xtr - mean) / std, dtype=torch.float32)
    yt = torch.tensor(ytr)
    off = np.array([OFFSET[k] for k in INPUTS], dtype=np.float32) * offset_aug / std
    off = torch.tensor(off, dtype=torch.float32)              # 정규화된 단위로 바꾼 오프셋 최대 크기

    model = PostureMLP(n_cls=n_cls)
    loss_fn = nn.CrossEntropyLoss(weight=class_weight_tensor(ytr, n_cls))
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    best, best_state = -1.0, None
    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(len(yt))
        for i in range(0, len(yt), 64):
            idx = perm[i:i + 64]
            xb = xt[idx]
            if offset_aug > 0:                                 # 프레임마다 -off ~ +off 균등 분포
                xb = xb + (torch.rand_like(xb) * 2 - 1) * off
            opt.zero_grad()
            loss = loss_fn(model(xb), yt[idx])
            loss.backward()
            opt.step()
        model.eval()
        pred = mlp_probs(model, mean, std, val_split.X).argmax(1)
        score = balanced_acc(confusion(val_split.y, pred, n_cls))
        if score > best:
            best, best_state = score, {k: v.clone() for k, v in model.state_dict().items()}
        if ep == 1 or ep % 20 == 0:
            print(f"epoch {ep:3d}  loss {loss.item():.4f}  검증 균형 정확도 {score:.3f}")
    model.load_state_dict(best_state)
    model.eval()
    return model, mean.astype(np.float32), std.astype(np.float32)


def main():
    args = parse_args()
    torch.manual_seed(0)
    names = classes(args.labels)
    train_split, val_split, _ = load_splits(args.data, args.val_ratio, args.images_only, scheme=args.labels,
                                            val_sessions=args.val_sessions)
    train_split = train_split.fraction(args.train_fraction)
    print(f"학습에 사용: {len(train_split)} 프레임 (비율 {args.train_fraction:.0%})")
    print("학습 클래스별:", dict(zip(names, np.bincount(train_split.y, minlength=len(names)).tolist())))
    if args.offset_aug > 0:
        print("오프셋 증강:", {k: round(v * args.offset_aug, 3) for k, v in OFFSET.items()})

    model, mean, std = train(train_split, val_split, args.epochs, args.lr, len(names), args.offset_aug)
    mlp_pred = mlp_probs(model, mean, std, val_split.X).argmax(1)
    rule_pred = rule_predict(val_split.X, scheme=args.labels)
    print("\n" + format_cm(confusion(val_split.y, rule_pred, len(names)), "검증 · 규칙", names))
    print("\n" + format_cm(confusion(val_split.y, mlp_pred, len(names)), "검증 · MLP", names))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    save_model(args.out, model, mean, std, names)
    print(f"\n저장: {args.out}  (최종 비교는 evaluation.compare로 테스트 세트에서)")


if __name__ == "__main__":
    main()