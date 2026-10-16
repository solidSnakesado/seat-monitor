"""얼굴 방식 B: 수치 데이터(blendshape 51개 + 영역 기준 각도 5개) → MLP 학습

프로젝트 루트(~/dev_ws/mediaPipe)에서 실행:
  python -m learning.face_train --val-sessions g0 --out models/face/check/mlp.pt
  python -m learning.face_train --val-sessions g0 --offset-aug 0      # 각도 오프셋 증강 끄기

분할은 learning.face_dataset.load_face_splits (세션 단위 검증). 검증 점수가 가장 높은 epoch를 저장한다.
각도 오프셋 증강: 영역 캘리브레이션 때 각도가 조금 어긋나면 모든 각도 특징이 일정하게 밀린다.
학습 때 각도에 무작위 오프셋을 더해 작은 어긋남을 상태 변화로 보지 않게 한다.
최종 비교는 evaluation.face_compare에서 같은 프레임으로 한다. 여기서는 A 규칙과 B를 검증 세트로 함께 출력한다.
"""
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from collect.face_guide import FACE_CLASSES, FACE_KO, SUBTAG_KO
from evaluation.eval_metrics import balanced_acc, confusion, format_cm
from learning.face_dataset import NUM_INPUTS, false_alarms, load_face_splits, rule_predict
from learning.posture_model import PostureMLP, mlp_probs

ANGLE_OFFSET = {"d_yaw": 3.0, "d_pitch": 3.0, "roll": 3.0, "d_desk": 3.0}   # --offset-aug 1.0일 때 최대 (도)


def parse_args():
    root = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser(description="얼굴 상태 MLP 학습 (수치 데이터)")
    ap.add_argument("--data", default=str(root / "data" / "face"))
    ap.add_argument("--out", default=str(root / "models" / "face" / "mlp.pt"))
    ap.add_argument("--val-sessions", nargs="+", required=True, help="통째로 검증에 쓸 세션 (파일 이름 일부)")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--offset-aug", type=float, default=1.0, help="각도 오프셋 증강 배율 (0 = 사용 안 함)")
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args()


def class_weight_tensor(y, n):
    counts = np.bincount(y, minlength=n).astype(np.float32)
    w = np.where(counts > 0, counts.sum() / (n * np.maximum(counts, 1)), 0.0)
    return torch.tensor(w, dtype=torch.float32)


def train(tr, va, epochs, lr, hidden, offset_aug):
    n_cls = len(FACE_CLASSES)
    mean, std = tr.X.mean(0), tr.X.std(0)
    std[std < 1e-6] = 1.0
    xt, yt = torch.tensor((tr.X - mean) / std, dtype=torch.float32), torch.tensor(tr.y)
    off = np.array([ANGLE_OFFSET.get(k, 0.0) for k in NUM_INPUTS], dtype=np.float32) * offset_aug / std
    off = torch.tensor(off, dtype=torch.float32)              # 정규화 단위의 오프셋 최대 크기

    model = PostureMLP(n_in=len(NUM_INPUTS), n_cls=n_cls, hidden=hidden)
    loss_fn = nn.CrossEntropyLoss(weight=class_weight_tensor(tr.y, n_cls))
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    best, best_state = -1.0, None
    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(len(yt))
        for i in range(0, len(yt), 128):
            idx = perm[i:i + 128]
            xb = xt[idx]
            if offset_aug > 0:                                 # 배치마다 같은 오프셋 (캘리브레이션 어긋남 흉내)
                xb = xb + (torch.rand(1, xb.shape[1]) * 2 - 1) * off
            opt.zero_grad()
            loss = loss_fn(model(xb), yt[idx])
            loss.backward()
            opt.step()
        model.eval()
        score = balanced_acc(confusion(va.y, mlp_probs(model, mean, std, va.X).argmax(1), n_cls))
        if score > best:
            best, best_state, best_ep = score, {k: v.clone() for k, v in model.state_dict().items()}, ep
        if ep == 1 or ep % 10 == 0:
            print(f"epoch {ep:3d}  loss {loss.item():.4f}  검증 균형 정확도 {score:.3f}")
    model.load_state_dict(best_state)
    model.eval()
    print(f"선택: epoch {best_ep} (검증 균형 정확도 {best:.3f})")
    return model, mean.astype(np.float32), std.astype(np.float32)


def save_face_mlp(path, model, mean, std, hidden):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state": model.state_dict(), "mean": mean.tolist(), "std": std.tolist(), "hidden": hidden,
                "classes": list(FACE_CLASSES), "inputs": list(NUM_INPUTS)}, path)


def load_face_mlp(path):
    """→ (model, mean, std)"""
    ck = torch.load(path, map_location="cpu")
    if tuple(ck["inputs"]) != NUM_INPUTS or tuple(ck["classes"]) != FACE_CLASSES:
        raise ValueError(f"{path}: 입력·클래스가 현재 코드와 다릅니다. 다시 학습하세요.")
    model = PostureMLP(n_in=len(NUM_INPUTS), n_cls=len(FACE_CLASSES), hidden=ck["hidden"])
    model.load_state_dict(ck["state"])
    model.eval()
    return model, np.array(ck["mean"], dtype=np.float32), np.array(ck["std"], dtype=np.float32)


def alarm_lines(split, pred):
    return [f"  {SUBTAG_KO[t]:<10} {n:6d} 프레임  오경보 {r:6.1%}"
            for t, (n, r) in false_alarms(split, pred).items()]


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    names = [FACE_KO[c] for c in FACE_CLASSES]
    tr, va, _ = load_face_splits(args.data, args.val_sessions)
    if len(tr) == 0 or len(va) == 0:
        raise SystemExit("학습 또는 검증 프레임이 없습니다. --val-sessions를 확인하세요.")
    print("학습 클래스별:", dict(zip(names, np.bincount(tr.y, minlength=len(names)).tolist())))
    print("검증 클래스별:", dict(zip(names, np.bincount(va.y, minlength=len(names)).tolist())))

    model, mean, std = train(tr, va, args.epochs, args.lr, args.hidden, args.offset_aug)
    preds = {"A 규칙 (MediaPipe만)": rule_predict(va.X), "B 수치 MLP": mlp_probs(model, mean, std, va.X).argmax(1)}
    for title, pred in preds.items():
        print("\n" + format_cm(confusion(va.y, pred, len(names)), f"검증 · {title}", names))
        print("하위 태그별 오경보 (영역 밖·졸음·하품으로 판정한 비율)")
        print("\n".join(alarm_lines(va, pred)))

    save_face_mlp(args.out, model, mean, std, args.hidden)
    print(f"\n저장: {args.out}  (이미지 모델과의 비교는 evaluation.face_compare)")


if __name__ == "__main__":
    main()