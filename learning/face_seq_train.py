"""얼굴 방식 T: 최근 30프레임의 수치 56개 → 시계열 모델 (1D-CNN 또는 GRU)

프로젝트 루트(~/dev_ws/mediaPipe)에서 실행:
  python -m learning.face_seq_train --val-sessions val --out models/face/v1/seq.pt               # 1D-CNN
  python -m learning.face_seq_train --val-sessions val --arch gru --out models/face/v1/seq_gru.pt

conv: 시간 방향 합성곱 2층(커널 5, 두 번째는 간격 2) → 시간 평균·최대 → 분류. 짧은 모양 변화(깜빡임)에 강함
gru : GRU 1층(은닉 64) → 마지막 상태 → 분류. 순서를 차례로 기억 (비교용)
분할은 learning.face_dataset.load_face_splits, 묶음은 learning.face_seq_dataset. 검증 점수가 가장 높은 epoch를 저장한다.
최종 비교는 evaluation.face_compare --seq 로 A~E와 같은 프레임에서 한다.
"""
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from collect.face_guide import FACE_CLASSES, FACE_KO
from evaluation.eval_metrics import balanced_acc, confusion, format_cm
from learning.face_dataset import NUM_INPUTS, load_face_splits
from learning.face_seq_dataset import DEFAULT_WIN, SeqDataset, angle_offset, num_stats
from learning.face_train import ANGLE_OFFSET, alarm_lines, class_weight_tensor


class SeqConv(nn.Module):
    def __init__(self, n_in=len(NUM_INPUTS), n_cls=len(FACE_CLASSES), hidden=64):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv1d(n_in, hidden, 5, padding=2), nn.ReLU(),
                                  nn.Conv1d(hidden, hidden, 5, padding=4, dilation=2), nn.ReLU())
        self.head = nn.Sequential(nn.Dropout(0.2), nn.Linear(hidden * 2, n_cls))

    def forward(self, x):                                   # x: (B, win, 56)
        h = self.conv(x.transpose(1, 2))                    # (B, hidden, win)
        return self.head(torch.cat([h.mean(2), h.amax(2)], 1))


class SeqGRU(nn.Module):
    def __init__(self, n_in=len(NUM_INPUTS), n_cls=len(FACE_CLASSES), hidden=64):
        super().__init__()
        self.gru = nn.GRU(n_in, hidden, batch_first=True)
        self.head = nn.Sequential(nn.Dropout(0.2), nn.Linear(hidden, n_cls))

    def forward(self, x):
        _, h = self.gru(x)
        return self.head(h[-1])


ARCHS = {"conv": SeqConv, "gru": SeqGRU}


def parse_args():
    root = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser(description="얼굴 시계열 모델 학습")
    ap.add_argument("--data", default=str(root / "data" / "face"))
    ap.add_argument("--out", default=str(root / "models" / "face" / "seq.pt"))
    ap.add_argument("--val-sessions", nargs="+", required=True, help="통째로 검증에 쓸 세션 (파일 이름 일부)")
    ap.add_argument("--arch", choices=list(ARCHS), default="conv")
    ap.add_argument("--win", type=int, default=DEFAULT_WIN, help="묶음 길이 (프레임, 약 24fps 기준 30 = 1.2초)")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--offset-aug", type=float, default=1.0, help="각도 오프셋 증강 배율 (0 = 사용 안 함)")
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args()


@torch.no_grad()
def seq_probs(model, split, mean, std, win, device="cpu"):
    """FaceSplit 전체(이미지 유무와 무관, 프레임 순서 유지) → (N, 클래스 수) 확률"""
    model.eval()
    out = []
    for xb, _ in DataLoader(SeqDataset(split, mean, std, win), batch_size=1024):
        out.append(torch.softmax(model(xb.to(device)), 1).cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, len(FACE_CLASSES)), np.float32)


def train(tr, va, args, device):
    n_cls = len(FACE_CLASSES)
    mean, std = num_stats(tr.X)
    loader = DataLoader(SeqDataset(tr, mean, std, args.win), batch_size=args.batch, shuffle=True)
    off = angle_offset(std, ANGLE_OFFSET, args.offset_aug).to(device)
    model = ARCHS[args.arch](hidden=args.hidden).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=class_weight_tensor(tr.y, n_cls).to(device))
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    best, best_state, best_ep = -1.0, None, 0
    for ep in range(1, args.epochs + 1):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            if args.offset_aug > 0:                          # 묶음 전체에 같은 오프셋
                xb = xb + (torch.rand(xb.shape[0], 1, xb.shape[2], device=device) * 2 - 1) * off
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
        score = balanced_acc(confusion(va.y, seq_probs(model, va, mean, std, args.win, device).argmax(1), n_cls))
        if score > best:
            best, best_ep = score, ep
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if ep == 1 or ep % 5 == 0:
            print(f"epoch {ep:3d}  loss {loss.item():.4f}  검증 균형 정확도 {score:.3f}")
    model.load_state_dict(best_state)
    print(f"선택: epoch {best_ep} (검증 균형 정확도 {best:.3f})")
    return model, mean, std


def save_seq_model(path, model, args, mean, std):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state": {k: v.cpu() for k, v in model.state_dict().items()}, "arch": args.arch,
                "win": args.win, "hidden": args.hidden, "mean": mean.tolist(), "std": std.tolist(),
                "classes": list(FACE_CLASSES), "inputs": list(NUM_INPUTS)}, path)


def load_seq_model(path, device="cpu"):
    """→ (model, mean, std, win)"""
    ck = torch.load(path, map_location=device)
    if tuple(ck["inputs"]) != NUM_INPUTS or tuple(ck["classes"]) != FACE_CLASSES:
        raise ValueError(f"{path}: 입력·클래스가 현재 코드와 다릅니다. 다시 학습하세요.")
    model = ARCHS[ck["arch"]](hidden=ck["hidden"])
    model.load_state_dict(ck["state"])
    model.to(device).eval()
    return model, np.array(ck["mean"], dtype=np.float32), np.array(ck["std"], dtype=np.float32), ck["win"]


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    names = [FACE_KO[c] for c in FACE_CLASSES]
    tr, va, _ = load_face_splits(args.data, args.val_sessions)
    if len(tr) == 0 or len(va) == 0:
        raise SystemExit("학습 또는 검증 프레임이 없습니다. --val-sessions를 확인하세요.")
    print(f"[{args.arch}] 묶음 {args.win}프레임, 학습 {len(tr)} / 검증 {len(va)} 프레임, device={device}")

    model, mean, std = train(tr, va, args, device)
    pred = seq_probs(model, va, mean, std, args.win, device).argmax(1)
    print("\n" + format_cm(confusion(va.y, pred, len(names)), f"검증 · T {args.arch}", names))
    print("하위 태그별 오경보 (영역 밖·졸음·하품으로 판정한 비율)")
    print("\n".join(alarm_lines(va, pred)))
    save_seq_model(args.out, model, args, mean, std)
    print(f"\n저장: {args.out}  (A~E와의 비교는 evaluation.face_compare --seq)")


if __name__ == "__main__":
    main()