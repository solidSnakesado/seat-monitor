"""이미지 비중별 성능 곡선: B(수치)와 C(이미지)의 확률을 섞는 비중을 바꿔 가며 테스트 점수 계산

  결합 확률 = (1 - w) × B 확률 + w × C 확률     (w = 이미지 비중, 0 = B만, 1 = C만, 0.5 = 지금의 E)
  F2 각도 보정(목 꺾임)을 적용한 결과도 함께 표시한다.
  --cnn에 모델을 여러 개 주면 C 확률 = 이미지 모델들의 확률 평균 (이미지 앙상블).
  모델마다 학습한 이미지 종류(images_cropmask, images_sil 등)로 각각 평가한다.

재학습 없이 저장된 모델로 계산한다. 테스트 열은 이미지가 어디까지 기여하는지 보는 분석용.
(테스트 세트로 비중을 고르면 그 테스트는 더 이상 공정한 확인용이 아니다)
비중은 --val-session으로 지정한 검증 세션 열을 보고 고른다 (모델도 그 세션을 빼고 학습한 것이어야 함).

프로젝트 루트에서 실행:
  python -m evaluation.weight_curve
  python -m evaluation.weight_curve --mlp models/v6/posture_mlp.pt --cnn models/v6/cnn_cropmask.pt --val-session round_stripe
  python -m evaluation.weight_curve --mlp models/v6/posture_mlp.pt --cnn models/v6/cnn_cropmask_reg.pt models/v6/cnn_sil.pt --val-session round_stripe
"""
import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from evaluation.compare import angle_override
from evaluation.eval_metrics import balanced_acc, confusion, recalls
from learning.image_model import FrameDataset, image_probs, load_image_model
from learning.labels import DEFAULT, SCHEMES, classes, convert_probs
from learning.posture_dataset import load_splits
from learning.posture_model import load_mlp, mlp_probs


def parse_args():
    root = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser(description="이미지 비중별 성능 곡선")
    ap.add_argument("--data", nargs="+", default=[str(root / d) for d in ("data", "eval/new_env", "eval/final1")],
                    help="테스트 세션이 있는 폴더들 (data는 10/14 테스트 세션)")
    ap.add_argument("--mlp", default=str(root / "models" / "v5" / "posture_mlp_off10.pt"))
    ap.add_argument("--cnn", nargs="+", default=[str(root / "models" / "v5" / "cnn_cropmask.pt")],
                    help="이미지 모델 (여러 개면 확률 평균)")
    ap.add_argument("--labels", choices=list(SCHEMES), default=DEFAULT)
    ap.add_argument("--steps", type=int, default=11, help="비중 개수 (11이면 0, 0.1, ..., 1.0)")
    ap.add_argument("--val-session", default="", help="data/ 안에서 통째로 검증에 쓴 세션 (예: round_stripe)")
    args = ap.parse_args()
    args.train_dir = str(root / "data")
    return args


def split_probs(name, split, mlp, cnns, sc, device):
    """Split → (정답, 수치 X, B 확률, C 확률 = 이미지 모델들의 평균). 이미지가 없으면 None"""
    if len(split) == 0:
        return None
    m, mean, std, mcls = mlp
    pb = convert_probs(mlp_probs(m, mean, std, split.X), mcls, sc)
    pcs = []
    for model, _, imean, istd, icls, idir in cnns:
        isplit = split.with_image_dir(idir)
        if isplit.missing_images():
            print(f"{name}: {idir} 이미지가 없어 제외합니다 (python -m learning.crop_images 필요)")
            return None
        loader = DataLoader(FrameDataset(isplit, imean, istd, train=False), batch_size=128)
        pcs.append(convert_probs(image_probs(model, loader, device), icls, sc))
    return split.y, split.X, pb, np.mean(pcs, axis=0)


def score(y, pred, n):
    cm = confusion(y, pred, n)
    return balanced_acc(cm), recalls(cm)[0]          # (균형 정확도, 바른 자세 재현율)


def print_table(title, cols, rows, n_val):
    """평균은 테스트 열만 (검증 열은 비중을 고르는 기준이라 따로 봄)"""
    print(f"\n[{title}]  각 칸: 균형 정확도 (바른 자세 재현율)")
    print("  w     " + "".join(f"{c:>16s}" for c in cols) + f"{'테스트 평균':>10s}")
    for w, cells in rows:
        mean = np.mean([b for b, _ in cells[n_val:]]) if len(cells) > n_val else float("nan")
        print(f" {w:4.2f}  " + "".join(f"{b:9.3f} ({g:.2f})" for b, g in cells) + f"{mean:10.3f}")


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sc, names = args.labels, classes(args.labels)
    mlp = load_mlp(args.mlp)
    cnns = [load_image_model(p, device) for p in args.cnn]
    print(f"B: {args.mlp}\nC: " + " + ".join(f"{p} ({c[5]})" for p, c in zip(args.cnn, cnns)))

    splits = {}
    if args.val_session:
        _, val, _ = load_splits(args.train_dir, images_only=True, scheme=sc, verbose=False,
                                val_sessions=[args.val_session])
        splits[f"검증:{args.val_session}"[:16]] = val
    for d in args.data:
        splits[Path(d).name] = load_splits(d, images_only=True, scheme=sc, verbose=False)[2]
    tests = {k: r for k, s in splits.items() if (r := split_probs(k, s, mlp, cnns, sc, device)) is not None}
    if not tests:
        raise SystemExit("평가할 데이터가 없습니다.")
    n_val = int(any(k.startswith("검증:") for k in tests))

    raw, fixed = [], []
    for w in np.linspace(0, 1, args.steps):
        r_cells, f_cells = [], []
        for y, X, pb, pc in tests.values():
            pred = ((1 - w) * pb + w * pc).argmax(1)
            r_cells.append(score(y, pred, len(names)))
            f_cells.append(score(y, angle_override(pred, X, names)["F2 각도 보정(목 꺾임)"], len(names)))
        raw.append((w, r_cells))
        fixed.append((w, f_cells))
    print_table("결합만 (E)", list(tests), raw, n_val)
    print_table("결합 + F2 각도 보정", list(tests), fixed, n_val)


if __name__ == "__main__":
    main()