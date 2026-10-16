"""얼굴 이미지 비중별 성능 곡선: E = (1 − w) × B(수치) + w × C(이미지) 의 w를 0~1로 바꿔 가며 계산

프로젝트 루트(~/dev_ws/mediaPipe)에서 실행:
  python -m evaluation.face_weight_curve --val-sessions val --dir models/face/v1
  python -m evaluation.face_weight_curve --val-sessions val --dir models/face/v1 --cnn models/face/v1/cnn_s1.pt ...
      # --cnn을 여러 개 주면 이미지 확률은 그 모델들의 평균 (seed 앙상블)

재학습 없이 저장된 모델의 확률만 섞는다. 검증 세션으로 w를 고르고, 테스트 세션은 쓰지 않는다.
결정한 w는 evaluation.face_compare --img-weight w --note ...로 다시 비교해 히스토리에 남긴다.
결과: 화면 출력 + results/compare_face_weight_날짜_시간.md (git 제외)
"""
import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from collect.face_guide import FACE_CLASSES
from evaluation.eval_metrics import balanced_acc, confusion, macro_f1, recalls
from learning.face_dataset import load_face_splits, zone_gate
from learning.face_image_train import face_loader, load_face_image_model
from learning.face_train import load_face_mlp
from learning.image_model import image_probs
from learning.posture_model import mlp_probs

ALARM = [FACE_CLASSES.index(c) for c in ("away", "drowsy", "yawn")]
AWAY = FACE_CLASSES.index("away")


def parse_args():
    root = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser(description="얼굴 이미지 비중 곡선")
    ap.add_argument("--data", default=str(root / "data" / "face"))
    ap.add_argument("--dir", default=str(root / "models" / "face"), help="mlp.pt, cnn.pt가 있는 폴더")
    ap.add_argument("--cnn", nargs="+", default=None, help="이미지 모델들 (기본: --dir의 cnn.pt)")
    ap.add_argument("--val-sessions", nargs="+", required=True, help="검증 세션 (파일 이름 일부)")
    ap.add_argument("--step", type=float, default=0.1)
    ap.add_argument("--zone-gate", action="store_true", help="이미지 확률에 영역 연결 적용 (face_compare와 같음)")
    ap.add_argument("--out", default=str(root / "results"))
    return ap.parse_args()


def front_alarm(split, pred):
    """앞 보기 구간(하위 태그 있는 front 프레임)을 영역 밖·졸음·하품으로 판정한 비율"""
    front = np.array([t != "" and FACE_CLASSES[y] == "front" for t, y in zip(split.subtags, split.y)])
    return float(np.isin(pred[front], ALARM).mean()) if front.any() else float("nan")


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _, val, _ = load_face_splits(args.data, args.val_sessions)
    split = val.with_images()
    if len(split) == 0:
        raise SystemExit("검증 프레임이 없습니다 (--val-sessions 확인)")

    model, mean, std = load_face_mlp(Path(args.dir) / "mlp.pt")
    pb = mlp_probs(model, mean, std, split.X)
    cnns = args.cnn or [str(Path(args.dir) / "cnn.pt")]
    pcs = []
    for path in cnns:
        m, _, nm, ns = load_face_image_model(path, device)
        pcs.append(image_probs(m, face_loader(split, nm, ns, False, 64, 2, device=device), device))
    pc = np.mean(pcs, axis=0)
    if args.zone_gate:
        pc = zone_gate(pc, split.X)
    conf_b, conf_c = pb.max(1).mean(), pc.max(1).mean()

    n = len(FACE_CLASSES)
    rows = []
    for w in np.round(np.arange(0, 1 + 1e-9, args.step), 2):
        pred = ((1 - w) * pb + w * pc).argmax(1)
        cm = confusion(split.y, pred, n)
        rows.append((w, balanced_acc(cm), macro_f1(cm), recalls(cm)[AWAY], front_alarm(split, pred)))
    best = max(rows, key=lambda r: r[1])

    gate = " · 영역 연결" if args.zone_gate else ""
    lines = [f"# 얼굴 이미지 비중 곡선 — 검증 ({' '.join(args.val_sessions)}){gate}", "",
             f"프레임 {len(split)}, 이미지 모델 {len(cnns)}개: {', '.join(Path(c).name for c in cnns)}",
             f"평균 최고 확률 (확신 정도): B {conf_b:.3f}, C {conf_c:.3f}", "",
             "| 이미지 비중 w | 균형 정확도 | macro F1 | 영역 밖 재현율 | 앞 보기 오경보 |",
             "|---|---|---|---|---|"]
    for w, ba, f1, rec, fa in rows:
        mark = " ◀ 최고" if w == best[0] else ""
        lines.append(f"| {w:.1f} | {ba:.3f}{mark} | {f1:.3f} | {rec:.2f} | {fa:.1%} |")
    lines += ["", f"균형 정확도 최고: w = {best[0]:.1f} ({best[1]:.3f}). "
              "w 0 = B만, w 1 = C만. 곡선이 평평하면 비중에 둔감(다이얼), 한곳에서 급변하면 스위치처럼 동작"]
    text = "\n".join(lines)
    print(text)
    Path(args.out).mkdir(parents=True, exist_ok=True)
    out = Path(args.out) / f"compare_face_weight_{datetime.now():%Y%m%d_%H%M%S}.md"
    out.write_text(text, encoding="utf-8")
    print(f"\n저장: {out}")


if __name__ == "__main__":
    main()