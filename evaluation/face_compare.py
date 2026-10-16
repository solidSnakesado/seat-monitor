"""얼굴 판정 방식 비교 (같은 프레임, 같은 지표): A MediaPipe만 / B 수치 / C 이미지 / D 수치+이미지 / E 결과 결합

프로젝트 루트(~/dev_ws/mediaPipe)에서 실행:
  python -m evaluation.face_compare --val-sessions g0 --dir models/face/check     # 검증 세션으로 비교
  python -m evaluation.face_compare --test --dir models/face/v1                   # 테스트 세션 (마지막에 한 번만)

--dir 안의 mlp.pt(B), cnn.pt(C), fusion.pt(D)를 읽는다. 없는 모델은 건너뛴다.
E = (1 − w) × B + w × C (--img-weight w, B·C가 모두 있을 때).
모든 방식을 얼굴 이미지가 있는 프레임(3프레임마다)으로 맞춰 평가한다.
결과: 화면 출력 + results/compare_face_날짜_시간.md (results/**/compare_*.md는 git 제외)
"""
import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from collect.face_guide import FACE_CLASSES, FACE_KO, SUBTAG_KO
from evaluation.eval_metrics import balanced_acc, confusion, format_cm, macro_f1, recalls
from learning.face_dataset import false_alarms, load_face_splits, rule_predict
from learning.face_image_train import face_loader, load_face_image_model
from learning.face_train import load_face_mlp
from learning.image_model import image_probs
from learning.posture_model import mlp_probs


def parse_args():
    root = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser(description="얼굴 판정 방식 비교")
    ap.add_argument("--data", default=str(root / "data" / "face"))
    ap.add_argument("--dir", default=str(root / "models" / "face"), help="mlp.pt, cnn.pt, fusion.pt가 있는 폴더")
    ap.add_argument("--val-sessions", nargs="+", default=None, help="비교에 쓸 검증 세션 (파일 이름 일부)")
    ap.add_argument("--test", action="store_true", help="검증 대신 테스트 세션(_test)으로 비교")
    ap.add_argument("--img-weight", type=float, default=0.5, help="E 결과 결합의 이미지 비중 w")
    ap.add_argument("--out", default=str(root / "results"))
    args = ap.parse_args()
    if not args.test and not args.val_sessions:
        ap.error("--val-sessions 또는 --test 중 하나가 필요합니다")
    return args


def predict_all(split, model_dir, w, device):
    """→ {방식 이름: 확률 또는 예측 (N,)}. A는 예측, 나머지는 확률"""
    out = {"A MediaPipe만 (규칙)": rule_predict(split.X)}
    probs = {}
    p = Path(model_dir) / "mlp.pt"
    if p.exists():
        model, mean, std = load_face_mlp(p)
        probs["B 수치 MLP"] = mlp_probs(model, mean, std, split.X)
    for fname, title in (("cnn.pt", "C 이미지 CNN"), ("fusion.pt", "D 수치+이미지 융합")):
        p = Path(model_dir) / fname
        if p.exists():
            model, _, mean, std = load_face_image_model(p, device)
            probs[title] = image_probs(model, face_loader(split, mean, std, False, 64, 2, device=device), device)
    if "B 수치 MLP" in probs and "C 이미지 CNN" in probs:
        probs[f"E 결과 결합 (이미지 {w:.1f})"] = (1 - w) * probs["B 수치 MLP"] + w * probs["C 이미지 CNN"]
    out.update({k: v.argmax(1) for k, v in probs.items()})
    return out


def report(split, preds, title):
    names = [FACE_KO[c] for c in FACE_CLASSES]
    n = len(names)
    lines = [f"# 얼굴 판정 비교 — {title}", "",
             f"프레임 {len(split)} (얼굴 이미지가 있는 프레임), 세션: {', '.join(dict.fromkeys(split.sessions))}", "",
             "## 요약", "", "| 방식 | 균형 정확도 | macro F1 | " + " | ".join(f"{c} 재현율" for c in names) + " |",
             "|---|---|---|" + "---|" * n]
    for k, pred in preds.items():
        cm = confusion(split.y, pred, n)
        rec = " | ".join("-" if np.isnan(r) else f"{r:.2f}" for r in recalls(cm))
        lines.append(f"| {k} | {balanced_acc(cm):.3f} | {macro_f1(cm):.3f} | {rec} |")
    tags = list(false_alarms(split, next(iter(preds.values()))))
    lines += ["", "## 앞 보기 구간 오경보율 (영역 밖·졸음·하품으로 판정한 비율)", "",
              "| 방식 | " + " | ".join(SUBTAG_KO[t] for t in tags) + " |", "|---|" + "---|" * len(tags)]
    for k, pred in preds.items():
        fa = false_alarms(split, pred)
        lines.append(f"| {k} | " + " | ".join(f"{fa[t][1]:.1%}" for t in tags) + " |")
    lines += ["", "## 혼동행렬", ""]
    for k, pred in preds.items():
        lines += ["```", format_cm(confusion(split.y, pred, n), k, names), "```", ""]
    return "\n".join(lines)


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _, val, test = load_face_splits(args.data, [] if args.test else args.val_sessions)
    split = (test if args.test else val).with_images()
    if len(split) == 0:
        raise SystemExit("비교할 프레임이 없습니다 (세션 이름 또는 _test 세션 확인)")
    missing = split.missing_images()
    if missing:
        raise SystemExit(f"얼굴 이미지 {len(missing)}장이 없습니다 (예: {missing[0]})")
    title = "테스트" if args.test else f"검증 ({' '.join(args.val_sessions)})"
    preds = predict_all(split, args.dir, args.img_weight, device)
    text = report(split, preds, title)
    print(text)
    Path(args.out).mkdir(parents=True, exist_ok=True)
    out = Path(args.out) / f"compare_face_{datetime.now():%Y%m%d_%H%M%S}.md"
    out.write_text(text, encoding="utf-8")
    print(f"\n저장: {out}")


if __name__ == "__main__":
    main()