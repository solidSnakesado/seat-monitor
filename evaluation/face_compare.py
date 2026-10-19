"""얼굴 판정 방식 비교 (같은 프레임, 같은 지표): A MediaPipe만 / B 수치 / C 이미지 / D 수치+이미지 / E 결과 결합 / T 시계열 / F

프로젝트 루트(~/dev_ws/mediaPipe)에서 실행:
  python -m evaluation.face_compare --val-sessions g0 --dir models/face/check     # 검증 세션으로 비교
  python -m evaluation.face_compare --test --dir models/face/v1                   # 테스트 세션 (마지막에 한 번만)

--dir 안의 mlp.pt(B), cnn.pt(C), fusion.pt(D)를 읽는다. 없는 모델은 건너뛴다.
--cnn / --fusion으로 모델 파일을 직접 고를 수 있고, 여러 개를 주면 확률을 평균한다 (seed 앙상블):
  python -m evaluation.face_compare --val-sessions val --dir models/face/v1 \
      --cnn models/face/v1/cnn_ls_erase.pt models/face/v1/cnn_ls_erase_s1.pt --img-weight 0.4 --note "C 스무딩+가리기 2개"
E = (1 − w) × B + w × C (--img-weight w, B·C가 모두 있을 때).
--seq: T 시계열 모델 (learning.face_seq_train), 여러 개면 확률 평균 (seed 앙상블). F = (1 − w) × T + w × C (E에서 B 대신 T).
--zone-gate: E·F에 영역 연결 적용 (learning.face_dataset.zone_gate). 히스토리의 E 열에 그대로 기록되므로 --note에 적는다.
--gate-yaw: 영역 연결 허용 범위 (기본 5°). 앞쪽 구간 밖으로 이 값까지는 '앞 보기'로 본다.
--events: 지속 시간 필터를 적용한 이벤트 단위 지표 표를 보고서에 추가 (evaluation.face_events).
모든 방식을 얼굴 이미지가 있는 프레임(3프레임마다)으로 맞춰 평가한다.
결과: 화면 출력 + results/compare_face_날짜_시간.md (results/**/compare_*.md는 git 제외)
히스토리: 실행할 때마다 results/face_history.md에 한 줄을 추가한다 (evaluation.face_history, git에 포함).
  --note로 단계 이름을 적는다 (예: --note "메인 5세션", --note "C freeze 9")
"""
import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from collect.face_guide import FACE_CLASSES, FACE_KO, SUBTAG_KO
from evaluation.eval_metrics import balanced_acc, confusion, format_cm, macro_f1, recalls
from evaluation.face_events import events_report
from evaluation.face_history import append_history
from learning.face_dataset import false_alarms, load_face_splits, rule_predict, zone_gate
from learning.face_image_train import face_loader, load_face_image_model
from learning.face_seq_train import load_seq_model, seq_probs
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
    ap.add_argument("--cnn", nargs="+", default=None, help="C 이미지 모델 파일 (여러 개면 평균, 기본: --dir/cnn.pt)")
    ap.add_argument("--fusion", nargs="+", default=None, help="D 융합 모델 파일 (여러 개면 평균, 기본: --dir/fusion.pt)")
    ap.add_argument("--seq", nargs="+", default=None, help="T 시계열 모델 파일 (여러 개면 평균, learning.face_seq_train)")
    ap.add_argument("--zone-gate", action="store_true",
                    help="E에서 고개가 화면·강의 공간 범위 안이면 이미지의 '영역 밖' 확률을 '앞 보기'로 옮김")
    ap.add_argument("--gate-yaw", type=float, default=5.0,
                    help="영역 연결 허용 범위: 앞쪽 구간 밖 거리가 이 값(°) 이하면 '앞 보기'로 본다")
    ap.add_argument("--events", action="store_true", help="이벤트 단위 지표(지속 시간 필터 적용) 추가")
    ap.add_argument("--img-weight", type=float, default=0.5, help="E 결과 결합의 이미지 비중 w")
    ap.add_argument("--out", default=str(root / "results"))
    ap.add_argument("--note", default="", help="히스토리에 남길 단계 이름 (예: 확인용 1세션)")
    args = ap.parse_args()
    if not args.test and not args.val_sessions:
        ap.error("--val-sessions 또는 --test 중 하나가 필요합니다")
    return args


def image_model_probs(split, paths, device):
    """이미지·융합 모델 여러 개의 확률 평균 (seed 앙상블)"""
    out = []
    for p in paths:
        model, _, mean, std = load_face_image_model(p, device)
        out.append(image_probs(model, face_loader(split, mean, std, False, 64, 2, device=device), device))
    return np.mean(out, axis=0)


def seq_ensemble_probs(split, paths, device):
    """T 시계열 모델 여러 개의 확률 평균 (seed 앙상블, 프레임 순서 유지)"""
    out = []
    for p in paths:
        model, mean, std, win = load_seq_model(p, device)
        out.append(seq_probs(model, split, mean, std, win, device))
    return np.mean(out, axis=0)


def model_paths(given, model_dir, fname):
    """직접 지정한 파일들, 없으면 --dir의 기본 파일 (있을 때만)"""
    if given:
        missing = [p for p in given if not Path(p).exists()]
        if missing:
            raise SystemExit(f"모델 파일이 없습니다: {missing[0]}")
        return list(given)
    p = Path(model_dir) / fname
    return [str(p)] if p.exists() else []


def predict_all(split, model_dir, w, device, cnn=None, fusion=None, gate=False, seq=None, gate_yaw=5.0):
    """→ ({방식 이름: 예측 (N,)}, 사용한 모델 설명)"""
    out = {"A MediaPipe만 (규칙)": rule_predict(split.X)}
    probs, used = {}, []
    p = Path(model_dir) / "mlp.pt"
    if p.exists():
        model, mean, std = load_face_mlp(p)
        probs["B 수치 MLP"] = mlp_probs(model, mean, std, split.X)
        used.append(f"B = {p.name}")
    for paths, key, title in ((model_paths(cnn, model_dir, "cnn.pt"), "C", "C 이미지 CNN"),
                              (model_paths(fusion, model_dir, "fusion.pt"), "D", "D 수치+이미지 융합")):
        if paths:
            name = title + (f" ×{len(paths)}" if len(paths) > 1 else "")
            probs[name] = image_model_probs(split, paths, device)
            used.append(f"{key} = {', '.join(Path(x).name for x in paths)}")
    b = probs.get("B 수치 MLP")
    c = next((v for k, v in probs.items() if k.startswith("C ")), None)
    cg = (zone_gate(c, split.X, yaw_tol=gate_yaw) if gate else c) if c is not None else None
    tail = f", 영역 연결 {gate_yaw:g}°)" if gate else ")"
    if gate:
        used.append(f"영역 연결 허용 {gate_yaw:g}°")
    if b is not None and cg is not None:
        probs[f"E 결과 결합 (이미지 {w:.1f}{tail}"] = (1 - w) * b + w * cg
    if seq is not None:
        probs["T 시계열" + (f" ×{seq[2]}" if seq[2] > 1 else "")] = seq[0]
        used.append(f"T = {seq[1]}")
        if cg is not None:
            probs[f"F T·C 결합 (이미지 {w:.1f}{tail}"] = (1 - w) * seq[0] + w * cg
    out.update({k: v.argmax(1) for k, v in probs.items()})
    return out, "; ".join(used)


def report(split, preds, title, used=""):
    names = [FACE_KO[c] for c in FACE_CLASSES]
    n = len(names)
    lines = [f"# 얼굴 판정 비교 — {title}", "",
             f"프레임 {len(split)} (얼굴 이미지가 있는 프레임), 세션: {', '.join(dict.fromkeys(split.sessions))}",
             f"모델: {used or '-'}", "",
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
    train, val, test = load_face_splits(args.data, [] if args.test else args.val_sessions)
    full = test if args.test else val
    keep = [i for i, p in enumerate(full.images) if p]
    split = full.subset(keep)
    seq = None
    if args.seq:                                       # T는 이미지가 없는 프레임까지 이어서 봐야 하므로 전체에서 계산
        seq = (seq_ensemble_probs(full, model_paths(args.seq, args.dir, ""), device)[keep],
               ", ".join(Path(p).name for p in args.seq), len(args.seq))
    if len(split) == 0:
        raise SystemExit("비교할 프레임이 없습니다 (세션 이름 또는 _test 세션 확인)")
    missing = split.missing_images()
    if missing:
        raise SystemExit(f"얼굴 이미지 {len(missing)}장이 없습니다 (예: {missing[0]})")
    title = "테스트" if args.test else f"검증 ({' '.join(args.val_sessions)})"
    preds, used = predict_all(split, args.dir, args.img_weight, device, args.cnn, args.fusion, args.zone_gate, seq,
                              args.gate_yaw)
    text = report(split, preds, title, used)
    if args.events:
        text += "\n" + events_report(split, preds) + "\n"
    print(text)
    Path(args.out).mkdir(parents=True, exist_ok=True)
    out = Path(args.out) / f"compare_face_{datetime.now():%Y%m%d_%H%M%S}.md"
    out.write_text(text, encoding="utf-8")
    hist = Path(args.out) / "face_history.md"
    append_history(hist, args.note, train, split, "테스트" if args.test else "검증", preds, out.name)
    print(f"\n저장: {out}\n히스토리 추가: {hist}")


if __name__ == "__main__":
    main()