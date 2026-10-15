"""네 방식 + 결과 융합 비교 (같은 테스트 프레임, 같은 지표)

  A 규칙       : MediaPipe 특징 + 임계값            (학습 없음)
  B 수치 MLP   : MediaPipe 특징 5개 → MLP           (models/posture_mlp.pt)
  C 이미지 CNN : 이미지 → MobileNetV3              (models/cnn.pt)
  D 융합       : 이미지 + 수치 → 하나의 모델        (models/fusion.pt)
  E 결과 융합  : B와 C의 확률 평균                  (B, C가 있으면 자동)
  F 각도 보정  : E의 판정을 귀 기울기 각도로 보정    (E가 있으면 자동)
    F1: E가 바른 자세라고 했는데 귀 기울기가 10° 이상이면 → 고개 기울임
    F2: 목이 꺾인 정도(귀 기울기 - 어깨 기울기)가 10° 이상이면 → 고개 기울임

프로젝트 루트(~/dev_ws/mediaPipe)에서 실행:
  python -m evaluation.compare                  # 라벨 merged (거북목·구부정 합침, 기본)
  python -m evaluation.compare --labels fine    # 5개 라벨 그대로
5개 라벨로 학습한 모델도 merged로 평가할 수 있다 (거북목·구부정 확률을 더함).
결과는 화면과 results/compare_날짜_시간.md 에 저장된다.
"""
import argparse
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from common.posture_judge import Thresholds
from evaluation.eval_metrics import balanced_acc, confusion, format_cm, macro_f1, recalls, rule_predict
from learning.image_model import FrameDataset, image_probs, load_image_model
from learning.labels import DEFAULT, SCHEMES, classes, convert_probs, ko_names
from learning.posture_dataset import load_splits
from learning.posture_model import INPUTS, load_mlp, mlp_probs

EAR, SHOULDER = INPUTS.index("ear_tilt"), INPUTS.index("shoulder_tilt")


def parse_args():
    root = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser(description="규칙 / 수치 / 이미지 / 융합 비교")
    ap.add_argument("--data", default=str(root / "data"))
    ap.add_argument("--mlp", default=str(root / "models" / "posture_mlp.pt"))
    ap.add_argument("--cnn", default=str(root / "models" / "cnn.pt"))
    ap.add_argument("--fusion", default=str(root / "models" / "fusion.pt"))
    ap.add_argument("--out", default=str(root / "results"))
    ap.add_argument("--labels", choices=list(SCHEMES), default=DEFAULT,
                    help="라벨 체계: merged = 거북목·구부정 합침 (기본), fine = 5개 그대로")
    ap.add_argument("--timing-frames", type=int, default=50, help="처리 시간 측정 프레임 수")
    return ap.parse_args()


def time_per_frame(fn, n):
    """프레임 1장씩 n번 처리한 평균 시간(ms). MediaPipe 검출 시간은 제외 (모든 방식 공통)"""
    fn(0)                                     # 첫 호출(초기화) 제외
    t0 = time.perf_counter()
    for i in range(n):
        fn(i)
    return (time.perf_counter() - t0) / n * 1000


def file_kb(path):
    return Path(path).stat().st_size / 1024 if path and Path(path).exists() else 0.0


def angle_override(pred, X, names):
    """E 예측 + 귀 기울기 각도 보정 → {방식 이름: 예측}. 고개 기울임은 각도로 직접 측정되는 자세라서"""
    good, head, deg = names.index("good"), names.index("head_tilt"), Thresholds().head_tilt_deg
    ear, neck = np.abs(X[:, EAR]), np.abs(X[:, EAR] - X[:, SHOULDER])
    return {"F1 각도 보정(바른→기울임)": np.where((pred == good) & (ear >= deg), head, pred),
            "F2 각도 보정(목 꺾임)": np.where(neck >= deg, head, pred)}


def converted(name, probs, model_classes, scheme):
    p = convert_probs(probs, model_classes, scheme)
    if p is None:
        print(f"{name}: 모델 라벨 {model_classes}를 {scheme} 체계로 바꿀 수 없어 제외합니다")
    return p


def evaluate(args, test, notes):
    """→ {방식 이름: (확률 또는 None, 예측, ms/프레임, 모델 KB)}. notes에 모델·입력 정보를 남긴다"""
    n, sc = min(args.timing_frames, len(test)), args.labels
    ms = time_per_frame(lambda i: rule_predict(test.X[i:i + 1], scheme=sc), n)
    res = {"A 규칙": (None, rule_predict(test.X, scheme=sc), ms, 0.0)}

    probs = {}
    if args.mlp and Path(args.mlp).is_file():
        m, mean, std, mcls = load_mlp(args.mlp)
        p = converted("B 수치 MLP", mlp_probs(m, mean, std, test.X), mcls, sc)
        if p is not None:
            notes.append(f"B 수치 MLP: {args.mlp}")
            probs["B 수치 MLP"] = p
            ms = time_per_frame(lambda i: mlp_probs(m, mean, std, test.X[i:i + 1]), n)
            res["B 수치 MLP"] = (p, p.argmax(1), ms, file_kb(args.mlp))

    for name, path in (("C 이미지 CNN", args.cnn), ("D 융합", args.fusion)):
        if not path or not Path(path).is_file():
            continue
        model, _, mean, std, icls, idir = load_image_model(path, "cpu")
        split = test.with_image_dir(idir)             # 모델이 학습한 이미지 종류로 평가 (원본 / crop)
        if split.missing_images():
            print(f"{name}: {idir} 이미지가 없어 제외합니다 (python -m learning.crop_images 실행 필요)")
            continue
        notes.append(f"{name}: {path} (이미지 {idir})")
        ds = FrameDataset(split, mean, std, train=False)
        p = converted(name, image_probs(model, DataLoader(ds, batch_size=64)), icls, sc)
        if p is None:
            continue
        probs[name] = p

        def one(i, model=model, ds=ds):
            img, num, _ = ds[i]
            with torch.no_grad():
                model(img[None], num[None])
        res[name] = (p, p.argmax(1), time_per_frame(one, n), file_kb(path))

    if "B 수치 MLP" in probs and "C 이미지 CNN" in probs:
        p = (probs["B 수치 MLP"] + probs["C 이미지 CNN"]) / 2
        ms = res["B 수치 MLP"][2] + res["C 이미지 CNN"][2]
        res["E 결과 융합(B+C)"] = (p, p.argmax(1), ms, res["B 수치 MLP"][3] + res["C 이미지 CNN"][3])
        for name, pred in angle_override(p.argmax(1), test.X, classes(sc)).items():
            res[name] = (None, pred, ms, res["E 결과 융합(B+C)"][3])
    return res


def summary_table(test, res, ko):
    head = ("| 방식 | 균형 정확도 | macro F1 | " + " | ".join(f"{k} 재현율" for k in ko)
            + " | ms/프레임 (CPU) | 모델 크기 |")
    lines = [head, "|" + "---|" * (5 + len(ko))]
    for name, (_, pred, ms, kb) in res.items():
        cm = confusion(test.y, pred, len(ko))
        rec = " | ".join("-" if np.isnan(r) else f"{r:.2f}" for r in recalls(cm))
        size = "-" if kb == 0 else f"{kb:,.0f} KB"
        lines.append(f"| {name} | {balanced_acc(cm):.3f} | {macro_f1(cm):.3f} | {rec} | {ms:.2f} | {size} |")
    return "\n".join(lines)


def intensity_table(test, res):
    """강도별 재현율 (나쁜 자세를 맞힌 비율) + 바른 자세 오경보율 (바른 자세를 나쁜 자세로 판정한 비율)"""
    inten = np.array(test.intensity)
    if not np.isin(inten, ["mild", "mid", "severe"]).any():
        return "(강도 기록이 없는 데이터입니다. collect.posture_collect 안내 모드로 수집하면 표시됩니다)"
    groups = [(k, (inten == k) & (test.y != 0)) for k in ("mild", "mid", "severe")]
    good = test.y == 0
    lines = ["| 방식 | 약하게 재현율 | 중간 재현율 | 심하게 재현율 | 바른 자세 오경보율 |",
             "|---|---|---|---|---|"]
    for name, (_, pred, _, _) in res.items():
        cells = [f"{(pred[m] == test.y[m]).mean():.2f} (n={m.sum()})" if m.any() else "-" for _, m in groups]
        fp = f"{(pred[good] != 0).mean():.2f}" if good.any() else "-"
        lines.append(f"| {name} | " + " | ".join(cells) + f" | {fp} |")
    return "\n".join(lines)


def main():
    args = parse_args()
    torch.set_num_threads(max(1, torch.get_num_threads()))
    names, ko = classes(args.labels), ko_names(args.labels)
    _, _, test = load_splits(args.data, images_only=True, scheme=args.labels)
    if len(test) == 0:                       # 수치만 수집한 경우: 전체 프레임으로 A, B만 비교
        print("이미지가 있는 프레임이 없어 전체 프레임으로 비교합니다 (이미지 모델 C, D, E 제외)")
        _, _, test = load_splits(args.data, images_only=False, scheme=args.labels)
        args.cnn = args.fusion = ""
    if len(test) == 0:
        raise SystemExit("테스트할 프레임이 없습니다.")
    counts = dict(zip(names, np.bincount(test.y, minlength=len(names)).tolist()))
    print(f"테스트 프레임 {len(test)}개: {counts}\n")

    notes = []
    res = evaluate(args, test, notes)
    table = summary_table(test, res, ko)
    by_level = intensity_table(test, res)
    cms = "\n\n".join(format_cm(confusion(test.y, pred, len(names)), name, names)
                      for name, (_, pred, _, _) in res.items())
    print(table + "\n\n[강도별]\n" + by_level + "\n\n" + cms)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"compare_{datetime.now():%Y%m%d_%H%M%S}.md"
    models = "".join(f"- {n}\n" for n in notes)
    path.write_text(f"# 자세 판정 방식 비교\n\n- 데이터: {args.data}\n- 라벨 체계: {args.labels}\n"
                    f"- 테스트 프레임: {len(test)} {counts}\n{models}"
                    f"- 처리 시간: 판정 단계만 (MediaPipe 검출 제외), CPU, 프레임 1장씩\n\n{table}\n\n"
                    f"## 강도별\n\n{by_level}\n\n"
                    f"```\n{cms}\n```\n", encoding="utf-8")
    print(f"\n저장: {path}")


if __name__ == "__main__":
    main()