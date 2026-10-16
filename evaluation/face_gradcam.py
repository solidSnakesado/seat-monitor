"""얼굴 Grad-CAM: 얼굴 이미지 모델이 판정할 때 얼굴 crop의 어느 부분을 근거로 삼았는지 히트맵으로 확인

프로젝트 루트(~/dev_ws/mediaPipe)에서 실행:
  python -m evaluation.face_gradcam --val-sessions val \\
      --models models/face/v1/cnn.pt models/face/v1/cnn_ls_erase.pt
  python -m evaluation.face_gradcam --val-sessions val --models ... --per-group 4

그룹(화면, 강의 공간, 책상, 영역 밖, 졸음, 하품, 잦은 깜빡임, 물 마시기)마다 N장을 고른다.
그룹 안에서는 어느 모델이든 틀린 프레임을 절반까지 먼저 넣고, 나머지는 맞힌 프레임을 시간 순서로 고르게 넣는다.
결과: eval/face_gradcam/face_gradcam_<검증 세션>.jpg (얼굴 이미지 포함 → git 제외 폴더)
  행 = 프레임, 열 = 원본(그룹 / 정답) + 모델별 히트맵 (예측과 확률, 맞으면 초록 / 틀리면 빨강)
히트맵은 MobileNetV3 마지막 특징 지도(128×128 입력이면 4×4칸)를 키운 것이라 대략적인 위치만 나타낸다.
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from collect.face_guide import FACE_CLASSES
from evaluation.gradcam import GradCAM
from learning.face_dataset import load_face_splits
from learning.face_image_train import FaceImageDataset, face_loader, load_face_image_model
from learning.image_model import image_probs

CELL, BAR = 128, 14
GREEN, RED, WHITE, GRAY = (0, 200, 0), (0, 0, 230), (255, 255, 255), (60, 60, 60)
# (그룹 이름, 조건) — 화면 글자는 OpenCV 기본 글꼴이라 영어 약어
GROUPS = (("screen", lambda c, t: t == "screen"), ("lecture", lambda c, t: t == "lecture"),
          ("desk", lambda c, t: c == "desk"), ("away", lambda c, t: c == "away"),
          ("drowsy", lambda c, t: c == "drowsy"), ("yawn", lambda c, t: c == "yawn"),
          ("blink", lambda c, t: t == "blink"), ("drink", lambda c, t: t == "drink"))


def parse_args():
    root = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser(description="얼굴 이미지 모델 Grad-CAM")
    ap.add_argument("--data", default=str(root / "data" / "face"))
    ap.add_argument("--val-sessions", nargs="+", required=True, help="검증 세션 (파일 이름 일부)")
    ap.add_argument("--models", nargs="+", required=True, help="이미지 모델 (cnn) 파일들, 열 순서대로")
    ap.add_argument("--per-group", type=int, default=3, help="그룹마다 고를 프레임 수")
    ap.add_argument("--out", default=str(root / "eval" / "face_gradcam"))
    return ap.parse_args()


def spread(idx, n):
    """시간 순서에서 고르게 n개"""
    idx = np.asarray(idx)
    return idx[np.linspace(0, len(idx) - 1, min(n, len(idx))).astype(int)].tolist() if len(idx) else []


def pick_frames(split, wrong_any, per_group):
    """→ [(그룹 이름, 프레임 번호)]"""
    picks = []
    for name, cond in GROUPS:
        idx = [i for i, (y, t) in enumerate(zip(split.y, split.subtags)) if cond(FACE_CLASSES[y], t)]
        wrong = [i for i in idx if wrong_any[i]]
        right = [i for i in idx if not wrong_any[i]]
        chosen = spread(wrong, (per_group + 1) // 2)
        chosen += spread(right, per_group - len(chosen))
        chosen += spread([i for i in wrong if i not in chosen], per_group - len(chosen))
        picks += [(name, i) for i in sorted(chosen)]
    return picks


def bar(text, color=WHITE, width=CELL):
    b = np.zeros((BAR, width, 3), np.uint8)
    cv2.putText(b, text, (3, 11), cv2.FONT_HERSHEY_SIMPLEX, 0.36, color, 1, cv2.LINE_AA)
    return b


def cell(path, cam=None):
    img = cv2.imread(path)
    img = np.zeros((CELL, CELL, 3), np.uint8) if img is None else cv2.resize(img, (CELL, CELL))
    if cam is None:
        return img
    heat = cv2.applyColorMap((cv2.resize(cam, (CELL, CELL)) * 255).astype(np.uint8), cv2.COLORMAP_JET)
    return cv2.addWeighted(img, 0.55, heat, 0.45, 0)


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _, val, _ = load_face_splits(args.data, args.val_sessions)
    split = val.with_images()
    if len(split) == 0:
        raise SystemExit("검증 프레임이 없습니다 (--val-sessions 확인)")

    loaded = []
    for path in args.models:
        model, kind, mean, std = load_face_image_model(path, device)
        if kind != "cnn":
            print(f"{path}: 융합(fusion) 모델은 제외합니다 (이미지만 쓰는 cnn만 비교)")
            continue
        pred = image_probs(model, face_loader(split, mean, std, False, 64, 2, device=device), device).argmax(1)
        loaded.append((Path(path).stem, GradCAM(model), FaceImageDataset(split, mean, std, False), pred))
    if not loaded:
        raise SystemExit("비교할 cnn 모델이 없습니다")
    wrong_any = np.any([pred != split.y for *_, pred in loaded], axis=0)
    picks = pick_frames(split, wrong_any, args.per_group)

    header = [bar("group / true", GRAY)] + [bar(name[:20], GRAY) for name, *_ in loaded]
    rows = [np.hstack(header)]
    for group, i in picks:
        true = FACE_CLASSES[split.y[i]]
        cols = [np.vstack([bar(f"{group} / {true}"), cell(split.images[i])])]
        for _, gradcam, ds, _ in loaded:
            img, num, _ = ds[i]
            cam, cls, p = gradcam(img.to(device), num.to(device))
            color = GREEN if cls == split.y[i] else RED
            cols.append(np.vstack([bar(f"{FACE_CLASSES[cls]} {p:.2f}", color), cell(split.images[i], cam)]))
        rows.append(np.hstack(cols))

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = "_".join(args.val_sessions)
    out = out_dir / f"face_gradcam_{tag}.jpg"
    cv2.imwrite(str(out), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 92])
    wrong_n = sum(bool(wrong_any[i]) for _, i in picks)
    print(f"프레임 {len(picks)}장 (그룹 {len(GROUPS)}개 × 최대 {args.per_group}, 틀린 프레임 {wrong_n}장 포함)")
    print(f"저장: {out}")


if __name__ == "__main__":
    main()