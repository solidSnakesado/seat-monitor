"""이미지 전처리: 저장된 프레임에서 사람 영역(머리·어깨)만 잘라 별도 폴더에 저장

이미지 모델이 배경·장소나 화면 속 위치를 외우지 않도록, 어깨 폭을 기준으로 상반신만 잘라낸다.
--mask를 주면 MediaPipe의 사람 윤곽(segmentation mask)으로 사람 밖을 검은색으로 칠한 뒤 자른다.
--silhouette를 주면 사람 윤곽만 흑백(사람 = 흰색, 배경 = 검은색)으로 저장한다. 옷의 색·무늬·질감이 모두 사라진다.
원본 이미지는 그대로 두고, 같은 이름으로 <CSV 폴더>/<폴더>/<세션>/000123.jpg 에 저장한다.

프로젝트 루트(~/dev_ws/mediaPipe)에서 실행:
  python -m learning.crop_images                   # crop만       → images_crop/
  python -m learning.crop_images --mask            # 배경 제거 + crop → images_cropmask/
  python -m learning.crop_images --silhouette      # 윤곽만 + crop    → images_sil/
  python -m learning.crop_images --roots data      # 특정 폴더만
이후 학습·비교: python -m learning.image_train --kind cnn --image-dir images_crop (또는 images_cropmask)
"""
import argparse
import csv
import math
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

NOSE, L_EAR, R_EAR, L_SH, R_SH = 0, 7, 8, 11, 12
OUT_W, OUT_H = 160, 120          # 원본 저장 크기와 같게 (학습 코드 변경 없이 사용)
WIDTH_K = 2.2                    # 잘라낼 너비 = 어깨 폭 × 2.2 (높이는 4:3 비율로 정해짐)
TOP_K = 1.1                      # 위쪽 끝 = 어깨 중심에서 어깨 폭 × 1.1 위 (머리까지 포함)
MIN_VIS = 0.5


def parse_args():
    root = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser(description="사람 영역 crop 이미지 생성")
    ap.add_argument("--roots", nargs="+", default=[str(root / "data"), str(root / "eval")],
                    help="세션 CSV를 찾을 폴더 (하위 폴더 포함)")
    ap.add_argument("--model", default=str(root / "assets" / "pose_landmarker_lite.task"))
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--mask", action="store_true", help="사람 밖 배경을 검은색으로 칠한 뒤 자르기")
    mode.add_argument("--silhouette", action="store_true", help="사람 윤곽만 흑백으로 저장")
    ap.add_argument("--out-name", default="",
                    help="저장 폴더 이름 (기본: images_crop / images_cropmask / images_sil)")
    ap.add_argument("--overwrite", action="store_true", help="이미 만든 이미지도 다시 만들기")
    args = ap.parse_args()
    args.out_name = args.out_name or ("images_sil" if args.silhouette else
                                      "images_cropmask" if args.mask else "images_crop")
    return args


def make_detector(model_path, mask):
    path = Path(model_path)
    if not path.exists():
        raise SystemExit(f"{path} 없음. assets/ 폴더의 pose_landmarker_lite.task를 확인하세요.")
    vision = mp.tasks.vision
    options = vision.PoseLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(path.resolve())),
        running_mode=vision.RunningMode.IMAGE,       # 저장된 프레임은 서로 떨어져 있어 한 장씩 검출
        num_poses=1,
        output_segmentation_masks=mask,              # 사람 윤곽 (0~1, 원본과 같은 크기)
    )
    return vision.PoseLandmarker.create_from_options(options)


def crop_box(lms, w, h):
    """랜드마크 → 잘라낼 영역 (x0, y0, x1, y1) 픽셀. 어깨가 잘 안 보이면 None"""
    if lms is None or min(getattr(lms[i], "visibility", 1.0) or 0.0 for i in (L_SH, R_SH)) < MIN_VIS:
        return None
    ls, rs = (lms[L_SH].x * w, lms[L_SH].y * h), (lms[R_SH].x * w, lms[R_SH].y * h)
    sw = math.dist(ls, rs)
    if sw < 4:
        return None
    cx, cy = (ls[0] + rs[0]) / 2, (ls[1] + rs[1]) / 2
    bw = sw * WIDTH_K
    bh = bw * OUT_H / OUT_W
    y0 = cy - sw * TOP_K
    return (cx - bw / 2, y0, cx + bw / 2, y0 + bh)


def apply_mask(img, mask):
    """사람 윤곽 밖을 검은색으로. 경계는 윤곽 값(0~1)만큼 섞어 부드럽게"""
    m = np.clip(np.squeeze(mask).astype(np.float32), 0.0, 1.0)[..., None]
    return (img.astype(np.float32) * m).astype(np.uint8)


def to_silhouette(img, mask):
    """사람 윤곽(0~1) → 흑백 3채널 이미지. 윤곽이 아직 없으면 전부 검은색"""
    if mask is None:
        return np.zeros_like(img)
    m = (np.clip(np.squeeze(mask).astype(np.float32), 0.0, 1.0) * 255).astype(np.uint8)
    return cv2.cvtColor(m, cv2.COLOR_GRAY2BGR)


def crop(img, box):
    """영역이 화면 밖으로 나가면 검은색으로 채운 뒤 잘라서 160×120으로 맞춘다"""
    h, w = img.shape[:2]
    x0, y0, x1, y1 = (int(round(v)) for v in box)
    pad = max(0, -x0, -y0, x1 - w, y1 - h)
    if pad:
        img = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        x0, y0, x1, y1 = x0 + pad, y0 + pad, x1 + pad, y1 + pad
    return cv2.resize(img[y0:y1, x0:x1], (OUT_W, OUT_H), interpolation=cv2.INTER_LINEAR)


def process_session(csv_path, detector, out_name, overwrite, use_mask=False, silhouette=False):
    """세션 하나 처리 → (검출 성공, 이전 위치 사용, 원본 없음, 건너뜀) 장 수"""
    with open(csv_path, newline="") as f:
        rels = [r["image"] for r in csv.DictReader(f) if r.get("image")]
    ok = reused = missing = skipped = 0
    last, last_mask = None, None                    # 검출 실패 시 바로 앞 프레임의 영역·윤곽 사용
    for rel in rels:
        src = csv_path.parent / rel
        dst = csv_path.parent / out_name / Path(*Path(rel).parts[1:])
        if dst.exists() and not overwrite:
            skipped += 1
            continue
        img = cv2.imread(str(src))
        if img is None:
            missing += 1
            continue
        h, w = img.shape[:2]
        res = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB,
                                       data=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
        box = crop_box(res.pose_landmarks[0] if res.pose_landmarks else None, w, h)
        if box is None:
            box = last or (0, 0, w, h)
            reused += 1
        else:
            last = box
            ok += 1
        if use_mask or silhouette:
            masks = getattr(res, "segmentation_masks", None)
            if masks:
                last_mask = masks[0].numpy_view()
            if silhouette:
                img = to_silhouette(img, last_mask)
            elif last_mask is not None:
                img = apply_mask(img, last_mask)
        dst.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(dst), crop(img, box), [cv2.IMWRITE_JPEG_QUALITY, 90])
    return ok, reused, missing, skipped


def main():
    args = parse_args()
    files = sorted({p for r in args.roots if Path(r).exists()
                    for p in Path(r).rglob("posture_session_*.csv")})
    if not files:
        raise SystemExit(f"세션 CSV가 없습니다: {args.roots}")
    detector = make_detector(args.model, args.mask or args.silhouette)
    try:
        for p in files:
            ok, reused, missing, skipped = process_session(p, detector, args.out_name,
                                                           args.overwrite, args.mask, args.silhouette)
            print(f"{p.parent.name}/{p.stem[16:]:<28} 검출 {ok:5d}장  이전 위치 사용 {reused:4d}장  "
                  f"원본 없음 {missing:3d}장  이미 있음 {skipped:5d}장")
    finally:
        detector.close()
    print(f"\n저장 위치: 각 CSV 폴더의 {args.out_name}/")


if __name__ == "__main__":
    main()