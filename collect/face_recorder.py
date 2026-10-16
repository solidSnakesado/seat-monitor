"""얼굴 세션 기록: 프레임마다 라벨 + 고개 각도 + blendshape 52개를 CSV로, N프레임마다 얼굴 crop 이미지를 저장

CSV 열: t, label, intensity, subtag, glasses, face, yaw, pitch, roll, blendshape 52개(이름 그대로), image
  - face: 1 얼굴 검출, 0 미검출 (녹화 중 얼굴을 놓친 프레임도 남긴다. 값 열은 빈칸)
    face 열이 없는 이전 세션은 모든 행이 검출 프레임 (불러올 때 1로 간주)
  - image: CSV 기준 상대 경로. 원본 해상도에서 얼굴을 잘라 128×128로 저장, 저장하지 않은 프레임은 빈칸
  - 파일 이름: face_session_날짜_시간[_태그]_g(0|1)[_test].csv
영역 캘리브레이션 결과는 같은 이름의 .json에 저장한다.
"""
import csv
import json
import math
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from collect.face_guide import FACE_CLASSES, FACE_KO, SUBTAG_KO
from common.face_features import head_angles

# Face Landmarker blendshape 52개 (모델 출력 순서). 얼굴이 없는 프레임도 같은 열로 쓰기 위해 고정
BLENDSHAPES = (
    "_neutral", "browDownLeft", "browDownRight", "browInnerUp", "browOuterUpLeft", "browOuterUpRight",
    "cheekPuff", "cheekSquintLeft", "cheekSquintRight", "eyeBlinkLeft", "eyeBlinkRight",
    "eyeLookDownLeft", "eyeLookDownRight", "eyeLookInLeft", "eyeLookInRight", "eyeLookOutLeft",
    "eyeLookOutRight", "eyeLookUpLeft", "eyeLookUpRight", "eyeSquintLeft", "eyeSquintRight",
    "eyeWideLeft", "eyeWideRight", "jawForward", "jawLeft", "jawOpen", "jawRight", "mouthClose",
    "mouthDimpleLeft", "mouthDimpleRight", "mouthFrownLeft", "mouthFrownRight", "mouthFunnel",
    "mouthLeft", "mouthLowerDownLeft", "mouthLowerDownRight", "mouthPressLeft", "mouthPressRight",
    "mouthPucker", "mouthRight", "mouthRollLower", "mouthRollUpper", "mouthShrugLower",
    "mouthShrugUpper", "mouthSmileLeft", "mouthSmileRight", "mouthStretchLeft", "mouthStretchRight",
    "mouthUpperUpLeft", "mouthUpperUpRight", "noseSneerLeft", "noseSneerRight")
FACE_SIZE = 128     # 저장 이미지 한 변 (픽셀)
MARGIN = 1.4        # 얼굴 랜드마크 상자 대비 crop 크기 배율


def head_roll(matrix):
    """얼굴 변환 행렬(4x4) → roll (고개 옆 기울기, 도)"""
    R = np.asarray(matrix)[:3, :3]
    return math.degrees(math.atan2(R[1, 0], R[0, 0]))


def crop_face(frame, landmarks):
    """랜드마크 상자 중심으로 정사각형 crop (화면 밖은 검은색) → FACE_SIZE×FACE_SIZE, 너무 작으면 None"""
    h, w = frame.shape[:2]
    xs = [p.x * w for p in landmarks]
    ys = [p.y * h for p in landmarks]
    side = int(max(max(xs) - min(xs), max(ys) - min(ys)) * MARGIN)
    if side < 16:
        return None
    x0 = int((min(xs) + max(xs)) / 2 - side / 2)
    y0 = int((min(ys) + max(ys)) / 2 - side / 2)
    canvas = np.zeros((side, side, 3), dtype=frame.dtype)
    sx0, sy0 = max(x0, 0), max(y0, 0)
    sx1, sy1 = min(x0 + side, w), min(y0 + side, h)
    if sx1 <= sx0 or sy1 <= sy0:
        return None
    canvas[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = frame[sy0:sy1, sx0:sx1]
    return cv2.resize(canvas, (FACE_SIZE, FACE_SIZE), interpolation=cv2.INTER_AREA)


class FaceRecorder:
    def __init__(self, out_dir, img_every, test, tag, glasses):
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = (f"face_session_{stamp}" + (f"_{tag}" if tag else "")
                + f"_g{glasses}" + ("_test" if test else ""))
        self.root = Path(out_dir)
        self.path = self.root / f"{name}.csv"
        self.zone_path = self.path.with_suffix(".json")
        self.img_dir = Path("images") / name
        self.img_every, self.glasses = img_every, glasses
        self.counts = {c: (0, 0) for c in FACE_CLASSES}   # 라벨 → (프레임, 얼굴 미검출)
        self.sub_counts = {}
        self.n_images = 0
        self._file = self._writer = None
        self._row = 0
        self.root.mkdir(parents=True, exist_ok=True)

    @property
    def is_test(self):
        return self.path.stem.endswith("_test")

    def write(self, t, step, result, clean_frame):
        """result: FaceDetector.detect 결과, 얼굴이 없으면 None (값 열은 빈칸, face=0)"""
        if self._writer is None:
            self._file = open(self.path, "w", newline="")
            self._writer = csv.writer(self._file)
            self._writer.writerow(["t", "label", "intensity", "subtag", "glasses", "face",
                                   "yaw", "pitch", "roll", *BLENDSHAPES, "image"])
        found = result is not None and bool(result.face_blendshapes) and bool(result.facial_transformation_matrixes)
        if found:
            matrix = result.facial_transformation_matrixes[0]
            yaw, pitch = head_angles(matrix)
            scores = {c.category_name: c.score for c in result.face_blendshapes[0]}
            values = [f"{yaw:.2f}", f"{pitch:.2f}", f"{head_roll(matrix):.2f}",
                      *[f"{scores[n]:.4f}" if n in scores else "" for n in BLENDSHAPES]]
            image = ""
            if self.img_every > 0 and self._row % self.img_every == 0:
                image = self._save_image(clean_frame, result.face_landmarks[0])
        else:
            values, image = [""] * (3 + len(BLENDSHAPES)), ""
        self._writer.writerow([f"{t:.3f}", step.label, step.intensity, step.subtag, self.glasses,
                               int(found), *values, image])
        for key, d in ((step.label, self.counts), (step.subtag, self.sub_counts)):
            if key:
                n, miss = d.get(key, (0, 0))
                d[key] = (n + 1, miss + (not found))
        self._row += 1

    def _save_image(self, frame, landmarks):
        face = crop_face(frame, landmarks)
        if face is None:
            return ""
        rel = self.img_dir / f"{self._row:06d}.jpg"
        (self.root / rel).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(self.root / rel), face, [cv2.IMWRITE_JPEG_QUALITY, 90])
        self.n_images += 1
        return rel.as_posix()

    def save_zones(self, zones, warnings):
        """영역 캘리브레이션 결과 (다시 측정하면 덮어씀)"""
        data = {"glasses": self.glasses, "zones": zones, "warnings": warnings}
        self.zone_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def close(self):
        if self._file:
            self._file.close()
            print(f"저장: {self.path}  (이미지 {self.n_images}장)")
            for c, (n, miss) in self.counts.items():
                print(f"  {FACE_KO[c]:<6} {n:6d} 프레임  (얼굴 미검출 {miss / max(n, 1):5.1%})")
            for s, (n, miss) in self.sub_counts.items():
                print(f"    └ {SUBTAG_KO[s]:<10} {n:6d} 프레임  (얼굴 미검출 {miss / max(n, 1):5.1%})")
        if self.zone_path.exists():
            print(f"영역: {self.zone_path}")