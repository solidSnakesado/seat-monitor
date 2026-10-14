"""얼굴: Face Landmarker → 눈 감김·하품·시선·고개 각도 특징

모델 다운로드:
  wget -O face_landmarker.task https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
"""
import math
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

GAZE_KEYS = ("eyeLookOutLeft", "eyeLookOutRight", "eyeLookInLeft", "eyeLookInRight",
             "eyeLookUpLeft", "eyeLookUpRight", "eyeLookDownLeft", "eyeLookDownRight")

# 화면 표시용 점: 양 눈 테두리, 입술 안쪽
EYE_MOUTH_POINTS = (33, 160, 158, 133, 153, 144, 362, 385, 387, 263, 373, 380, 13, 14, 61, 291)


class FaceDetector:
    def __init__(self, model_path):
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(f"{path} 없음. 파일 상단의 wget 명령으로 받으세요.")
        vision = mp.tasks.vision
        options = vision.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(path.resolve())),
            running_mode=vision.RunningMode.VIDEO,
            output_face_blendshapes=True,                  # 눈 감김, 입 벌림, 시선 점수
            output_facial_transformation_matrixes=True,    # 고개 각도
        )
        self._landmarker = vision.FaceLandmarker.create_from_options(options)
        self._last_ts = 0

    def detect(self, frame_bgr):
        """→ 검출 결과, 얼굴이 없으면 None"""
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ts = max(self._last_ts + 1, int(time.monotonic() * 1000))   # VIDEO 모드: 증가하는 타임스탬프
        self._last_ts = ts
        result = self._landmarker.detect_for_video(image, ts)
        return result if result.face_landmarks else None

    def close(self):
        self._landmarker.close()


def head_angles(matrix):
    """얼굴 변환 행렬(4x4) → (yaw 좌우, pitch 상하) 도"""
    R = np.asarray(matrix)[:3, :3]
    pitch = math.degrees(math.atan2(R[2, 1], R[2, 2]))
    yaw = math.degrees(math.atan2(-R[2, 0], math.hypot(R[2, 1], R[2, 2])))
    return yaw, pitch


def compute_face_features(result):
    """검출 결과 → 특징 dict. 얼굴이 없으면 None"""
    if result is None or not result.face_blendshapes or not result.facial_transformation_matrixes:
        return None
    bs = {c.category_name: c.score for c in result.face_blendshapes[0]}
    yaw, pitch = head_angles(result.facial_transformation_matrixes[0])
    return {
        "eye_closed": (bs.get("eyeBlinkLeft", 0.0) + bs.get("eyeBlinkRight", 0.0)) / 2,
        "jaw_open": bs.get("jawOpen", 0.0),
        "gaze": max(bs.get(k, 0.0) for k in GAZE_KEYS),    # 눈동자가 한쪽으로 쏠린 정도
        "yaw": yaw,
        "pitch": pitch,
    }


def draw_face_points(frame, result):
    if result is None:
        return
    h, w = frame.shape[:2]
    lms = result.face_landmarks[0]
    for i in EYE_MOUTH_POINTS:
        cv2.circle(frame, (int(lms[i].x * w), int(lms[i].y * h)), 2, (0, 255, 0), -1)