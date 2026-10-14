"""1~3단계: 카메라 프레임 → Pose 랜드마크 → 자세 특징

사용 랜드마크: 코(0), 왼귀(7), 오른귀(8), 왼어깨(11), 오른어깨(12)
거리 특징은 모두 어깨 폭으로 나눠, 카메라와의 거리가 바뀌어도 값이 유지되게 한다.

모델 다운로드:
  wget -O pose_landmarker_lite.task https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task
"""
import math
import time
from pathlib import Path

import cv2
import mediapipe as mp

NOSE, L_EAR, R_EAR, L_SH, R_SH = 0, 7, 8, 11, 12
KEY_POINTS = (NOSE, L_EAR, R_EAR, L_SH, R_SH)
MIN_VISIBILITY = 0.5          # 이보다 낮은 점이 있으면 특징을 계산하지 않음

FEATURE_NAMES = ("ear_shoulder", "head_ratio", "shoulder_y", "shoulder_tilt", "ear_tilt")


# ------------------------------------------------------------
# 2단계: 랜드마크 검출
# ------------------------------------------------------------
class PoseDetector:
    def __init__(self, model_path):
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(f"{path} 없음. 파일 상단의 wget 명령으로 받으세요.")
        vision = mp.tasks.vision
        options = vision.PoseLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(path.resolve())),
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,
        )
        self._landmarker = vision.PoseLandmarker.create_from_options(options)
        self._last_ts = 0

    def detect(self, frame_bgr):
        """→ 33개 랜드마크 리스트, 사람이 없으면 None"""
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ts = max(self._last_ts + 1, int(time.monotonic() * 1000))   # VIDEO 모드: 증가하는 타임스탬프
        self._last_ts = ts
        result = self._landmarker.detect_for_video(image, ts)
        return result.pose_landmarks[0] if result.pose_landmarks else None

    def close(self):
        self._landmarker.close()


# ------------------------------------------------------------
# 3단계: 특징 계산
# ------------------------------------------------------------
def _visibility(lm):
    v = getattr(lm, "visibility", None)
    return 1.0 if v is None else v


def level_angle(a, b):
    """두 점을 이은 선이 수평에서 기울어진 각도(도). 점 순서와 무관"""
    if a[0] > b[0]:
        a, b = b, a
    return math.degrees(math.atan2(b[1] - a[1], b[0] - a[0]))


def compute_features(lms, w, h):
    """랜드마크 → 특징 dict. 주요 점이 안 보이면 None"""
    if lms is None or min(_visibility(lms[i]) for i in KEY_POINTS) < MIN_VISIBILITY:
        return None

    def px(i):
        return (lms[i].x * w, lms[i].y * h)

    ls, rs, le, re = px(L_SH), px(R_SH), px(L_EAR), px(R_EAR)
    shoulder_w = math.dist(ls, rs)
    if shoulder_w < 1e-6:
        return None
    sh_mid_y = (ls[1] + rs[1]) / 2
    ear_mid_y = (le[1] + re[1]) / 2

    return {
        "ear_shoulder": (sh_mid_y - ear_mid_y) / shoulder_w,   # 귀-어깨 세로 간격 / 어깨 폭
        "head_ratio": math.dist(le, re) / shoulder_w,         # 양 귀 거리 / 어깨 폭
        "shoulder_y": sh_mid_y / h,                           # 어깨 높이 (아래로 갈수록 큼)
        "shoulder_tilt": level_angle(ls, rs),                 # 어깨 기울기(도)
        "ear_tilt": level_angle(le, re),                      # 귀 기울기(도)
    }


class Smoother:
    """지수평활: 프레임마다 흔들리는 특징값을 부드럽게 (alpha = 새 값 비중)"""

    def __init__(self, alpha=0.3):
        self.alpha = alpha
        self._state = None

    def update(self, feats):
        if feats is None:
            return self._state
        if self._state is None:
            self._state = dict(feats)
        else:
            a = self.alpha
            self._state = {k: (1 - a) * self._state[k] + a * feats[k] for k in FEATURE_NAMES}
        return self._state

    def reset(self):
        self._state = None


# ------------------------------------------------------------
# 그리기
# ------------------------------------------------------------
def draw_key_points(frame, lms):
    if lms is None:
        return
    h, w = frame.shape[:2]
    pts = {i: (int(lms[i].x * w), int(lms[i].y * h)) for i in KEY_POINTS}
    cv2.line(frame, pts[L_SH], pts[R_SH], (255, 200, 0), 2)
    cv2.line(frame, pts[L_EAR], pts[R_EAR], (255, 200, 0), 1)
    for i, p in pts.items():
        color = (0, 200, 255) if _visibility(lms[i]) >= MIN_VISIBILITY else (0, 0, 255)
        cv2.circle(frame, p, 5, color, -1)