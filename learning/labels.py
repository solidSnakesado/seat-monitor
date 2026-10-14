"""라벨 체계: 수집은 항상 5개 자세로 기록하고, 학습·평가에서 라벨 체계를 골라 변환한다

  merged (기본) : 거북목 + 구부정 → 앞으로 숙임 (4개). 정면 카메라에서 안정적으로 보이는 범위
  fine          : 수집 라벨 그대로 5개. 측면 카메라 추가 후 두 자세를 나눠 진단할 때 사용
"""
import numpy as np

from learning.posture_model import CLASSES as FINE_CLASSES

SCHEMES = {
    "merged": {"classes": ["good", "forward_lean", "head_tilt", "shoulder_tilt"],
               "ko": ["바른", "앞으로숙임", "고개기울임", "어깨비대칭"],
               "map": [0, 1, 1, 2, 3]},              # 수집 라벨 인덱스 → 이 체계의 인덱스
    "fine": {"classes": list(FINE_CLASSES),
             "ko": ["바른", "거북목", "구부정", "고개기울임", "어깨비대칭"],
             "map": [0, 1, 2, 3, 4]},
}
DEFAULT = "merged"


def classes(scheme):
    return SCHEMES[scheme]["classes"]


def ko_names(scheme):
    return SCHEMES[scheme]["ko"]


def convert(y, scheme):
    """수집 라벨(5개 기준 인덱스) → 라벨 체계의 인덱스"""
    return np.asarray(SCHEMES[scheme]["map"], dtype=int)[np.asarray(y, dtype=int)]


def convert_probs(probs, model_classes, scheme):
    """모델 출력 확률 → 라벨 체계의 확률. 변환할 수 없으면 None
    5개로 학습한 모델을 merged로 평가할 때는 거북목·구부정 확률을 더한다"""
    target = classes(scheme)
    if list(model_classes) == target:
        return probs
    if list(model_classes) != list(FINE_CLASSES):     # 예: merged로 학습한 모델을 fine으로 평가
        return None
    out = np.zeros((len(probs), len(target)), dtype=probs.dtype)
    for i, j in enumerate(SCHEMES[scheme]["map"]):
        out[:, j] += probs[:, i]
    return out