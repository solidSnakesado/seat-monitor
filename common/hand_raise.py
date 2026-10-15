"""손 듦: 팔꿈치가 같은 쪽 어깨보다 위, 또는 손목이 같은 쪽 귀보다 위

손을 들면 고개가 움직이고 몸이 앞으로 숙여져 자세·얼굴 판정이 흔들린다.
손 듦이 hold_sec 이상 이어지면 확정하고, 손을 내린 뒤 release_sec 동안도 손 듦으로 유지해
팔을 내리며 몸이 흔들리는 구간까지 넘긴다. 확정 동안 monitor_main은 자세 판정과
얼굴의 전방 주시 미흡·얼굴 미검출 판정을 멈춘다 (졸음·하품은 계속 판정).
"""
import cv2

from common.posture_judge import DurationFilter

L_EAR, R_EAR, L_SH, R_SH, L_EL, R_EL, L_WR, R_WR = 7, 8, 11, 12, 13, 14, 15, 16
SIDES = ((L_SH, L_EL, L_WR, L_EAR), (R_SH, R_EL, R_WR, R_EAR))   # (어깨, 팔꿈치, 손목, 귀) 같은 쪽끼리
MIN_VISIBILITY = 0.5
FACE_PAUSED = ("not_forward", "no_face")                          # 손 듦 동안 멈추는 얼굴 판정


def _seen(lm):
    v = getattr(lm, "visibility", None)
    return v is None or v >= MIN_VISIBILITY


def is_hand_raised(lms):
    """이번 프레임 손 듦 여부 (화면 y는 아래로 갈수록 큼). 잘 보이는 점만 사용"""
    if lms is None:
        return False
    for sh, el, wr, ear in SIDES:
        if _seen(lms[sh]) and _seen(lms[el]) and lms[el].y < lms[sh].y:     # 손이 화면 밖이어도 팔꿈치로
            return True
        if _seen(lms[ear]) and _seen(lms[wr]) and lms[wr].y < lms[ear].y:
            return True
    return False


class HandRaise:
    """손 듦 상태: hold_sec 이상 이어지면 확정, 내린 뒤 release_sec 지나면 해제"""

    def __init__(self, hold_sec=0.5, release_sec=1.5):
        self._filter = DurationFilter({"hand_up": hold_sec}, release_sec)
        self.active = False

    def update(self, lms, now):
        raised = is_hand_raised(lms)
        self.active = bool(self._filter.update(["hand_up"] if raised else [], now))
        return self.active

    def reset(self):
        self._filter.reset()
        self.active = False


def draw_arm_points(frame, lms, active=False):
    """팔꿈치·손목 점 (손 듦 확정이면 노란색)"""
    if lms is None:
        return
    h, w = frame.shape[:2]
    color = (0, 255, 255) if active else (200, 200, 200)
    for i in (L_EL, R_EL, L_WR, R_WR):
        if _seen(lms[i]):
            cv2.circle(frame, (int(lms[i].x * w), int(lms[i].y * h)), 5, color, -1)