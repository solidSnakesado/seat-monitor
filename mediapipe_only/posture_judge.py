"""4~7단계: 캘리브레이션 → 기준 대비 변화량 → 판정 규칙 → 지속 시간

카메라·MediaPipe와 무관한 순수 로직이라 숫자만 넣어 테스트할 수 있다.
"""
import statistics
from dataclasses import dataclass

FEATURE_NAMES = ("ear_shoulder", "head_ratio", "shoulder_y", "shoulder_tilt", "ear_tilt")

LABELS = {
    "forward_head": "거북목",
    "slouch": "구부정한 자세",
    "head_tilt": "고개 기울임",
    "shoulder_tilt": "어깨 비대칭",
}


@dataclass(frozen=True)
class Thresholds:
    """시작용 초기값. 화면의 변화량을 보면서 조정한다"""
    # 거북목: 머리 크기 비율 증가 + 귀-어깨 간격 감소
    fh_head_ratio_up: float = 0.08      # +8% 이상
    fh_ear_shoulder_down: float = 0.08  # -8% 이상
    # 구부정: 어깨 하강 + 귀-어깨 간격 감소
    slouch_shoulder_down: float = 0.03  # 화면 높이의 3% 이상 내려감
    slouch_ear_shoulder_down: float = 0.10
    # 기울임
    head_tilt_deg: float = 10.0
    shoulder_tilt_deg: float = 6.0
    # 지속 시간
    hold_sec: float = 3.0               # 이 시간 이상 유지돼야 확정 (실사용은 30초 이상 권장)
    release_sec: float = 2.0            # 바른 자세가 이 시간 유지돼야 해제


# ------------------------------------------------------------
# 4단계: 캘리브레이션
# ------------------------------------------------------------
class Calibrator:
    def __init__(self, seconds=5.0, min_valid=0.6, max_shoulder_tilt=6.0, max_ear_tilt=8.0,
                 keys=FEATURE_NAMES):
        self.seconds = seconds
        self.keys = tuple(keys)                     # 기준값으로 저장할 항목 (얼굴 각도 추가 가능)
        self.min_valid = min_valid                  # 유효 프레임 비율 하한
        self.max_shoulder_tilt = max_shoulder_tilt
        self.max_ear_tilt = max_ear_tilt
        self.active = False
        self.message = ""

    def start(self, now):
        self._t0, self._samples, self._frames = now, [], 0
        self.active = True

    def remaining(self, now):
        return max(0.0, self.seconds - (now - self._t0))

    def add(self, feats, now):
        """기준값이 확정되면 dict 반환. 진행 중이거나 실패해 재시작하면 None"""
        self._frames += 1
        if feats is not None and self._is_straight(feats):
            self._samples.append(feats)
        if self.remaining(now) > 0:
            return None
        if len(self._samples) >= max(10, self.min_valid * self._frames):
            self.active = False
            self.message = "기준 자세 저장 완료"
            return {k: statistics.median(s[k] for s in self._samples) for k in self.keys}
        self.message = "기준 자세 저장 실패: 허리를 펴고 어깨를 수평으로 맞춰 주세요"
        self.start(now)
        return None

    def _is_straight(self, f):
        return abs(f["shoulder_tilt"]) < self.max_shoulder_tilt and abs(f["ear_tilt"]) < self.max_ear_tilt


# ------------------------------------------------------------
# 5단계: 기준 대비 변화량
# ------------------------------------------------------------
def compute_deltas(feats, base):
    return {
        "ear_shoulder": feats["ear_shoulder"] / base["ear_shoulder"] - 1.0,   # 비율 (-0.1 = 10% 감소)
        "head_ratio": feats["head_ratio"] / base["head_ratio"] - 1.0,
        "shoulder_y": feats["shoulder_y"] - base["shoulder_y"],             # + 면 어깨가 내려감
        "shoulder_tilt": feats["shoulder_tilt"] - base["shoulder_tilt"],
        "ear_tilt": feats["ear_tilt"] - base["ear_tilt"],
    }


# ------------------------------------------------------------
# 6단계: 판정 규칙
# ------------------------------------------------------------
def detect_issues(d, th: Thresholds):
    """→ 이번 프레임에 해당하는 자세 문제 목록 (빈 목록 = 바른 자세)"""
    issues = []
    if d["head_ratio"] >= th.fh_head_ratio_up and d["ear_shoulder"] <= -th.fh_ear_shoulder_down:
        issues.append("forward_head")
    if d["shoulder_y"] >= th.slouch_shoulder_down and d["ear_shoulder"] <= -th.slouch_ear_shoulder_down:
        issues.append("slouch")
    if abs(d["ear_tilt"]) >= th.head_tilt_deg:
        issues.append("head_tilt")
    if abs(d["shoulder_tilt"]) >= th.shoulder_tilt_deg:
        issues.append("shoulder_tilt")
    return issues


# ------------------------------------------------------------
# 7단계: 지속 시간
# ------------------------------------------------------------
class DurationFilter:
    """문제가 hold_sec 이상 이어지면 확정, release_sec 이상 사라지면 해제
    hold_sec는 숫자 하나 또는 {문제: 초} dict (문제마다 다른 시간)"""

    def __init__(self, hold_sec, release_sec):
        self.hold_sec, self.release_sec = hold_sec, release_sec
        self._since = {}       # 문제가 시작된 시각
        self._gone = {}        # 문제가 사라진 시각
        self.confirmed = set()

    def _hold(self, issue):
        return self.hold_sec[issue] if isinstance(self.hold_sec, dict) else self.hold_sec

    def update(self, issues, now):
        current = set(issues)
        for k in current:
            self._since.setdefault(k, now)
            self._gone.pop(k, None)
            if now - self._since[k] >= self._hold(k):
                self.confirmed.add(k)
        for k in set(self._since) - current:
            self._gone.setdefault(k, now)
            if now - self._gone[k] >= self.release_sec:
                self._since.pop(k)
                self._gone.pop(k)
                self.confirmed.discard(k)
        return sorted(self.confirmed)

    def elapsed(self, issue, now):
        return now - self._since[issue] if issue in self._since else 0.0

    def reset(self):
        self._since.clear()
        self._gone.clear()
        self.confirmed.clear()