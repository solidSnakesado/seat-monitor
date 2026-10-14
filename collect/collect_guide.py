"""안내 모드: 자세와 강도를 정해진 순서·시간으로 화면에 안내하고 자동으로 녹화한다

순서 (기본값 기준 약 9분)
  바른 자세 (가만히 60초 → 움직이며 60초)
  → 거북목·구부정·고개 기울임·어깨 비대칭을 회차마다 섞은 순서로,
    각각 약하게 30초 → 중간 30초 → 심하게 30초
단계마다 3초 준비(녹화 안 함) 후 녹화가 자동으로 시작되고, 시간이 끝나면 다음 단계로 넘어간다.
"""
import random
from dataclasses import dataclass

from learning.posture_model import CLASSES

INTENSITY_KO = {"none": "", "mild": "약하게", "mid": "중간", "severe": "심하게"}
PREPARE_SEC = 3.0
GREEN, RED, YELLOW, WHITE, CYAN = (0, 255, 0), (0, 0, 255), (0, 255, 255), (255, 255, 255), (255, 255, 0)

# (자세, 강도) → 안내 문구
HOW = {
    ("good", "calm"): "허리를 펴고 정면 화면을 보며 바르게 앉으세요",
    ("good", "active"): "바른 자세 그대로 타이핑·말하기·화면 안 시선 이동을 하세요",
    ("forward_head", "mild"): "등은 그대로, 머리만 화면 쪽으로 살짝(2~3cm)",
    ("forward_head", "mid"): "등은 그대로, 머리만 앞으로 확실히(약 5cm)",
    ("forward_head", "severe"): "머리를 화면에 바짝 다가가게 내미세요",
    ("slouch", "mild"): "등을 살짝 둥글게, 어깨를 조금 떨어뜨리세요",
    ("slouch", "mid"): "등을 둥글게 말고 어깨를 내리세요",
    ("slouch", "severe"): "의자에 무너지듯 크게 구부정하게 앉으세요",
    ("head_tilt", "mild"): "고개를 옆으로 살짝(5~10°) 기울이세요",
    ("head_tilt", "mid"): "고개를 옆으로 확실히(10~20°) 기울이세요",
    ("head_tilt", "severe"): "고개를 옆으로 크게(20° 이상) 기울이세요",
    ("shoulder_tilt", "mild"): "한쪽 어깨를 살짝 내리세요",
    ("shoulder_tilt", "mid"): "한쪽 어깨를 확실히 내리거나 한쪽으로 기대세요",
    ("shoulder_tilt", "severe"): "한쪽 팔로 턱을 괴거나 크게 기울어 앉으세요",
}
SIDED = {"head_tilt", "shoulder_tilt"}          # 시간의 절반이 지나면 반대쪽으로


@dataclass
class Step:
    label: int          # CLASSES 인덱스
    intensity: str      # none / mild / mid / severe (CSV에 기록)
    seconds: float
    text: str
    sided: bool = False


def build_steps(good_sec=120, step_sec=30, shuffle=True, seed=None):
    steps = [Step(0, "none", good_sec / 2, HOW[("good", "calm")]),
             Step(0, "none", good_sec / 2, HOW[("good", "active")])]
    postures = [c for c in CLASSES if c != "good"]
    if shuffle:                                 # 순서 효과(피로 등)를 줄이기 위해 회차마다 섞음
        random.Random(seed).shuffle(postures)
    for name in postures:
        for level in ("mild", "mid", "severe"):
            steps.append(Step(CLASSES.index(name), level, step_sec, HOW[(name, level)], name in SIDED))
    return steps


class Guide:
    """상태: idle → (prepare → record) × 단계 수 → done.  pause 중에는 시간이 멈춘다"""

    def __init__(self, steps):
        self.steps = steps
        self.state, self.idx, self.elapsed, self.paused = "idle", 0, 0.0, False

    @property
    def running(self):
        return self.state in ("prepare", "record")

    @property
    def recording(self):
        return self.state == "record" and not self.paused

    @property
    def step(self):
        return self.steps[self.idx]

    def start(self):
        self.state, self.idx, self.elapsed, self.paused = "prepare", 0, 0.0, False

    def toggle_pause(self):
        self.paused = not self.paused

    def skip(self):
        self._next()

    def stop(self):
        self.state = "idle"

    def update(self, dt, active=True):
        """active=False(사람이 안 보임 등)면 시간을 진행하지 않는다"""
        if not self.running or self.paused or not active:
            return
        self.elapsed += dt
        if self.state == "prepare" and self.elapsed >= PREPARE_SEC:
            self.state, self.elapsed = "record", 0.0
        elif self.state == "record" and self.elapsed >= self.step.seconds:
            self._next()

    def _next(self):
        self.idx += 1
        self.elapsed = 0.0
        self.state = "prepare" if self.idx < len(self.steps) else "done"

    def side_text(self):
        if not self.step.sided:
            return ""
        half = self.state == "record" and self.elapsed >= self.step.seconds / 2
        return "  → 반대쪽으로" if half else "  (먼저 왼쪽)"

    def lines(self, ko_labels):
        """화면 안내 줄 [(문자열, BGR 색)]"""
        if self.state == "done":
            return [("안내 완료! ESC로 종료하면 저장 결과가 출력됩니다", GREEN)]
        if not self.running:
            return []
        s = self.step
        title = f"[안내 {self.idx + 1}/{len(self.steps)}] {ko_labels[CLASSES[s.label]]}"
        if s.intensity != "none":
            title += f" · {INTENSITY_KO[s.intensity]}"
        if self.paused:
            status = ("일시정지 (Space로 계속)", YELLOW)
        elif self.state == "prepare":
            status = (f"준비: 자세를 취하세요  {PREPARE_SEC - self.elapsed:.0f}초 후 녹화 시작", YELLOW)
        else:
            status = (f"● 녹화 중  남은 {s.seconds - self.elapsed:.0f}초{self.side_text()}", RED)
        return [(title, CYAN), (s.text, WHITE), status]

    def total_seconds(self):
        return sum(s.seconds + PREPARE_SEC for s in self.steps)