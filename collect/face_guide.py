"""얼굴 안내 모드: 시선 영역·졸음·하품·헷갈리는 동작을 정해진 순서·시간으로 안내하고 자동으로 녹화한다

순서 (기본값 기준 약 12분)
  1 front 화면 보기 60초 → 2 front 강의 공간 보기 60초 (--no-lecture면 생략) → 3 desk 60초
  → 4 away · 5 drowsy · 6 yawn 을 회차마다 섞은 순서로, 강도별 30초
  → 7 헷갈리는 동작 (front + 하위 태그) 각 30초 (안경 미착용이면 안경 고쳐 쓰기 생략)
단계마다 3초 준비(녹화 안 함) 후 자동 녹화. 한 단계가 끝나면 대기하고, Space를 눌러야 다음 단계로 간다
(화면을 보지 않는 동작이 많아 자동으로 넘어가면 다음 단계를 놓치기 때문). 준비·녹화 진행은 collect_guide.Guide를 쓴다.
"""
import random
from dataclasses import dataclass

from collect.collect_guide import CYAN, GREEN, PREPARE_SEC, RED, WHITE, YELLOW, Guide

FACE_CLASSES = ("front", "desk", "away", "drowsy", "yawn")
FACE_KO = {"front": "앞 보기", "desk": "책상 보기", "away": "영역 밖",
           "drowsy": "졸음", "yawn": "하품"}
INTENSITY_KO = {"none": "", "mild": "약하게", "mid": "중간", "severe": "심하게"}
SUBTAG_KO = {"screen": "화면", "lecture": "강의 공간", "talk": "말하기", "laugh": "웃기",
             "blink": "잦은 깜빡임", "glasses_fix": "안경 고쳐 쓰기", "drink": "물 마시기",
             "chin_rest": "턱 괴기"}

# 기본 단계 (라벨, 하위 태그) → 안내 문구
BASE_HOW = {
    ("front", "screen"): "내 화면을 자연스럽게 보세요 (평소처럼 깜빡여도 됩니다)",
    ("front", "lecture"): "강의 공간 한쪽 끝에서 반대쪽 끝까지 천천히 시선을 옮기세요",
    ("desk", ""): "책상 위 노트에 필기하거나 교재를 읽으세요",
}
# 상태 (라벨, 강도) → 안내 문구
LEVEL_HOW = {
    ("away", "mild"): "화면·강의 공간·책상이 아닌 곳을 살짝 보세요 (구간 바로 바깥)",
    ("away", "mid"): "옆이나 위를 확실히 보세요 (창밖, 옆자리 등)",
    ("away", "severe"): "고개를 크게 돌려 옆·뒤쪽을 보세요 (얼굴이 화면에 남는 범위까지)",
    ("drowsy", "mild"): "눈을 반쯤 감고 졸린 듯 화면을 보세요",
    ("drowsy", "mid"): "눈을 감은 채 유지하세요 (가끔 힘겹게 떴다 감기)",
    ("drowsy", "severe"): "눈을 감고 고개를 천천히 떨구세요 (얼굴이 화면에 남는 범위까지)",
    ("yawn", "mild"): "입을 작게 벌리는 하품을 반복하세요",
    ("yawn", "severe"): "입을 크게 벌리는 하품을 반복하세요",
}
LEVELS = {"away": ("mild", "mid", "severe"), "drowsy": ("mild", "mid", "severe"),
          "yawn": ("mild", "severe")}
# 헷갈리는 동작: 라벨은 front, 하위 태그로 구분해 오경보를 따로 측정
HARD_HOW = {
    "talk": "화면을 보며 소리 내어 읽거나 말하세요",
    "laugh": "화면을 보며 웃으세요 (미소부터 크게 웃기까지)",
    "blink": "화면을 보며 평소보다 자주 깜빡이세요",
    "glasses_fix": "화면을 보다가 안경 고쳐 쓰기를 반복하세요",
    "drink": "화면을 보다가 컵으로 물 마시기를 반복하세요",
    "chin_rest": "한 손으로 턱을 괴고 화면을 보세요",
}


@dataclass
class FaceStep:
    label: str          # FACE_CLASSES 중 하나 (CSV에 기록)
    intensity: str      # none / mild / mid / severe
    seconds: float
    text: str
    subtag: str = ""    # screen / lecture / 헷갈리는 동작
    sided: bool = False


def build_face_steps(base_sec=60, step_sec=30, lecture=True, glasses=True, shuffle=True, seed=None):
    steps = [FaceStep("front", "none", base_sec, BASE_HOW[("front", "screen")], "screen")]
    if lecture:
        steps.append(FaceStep("front", "none", base_sec, BASE_HOW[("front", "lecture")], "lecture"))
    steps.append(FaceStep("desk", "none", base_sec, BASE_HOW[("desk", "")]))
    order = list(LEVELS)
    if shuffle:                                 # 순서 효과(피로 등)를 줄이기 위해 회차마다 섞음
        random.Random(seed).shuffle(order)
    for name in order:
        for level in LEVELS[name]:
            steps.append(FaceStep(name, level, step_sec, LEVEL_HOW[(name, level)], sided=name == "away"))
    for tag, how in HARD_HOW.items():
        if tag == "glasses_fix" and not glasses:
            continue
        steps.append(FaceStep("front", "none", step_sec, how, tag))
    return steps


class FaceGuide(Guide):
    """collect_guide.Guide의 준비·녹화 진행을 쓰되, 단계가 끝나면 wait 상태로 멈추고 화면 문구를 얼굴용으로 바꾼다"""

    @property
    def waiting(self):
        return self.state == "wait"

    def _next(self):
        """녹화 시간이 끝났거나 n으로 건너뜀 → 마지막 단계면 done, 아니면 대기"""
        self.elapsed = 0.0
        self.state = "done" if self.idx + 1 >= len(self.steps) else "wait"

    def advance(self):
        """대기 중 Space: 다음 단계의 준비부터 시작"""
        if self.waiting:
            self.idx += 1
            self.state, self.elapsed, self.paused = "prepare", 0.0, False

    def skip(self):
        """준비·녹화 중이면 지금 단계를, 대기 중이면 다음 단계를 건너뛰고 대기"""
        if self.waiting:
            self.idx += 1
            self.state = "done" if self.idx + 1 >= len(self.steps) else "wait"
        else:
            self._next()

    def side_text(self):
        if not self.step.sided:
            return ""
        half = self.state == "record" and self.elapsed >= self.step.seconds / 2
        return "  → 다른 방향으로" if half else "  (먼저 한 방향)"

    def _title(self, i, ko_labels):
        s = self.steps[i]
        title = f"{ko_labels[s.label]}"
        if s.intensity != "none":
            title += f" · {INTENSITY_KO[s.intensity]}"
        if s.subtag:
            title += f" · {SUBTAG_KO[s.subtag]}"
        return title

    def lines(self, ko_labels=FACE_KO):
        """화면 안내 줄 [(문자열, BGR 색)]"""
        if self.state == "done":
            return [("안내 완료! ESC로 종료하면 저장 결과가 출력됩니다", GREEN)]
        n = len(self.steps)
        if self.waiting:
            nxt = self.steps[self.idx + 1]
            return [(f"단계 완료 ({self.idx + 1}/{n})  다음: [{self.idx + 2}/{n}] "
                     f"{self._title(self.idx + 1, ko_labels)}", CYAN),
                    (nxt.text, WHITE),
                    ("화면을 보고 Space로 시작 (n: 이 단계 건너뛰기)", YELLOW)]
        if not self.running:
            return []
        s = self.step
        title = f"[안내 {self.idx + 1}/{n}] {self._title(self.idx, ko_labels)}"
        if self.paused:
            status = ("일시정지 (Space로 계속)", YELLOW)
        elif self.state == "prepare":
            status = (f"준비: 동작을 준비하세요  {PREPARE_SEC - self.elapsed:.0f}초 후 녹화 시작", YELLOW)
        else:
            status = (f"● 녹화 중  남은 {s.seconds - self.elapsed:.0f}초{self.side_text()}", RED)
        return [(title, CYAN), (s.text, WHITE), status]