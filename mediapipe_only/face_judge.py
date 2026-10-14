"""얼굴 상태 판정: 졸음, 하품, 전방 주시 미흡, 얼굴 미검출

고개 각도는 캘리브레이션 때 저장한 정면 각도 대비 변화량으로 판정한다.
(카메라가 모니터 위에 있으면 정면을 봐도 pitch가 0이 아니기 때문)
"""
from dataclasses import dataclass

from posture_judge import DurationFilter

FACE_LABELS = {
    "drowsy": "졸음",
    "yawn": "하품",
    "not_forward": "전방 주시 미흡",
    "no_face": "얼굴 미검출",
}


@dataclass(frozen=True)
class FaceThresholds:
    eye_closed: float = 0.5     # eyeBlink 평균이 이 이상이면 눈 감음
    jaw_open: float = 0.6       # jawOpen이 이 이상이면 입 벌림
    yaw_deg: float = 20.0       # 정면 대비 좌우 허용 각도
    pitch_deg: float = 15.0     # 정면 대비 상하 허용 각도
    gaze: float = 0.6           # 시선 쏠림 허용치
    drowsy_sec: float = 1.5     # 눈 감김 지속 → 졸음
    yawn_sec: float = 1.0       # 입 벌림 지속 → 하품 (짧은 말하기 제외)
    away_sec: float = 2.0       # 고개·시선 이탈 지속 → 전방 주시 미흡
    no_face_sec: float = 3.0    # 어깨는 보이는데 얼굴이 없음 → 엎드림·고개 돌림 가능성
    release_sec: float = 1.0    # 정상이 이 시간 유지되면 해제


def face_deltas(f, base):
    return {"yaw": f["yaw"] - base["yaw"], "pitch": f["pitch"] - base["pitch"]}


def detect_face_issues(f, base, th: FaceThresholds):
    """→ 이번 프레임의 얼굴 문제 목록. f가 None이면 얼굴 미검출"""
    if f is None:
        return ["no_face"]
    issues = []
    eyes_closed = f["eye_closed"] >= th.eye_closed
    if eyes_closed:
        issues.append("drowsy")
    if f["jaw_open"] >= th.jaw_open:
        issues.append("yawn")
    d = face_deltas(f, base)
    head_off = abs(d["yaw"]) > th.yaw_deg or abs(d["pitch"]) > th.pitch_deg
    gaze_off = (not eyes_closed) and f["gaze"] > th.gaze   # 눈을 감으면 시선 값이 흔들려 제외
    if head_off or gaze_off:
        issues.append("not_forward")
    return issues


def make_face_filter(th: FaceThresholds):
    hold = {"drowsy": th.drowsy_sec, "yawn": th.yawn_sec,
            "not_forward": th.away_sec, "no_face": th.no_face_sec}
    return DurationFilter(hold, th.release_sec)