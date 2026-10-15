"""실시간 화면 표시용 줄 만들기 (monitor_main.py에서 사용)"""
from common.face_judge import face_deltas

GREEN, RED, YELLOW, WHITE = (0, 255, 0), (0, 0, 255), (0, 255, 255), (255, 255, 255)


def status_line(title, confirmed, labels):
    """확정된 문제 → (문자열, 색)"""
    if confirmed:
        return (f"{title}: " + ", ".join(labels[k] for k in confirmed), RED)
    return (f"{title}: 정상" if title == "얼굴" else f"{title}: 바른 자세", GREEN)


def posture_line(hand, visible, confirmed, labels):
    """자세 상태 줄: 손 듦 > 어깨 안 보임 > 판정 결과"""
    if hand:
        return ("자세: 손 듦 (자세 판정 멈춤)", YELLOW)
    if not visible:
        return ("자세: 어깨가 보이지 않습니다", YELLOW)
    return status_line("자세", confirmed, labels)


def pending_text(candidates, filt, labels, now):
    """아직 확정되지 않은 문제와 경과 시간"""
    return [f"{labels[k]} {filt.elapsed(k, now):.1f}초" for k in candidates if k not in filt.confirmed]


def value_lines(f, base, d):
    """얼굴·자세 원시 값 (기준 대비)"""
    lines = []
    if f is not None:
        fd = face_deltas(f, base)
        lines.append((f"얼굴  눈 감김 {f['eye_closed']:.2f}  입 벌림 {f['jaw_open']:.2f}  "
                      f"시선 {f['gaze']:.2f}  고개 좌우 {fd['yaw']:+.0f}°  상하 {fd['pitch']:+.0f}°", WHITE))
    if d is not None:
        lines.append((f"자세  귀-어깨 {d['ear_shoulder']:+.0%}  머리 비율 {d['head_ratio']:+.0%}  "
                      f"어깨 높이 {d['shoulder_y']:+.3f}", WHITE))
        lines.append((f"      어깨 기울기 {d['shoulder_tilt']:+.1f}°  귀 기울기 {d['ear_tilt']:+.1f}°", WHITE))
    return lines


def paste_inputs(frame, inputs):
    """모델 입력 이미지(160×120)를 화면 오른쪽 아래에 나란히 붙임 (학습 이미지와 같은 모양인지 확인용)"""
    h, x = frame.shape[0], frame.shape[1]
    for img in inputs.values():
        x -= img.shape[1]
        if x < 0 or img.shape[0] > h:
            break
        frame[h - img.shape[0]:h, x:x + img.shape[1]] = img