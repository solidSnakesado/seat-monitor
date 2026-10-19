"""얼굴 성능 히스토리: evaluation.face_compare를 실행할 때마다 results/face_history.md에 한 줄을 추가한다

열: A MediaPipe만 / B 수치 / C 이미지 / D 수치+이미지 / E B·C 결합 / T 시계열 / F T·C 결합.
T·F 열은 나중에 추가되었으므로 표 끝에 붙인다 (이전 줄은 그 칸이 비어 보인다).
기존 파일의 머리줄에 T·F 열이 없으면 머리줄만 새 형식으로 바꾸고, 이전 줄은 그대로 둔다.
"""
from datetime import datetime
from pathlib import Path

import numpy as np

from collect.face_guide import FACE_CLASSES
from evaluation.eval_metrics import balanced_acc, confusion

OLD_HEADER = "| 날짜 | 단계 | 학습 세션 | 평가 세션 | A | B | C | D | E | 오경보 A / B / C / D / E | 상세 |"
OLD_RULE = "|---|---|---|---|---|---|---|---|---|---|---|"
HEADER = OLD_HEADER + " T | F | 오경보 T / F |"
RULE = OLD_RULE + "---|---|---|"
HISTORY_HEAD = f"""# 얼굴 판정 성능 히스토리

`python -m evaluation.face_compare`를 실행할 때마다 한 줄씩 추가된다 (지우지 않고 누적).
A MediaPipe만 / B 수치만 학습 / C 이미지만 학습 / D 수치 + 이미지 학습 / E B·C 결과 결합 / T 시계열 / F T·C 결과 결합.
점수는 균형 정확도, 오경보는 앞 보기 구간(화면·강의 공간·헷갈리는 동작)을 영역 밖·졸음·하품으로 판정한 비율.

{HEADER}
{RULE}
"""


def short(sessions):
    """세션 이름 → 태그만 (face_session_날짜_시간_ 제거)"""
    names = dict.fromkeys(s.split("_", 4)[-1] if s.count("_") >= 4 else s for s in sessions)
    return ", ".join(names) or "-"


def _scores(split, preds):
    """→ ({방식 글자: 균형 정확도 문자열}, {방식 글자: 앞 보기 오경보 문자열})"""
    n = len(FACE_CLASSES)
    front = np.array([t != "" and FACE_CLASSES[y] == "front" for t, y in zip(split.subtags, split.y)])
    alarm_idx = [FACE_CLASSES.index(c) for c in ("away", "drowsy", "yawn")]
    score, alarm = {}, {}
    for k, pred in preds.items():
        score[k[0]] = f"{balanced_acc(confusion(split.y, pred, n)):.3f}"
        alarm[k[0]] = f"{np.isin(pred[front], alarm_idx).mean():.0%}" if front.any() else "-"
    return score, alarm


def _ensure_header(path):
    """파일이 없으면 만들고, 이전 형식 머리줄이면 T·F 열을 붙인다"""
    if not path.exists():
        path.write_text(HISTORY_HEAD, encoding="utf-8")
        return
    text = path.read_text(encoding="utf-8")
    if HEADER not in text and OLD_HEADER in text:
        text = text.replace(OLD_HEADER, HEADER, 1).replace(OLD_RULE, RULE, 1)
        text = text.replace("/ E B·C 결과 결합.", "/ E B·C 결과 결합 / T 시계열 / F T·C 결과 결합.", 1)
        path.write_text(text, encoding="utf-8")


def append_history(path, note, train, split, title, preds, detail):
    path = Path(path)
    _ensure_header(path)
    score, alarm = _scores(split, preds)
    cell = lambda d, ms: [d.get(m, "-") for m in ms]          # noqa: E731
    row = (f"| {datetime.now():%Y-%m-%d %H:%M} | {note or '-'} | {short(train.sessions)} | "
           f"{title}: {short(split.sessions)} | " + " | ".join(cell(score, "ABCDE")) + " | "
           + " / ".join(cell(alarm, "ABCDE")) + f" | {detail} | "
           + " | ".join(cell(score, "TF")) + " | " + " / ".join(cell(alarm, "TF")) + " |\n")
    with open(path, "a", encoding="utf-8") as f:
        f.write(row)