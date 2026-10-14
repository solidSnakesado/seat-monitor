"""평가 지표 공통: 혼동행렬, 균형 정확도, macro F1, 자세별 재현율, 규칙 판정

모든 방식(규칙 / 수치 MLP / 이미지 CNN / 융합)을 같은 함수로 평가한다.
클래스 수와 이름은 라벨 체계(learning/labels.py)에 따라 달라지므로 호출하는 쪽에서 넘긴다.
"""
import numpy as np

from common.posture_judge import Thresholds, detect_issues
from learning.labels import convert
from learning.posture_model import CLASSES, INPUTS


def confusion(y_true, y_pred, n):
    cm = np.zeros((n, n), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm


def recalls(cm):
    return [cm[i, i] / cm[i].sum() if cm[i].sum() else float("nan") for i in range(len(cm))]


def balanced_acc(cm):
    r = [v for v in recalls(cm) if not np.isnan(v)]
    return float(np.mean(r)) if r else 0.0


def macro_f1(cm):
    f1s = []
    for i in range(len(cm)):
        tp, fp, fn = cm[i, i], cm[:, i].sum() - cm[i, i], cm[i].sum() - cm[i, i]
        if cm[i].sum() == 0:
            continue
        f1s.append(2 * tp / (2 * tp + fp + fn) if tp else 0.0)
    return float(np.mean(f1s)) if f1s else 0.0


def format_cm(cm, title, names):
    lines = [f"[{title}] 혼동행렬 (행=실제, 열=예측)",
             " " * 15 + "".join(f"{c[:8]:>10}" for c in names)]
    for i, c in enumerate(names):
        lines.append(f"{c:<15}" + "".join(f"{v:>10d}" for v in cm[i]))
    lines.append(f"균형 정확도 {balanced_acc(cm):.3f}  macro F1 {macro_f1(cm):.3f}")
    return "\n".join(lines)


def rule_predict(X, th=None, scheme="fine"):
    """방식 A: MediaPipe 특징 + 규칙 (여러 문제가 동시에 나오면 첫 번째)
    규칙은 5개 자세로 판정한 뒤 라벨 체계에 맞게 변환한다"""
    th = th or Thresholds()
    out = []
    for x in X:
        issues = detect_issues(dict(zip(INPUTS, x)), th)
        out.append(CLASSES.index(issues[0]) if issues else 0)
    return convert(out, scheme)