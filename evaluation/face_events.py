"""얼굴 이벤트 단위 지표: 프레임 예측 → 지속 시간 필터(실시간과 같은 시간) → 확정 경보를 실제 구간과 맞대어 평가

프레임 지표(균형 정확도, 프레임 오경보율)는 순간 판정의 정확도라, 사용자가 실제로 받는 '확정 경보'와 다르다.
  - 구간: 같은 세션에서 라벨·하위 태그가 같고 시간이 이어진 프레임들 (수집 단계 1개 ≈ 구간 1개).
    프레임 간격이 GAP_SEC 이상이면 새 구간 (단계 사이 준비·대기 시간은 녹화되지 않음)
  - 지속 시간 필터: common.face_judge.FaceThresholds의 시간 (졸음 1.5초, 하품 1초, 영역 밖 2초, 해제 1초).
    빈 시간(GAP_SEC 이상)에서는 필터를 새로 만든다 (녹화되지 않은 시간의 상태를 이어 붙이지 않음)
지표
  - 이벤트 재현율: 졸음·하품·영역 밖 구간 중 같은 종류 경보가 구간 안(끝 + MARGIN_SEC까지)에서 확정된 구간 수
  - 경보 유지율: 그 구간 시간 중 같은 종류 경보가 확정 상태였던 비율
  - 탐지 지연: 구간 시작 → 첫 확정까지 초 (중앙값)
  - 정상 10분당 오경보: 앞 보기·책상 보기 구간에서 경보가 새로 확정된 횟수 ÷ 정상 시간 × 600초 (하위 태그별)
evaluation.face_compare --events 로 보고서에 표를 붙인다.
"""
from dataclasses import dataclass

import numpy as np

from collect.face_guide import FACE_CLASSES, FACE_KO, SUBTAG_KO
from common.face_judge import FaceThresholds
from common.posture_judge import DurationFilter
from learning.face_dataset import ALARM

GAP_SEC = 2.5        # 이 간격 이상 프레임이 비면 다른 단계 (준비 3초 + Space 대기)
MARGIN_SEC = 1.0     # 구간이 끝난 뒤 이 시간 안의 확정도 그 구간의 탐지로 인정 (해제 시간만큼)
NORMAL = ("front", "desk")


@dataclass
class Segment:
    session: str
    label: str
    tag: str             # 하위 태그, 없으면 라벨 (desk, away, ...)
    t0: float
    t1: float

    @property
    def sec(self):
        return max(self.t1 - self.t0, 0.0)


def segments(split):
    """→ [Segment] 세션·라벨·하위 태그가 같고 시간이 이어진 프레임 묶음"""
    out, start = [], 0
    for i in range(1, len(split) + 1):
        if (i < len(split) and split.sessions[i] == split.sessions[start] and split.y[i] == split.y[start]
                and split.subtags[i] == split.subtags[start] and split.t[i] - split.t[i - 1] < GAP_SEC):
            continue
        label = FACE_CLASSES[split.y[start]]
        out.append(Segment(split.sessions[start], label, split.subtags[start] or label,
                           float(split.t[start]), float(split.t[i - 1])))
        start = i
    return out


def make_filter(th):
    hold = {"drowsy": th.drowsy_sec, "yawn": th.yawn_sec, "away": th.away_sec}
    return DurationFilter(hold, th.release_sec)


def confirmed_alarms(split, pred, th):
    """→ [(세션, 종류, 확정 시각, 해제 시각)] 실시간과 같은 지속 시간 필터를 프레임 순서대로 적용"""
    alarms, opened = [], {}
    filt, prev_t, prev_s = None, 0.0, None
    for i in range(len(split)):
        t, s = float(split.t[i]), split.sessions[i]
        if filt is None or s != prev_s or t - prev_t >= GAP_SEC:
            alarms += [(prev_s, k, t0, prev_t) for k, t0 in opened.items()]
            opened.clear()
            filt = make_filter(th)
        name = FACE_CLASSES[pred[i]]
        conf = set(filt.update([name] if name in ALARM else [], t))
        for k in conf - set(opened):
            opened[k] = t
        for k in set(opened) - conf:
            alarms.append((s, k, opened.pop(k), t))
        prev_t, prev_s = t, s
    alarms += [(prev_s, k, t0, prev_t) for k, t0 in opened.items()]
    return alarms


def event_metrics(split, pred, th=None):
    """→ (events {종류: [(탐지 여부, 유지율, 지연 초 | None)]}, normal {태그: [오경보 수, 정상 초]})"""
    th = th or FaceThresholds()
    segs, alarms = segments(split), confirmed_alarms(split, pred, th)
    events = {c: [] for c in ALARM}
    normal = {}
    for g in segs:
        mine = [a for a in alarms if a[0] == g.session]
        if g.label in ALARM:
            same = [a for a in mine if a[1] == g.label]
            hits = [a[2] for a in same if g.t0 <= a[2] <= g.t1 + MARGIN_SEC]
            on = sum(max(0.0, min(a[3], g.t1) - max(a[2], g.t0)) for a in same)
            events[g.label].append((bool(hits), on / g.sec if g.sec > 0 else 0.0,
                                    min(hits) - g.t0 if hits else None))
        elif g.label in NORMAL:
            c = normal.setdefault(g.tag, [0, 0.0])
            c[0] += sum(1 for a in mine if g.t0 <= a[2] <= g.t1)
            c[1] += g.sec
    return events, normal


def per10(count, sec):
    return count / sec * 600.0 if sec > 0 else float("nan")


def _event_cell(items):
    if not items:
        return "-"
    hit = sum(d for d, _, _ in items)
    keep = np.mean([k for _, k, _ in items])
    lat = [x for *_, x in items if x is not None]
    lat_s = f"{np.median(lat):.1f}초" if lat else "-"
    return f"{hit}/{len(items)} · {keep:.0%} · {lat_s}"


def events_report(split, preds, th=None):
    """face_compare 보고서에 붙일 마크다운 (방식별 이벤트 지표 + 하위 태그별 10분당 오경보)"""
    th = th or FaceThresholds()
    res = {k: event_metrics(split, p, th) for k, p in preds.items()}
    tags = list(next(iter(res.values()))[1]) if res else []
    lines = ["## 이벤트 단위 지표 (지속 시간 필터 적용)", "",
             f"확정: 졸음 {th.drowsy_sec:g}초 · 하품 {th.yawn_sec:g}초 · 영역 밖 {th.away_sec:g}초, "
             f"해제 {th.release_sec:g}초. 이벤트 = 수집 단계 1개 구간.",
             "칸 = 탐지한 구간 / 전체 구간 · 경보 유지율 · 탐지 지연 중앙값", "",
             "| 방식 | " + " | ".join(FACE_KO[c] for c in ALARM) + " | 정상 10분당 오경보 |",
             "|---|" + "---|" * (len(ALARM) + 1)]
    for k, (events, normal) in res.items():
        n = sum(c for c, _ in normal.values())
        sec = sum(s for _, s in normal.values())
        lines.append(f"| {k} | " + " | ".join(_event_cell(events[c]) for c in ALARM)
                     + f" | {per10(n, sec):.1f} ({n}회) |")
    if tags:
        ko = lambda t: SUBTAG_KO.get(t) or FACE_KO.get(t, t)        # noqa: E731
        normal0 = next(iter(res.values()))[1]
        lines += ["", "### 정상 구간 10분당 오경보 (하위 태그별)", "",
                  "| 방식 | " + " | ".join(f"{ko(t)} ({normal0[t][1] / 60:.1f}분)" for t in tags) + " |",
                  "|---|" + "---|" * len(tags)]
        for k, (_, normal) in res.items():
            lines.append(f"| {k} | " + " | ".join(
                f"{per10(*normal[t]):.1f} ({normal[t][0]})" for t in tags) + " |")
    return "\n".join(lines)