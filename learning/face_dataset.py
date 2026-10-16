"""얼굴 데이터셋: 세션 CSV + 영역 JSON → 특징 행렬, 세션 단위 학습 / 검증 / 테스트 분할

  - data/face/face_session_*.csv (collect.face_collect로 수집), 같은 이름의 .json = 영역 캘리브레이션
  - 파일 이름에 '_test'가 있는 세션 → 테스트 (학습·모델 선택에 사용 안 함)
  - val_sessions에 이름 일부가 맞는 세션 → 통째로 검증, 나머지 → 학습
  - face=0(얼굴 미검출) 행은 제외. face 열이 없는 이전 세션은 모두 검출 프레임으로 본다
특징 (방식 B, D의 수치 입력):
  blendshape 51개(_neutral 제외), 화면 대비 yaw·pitch, roll,
  앞쪽 구간 밖 거리(yaw가 화면·경계 A·B를 포함한 범위 밖으로 벗어난 각도), 책상 대비 pitch
방식 A(MediaPipe만)는 현재 실시간 규칙(common/face_judge.py 임계값)을 같은 프레임에 적용한다.
"""
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from collect.face_guide import FACE_CLASSES
from collect.face_recorder import BLENDSHAPES
from common.face_features import GAZE_KEYS
from common.face_judge import FaceThresholds

BS_INPUTS = tuple(n for n in BLENDSHAPES if n != "_neutral")
ANGLE_INPUTS = ("d_yaw", "d_pitch", "roll", "out_front", "d_desk")
NUM_INPUTS = BS_INPUTS + ANGLE_INPUTS
ALARM = ("away", "drowsy", "yawn")          # 헷갈리는 동작에서 이 판정이 나오면 오경보


@dataclass
class FaceSplit:
    X: np.ndarray        # (N, len(NUM_INPUTS))
    y: np.ndarray        # (N,) FACE_CLASSES 인덱스
    sessions: list       # 세션 이름
    subtags: list        # 하위 태그 (screen, lecture, talk, ...), 없으면 ""
    images: list         # 얼굴 crop 이미지 절대 경로, 없으면 ""

    def __len__(self):
        return len(self.y)

    def subset(self, idx):
        idx = np.asarray(idx, dtype=int)
        return FaceSplit(self.X[idx], self.y[idx], [self.sessions[i] for i in idx],
                         [self.subtags[i] for i in idx], [self.images[i] for i in idx])

    def with_images(self):
        return self.subset([i for i, p in enumerate(self.images) if p])

    def missing_images(self):
        return [p for p in self.images if p and not Path(p).exists()]


def _front_range(zones):
    """앞쪽 구간 yaw 범위: 화면과 경계 A·B를 모두 포함 (강의 공간이 없으면 화면 방향 한 점)"""
    yaws = [zones[k]["yaw"] for k in ("screen", "edge_a", "edge_b") if k in zones]
    return min(yaws), max(yaws)


def _session_rows(csv_path):
    zones = json.loads(csv_path.with_suffix(".json").read_text(encoding="utf-8"))["zones"]
    scr, desk = zones["screen"], zones.get("desk", zones["screen"])
    lo, hi = _front_range(zones)
    X, y, subtags, images = [], [], [], []
    with open(csv_path, newline="") as f:
        for r in csv.DictReader(f):
            if r.get("face", "1") == "0" or not r["yaw"]:
                continue
            yaw, pitch = float(r["yaw"]), float(r["pitch"])
            angles = [yaw - scr["yaw"], pitch - scr["pitch"], float(r["roll"]),
                      max(lo - yaw, yaw - hi, 0.0), pitch - desk["pitch"]]
            X.append([float(r[n]) for n in BS_INPUTS] + angles)
            y.append(FACE_CLASSES.index(r["label"]))
            subtags.append(r["subtag"])
            images.append(str((csv_path.parent / r["image"]).resolve()) if r["image"] else "")
    return X, y, subtags, images


def load_face_splits(data_dir, val_sessions):
    """→ (train, val, test) FaceSplit. val_sessions: 검증 세션 이름 일부 목록"""
    paths = sorted(Path(data_dir).glob("face_session_*.csv"))
    if not paths:
        raise SystemExit(f"{data_dir}에 face_session_*.csv가 없습니다")
    parts = {"train": [], "val": [], "test": []}
    for p in paths:
        if not p.with_suffix(".json").exists():
            print(f"건너뜀 (영역 JSON 없음): {p.name}")
            continue
        kind = ("test" if p.stem.endswith("_test")
                else "val" if any(v in p.stem for v in (val_sessions or [])) else "train")
        X, y, subtags, images = _session_rows(p)
        parts[kind].append((p.stem, X, y, subtags, images))
        print(f"  [{kind:<5}] {p.stem}  {len(y)} 프레임")
    return tuple(_merge(parts[k]) for k in ("train", "val", "test"))


def _merge(items):
    n_in = len(NUM_INPUTS)
    X = np.array([x for _, xs, *_ in items for x in xs], dtype=np.float32).reshape(-1, n_in)
    y = np.array([v for _, _, ys, *_ in items for v in ys], dtype=np.int64)
    sessions = [name for name, _, ys, *_ in items for _ in ys]
    subtags = [s for *_, ss, _ in items for s in ss]
    images = [i for *_, ims in items for i in ims]
    return FaceSplit(X, y, sessions, subtags, images)


def rule_predict(X, th=None):
    """방식 A: MediaPipe 출력 + 현재 실시간 규칙 → FACE_CLASSES 인덱스
    졸음(눈 감김) > 하품(입 벌림) > 영역 밖(정면 대비 각도·시선 이탈) 순서, 해당 없으면 앞 보기.
    현재 규칙에는 책상 개념이 없어 책상 보기는 각도에 따라 앞 보기 또는 영역 밖이 된다."""
    th = th or FaceThresholds()
    col = {n: i for i, n in enumerate(NUM_INPUTS)}
    eye = (X[:, col["eyeBlinkLeft"]] + X[:, col["eyeBlinkRight"]]) / 2
    jaw = X[:, col["jawOpen"]]
    gaze = X[:, [col[k] for k in GAZE_KEYS]].max(1)
    closed = eye >= th.eye_closed
    head_off = (np.abs(X[:, col["d_yaw"]]) > th.yaw_deg) | (np.abs(X[:, col["d_pitch"]]) > th.pitch_deg)
    away = head_off | (~closed & (gaze > th.gaze))
    pred = np.full(len(X), FACE_CLASSES.index("front"))
    pred[away] = FACE_CLASSES.index("away")
    pred[jaw >= th.jaw_open] = FACE_CLASSES.index("yawn")
    pred[closed] = FACE_CLASSES.index("drowsy")
    return pred


def false_alarms(split, pred):
    """하위 태그별 (프레임 수, 오경보율). 오경보 = 앞 보기 구간을 영역 밖·졸음·하품으로 판정"""
    alarm = np.isin(pred, [FACE_CLASSES.index(c) for c in ALARM])
    out = {}
    for tag in dict.fromkeys(split.subtags):
        if not tag:
            continue
        m = np.array([s == tag for s in split.subtags])
        out[tag] = (int(m.sum()), float(alarm[m].mean()) if m.any() else float("nan"))
    return out


def zone_gate(probs, X, yaw_tol=5.0, pitch_tol=10.0):
    """영역 연결: 고개 방향이 화면·강의 공간 범위 안이면 이미지 모델의 '영역 밖' 확률을 '앞 보기'로 옮긴다.
    이미지는 고개를 돌렸는지는 보지만, 그 방향이 자리별 강의 공간인지는 알 수 없기 때문 (영역 캘리브레이션 정보).
    범위 안 = 앞쪽 구간 밖 거리 ≤ yaw_tol°, 그리고 화면보다 pitch_tol° 넘게 위를 보지 않음 (pitch는 아래가 +)"""
    col = {n: i for i, n in enumerate(NUM_INPUTS)}
    inside = (X[:, col["out_front"]] <= yaw_tol) & (X[:, col["d_pitch"]] >= -pitch_tol)
    out = probs.copy()
    f, a = FACE_CLASSES.index("front"), FACE_CLASSES.index("away")
    out[inside, f] += out[inside, a]
    out[inside, a] = 0.0
    return out