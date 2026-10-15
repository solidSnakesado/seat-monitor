"""데이터셋 공통: 세션 CSV 로드와 학습 / 검증 / 테스트 분할

모든 학습·비교 스크립트가 이 모듈로 같은 방식으로 나눈다 (공정한 비교의 조건).
  - 파일 이름에 '_test'가 있는 세션 → 테스트 세트 (학습·모델 선택에 사용 안 함)
  - 나머지 세션 → 같은 (라벨, 강도)가 이어진 구간마다 끝 val_ratio를 검증, 나머지를 학습
  - val_sessions를 주면 → 그 세션들을 통째로 검증, 나머지 세션은 전부 학습 (세션 단위 검증)
    같은 세션의 뒷부분은 학습 데이터와 거의 같아 점수가 부풀려지므로, 처음 보는 세션으로 모델을 고르기 위함
  - images_only=True → 이미지가 함께 저장된 프레임만 사용 (네 방식을 같은 프레임으로 비교)
  - scheme → 분할은 수집 라벨(5개) 기준으로 한 뒤, 라벨만 선택한 체계로 변환 (learning/labels.py)
"""
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from learning.labels import convert
from learning.posture_model import INPUTS


@dataclass
class Split:
    X: np.ndarray        # (N, 5) 기준 대비 변화량
    y: np.ndarray        # (N,) 라벨 인덱스
    images: list         # 이미지 절대 경로, 없으면 ""
    intensity: list      # none / mild / mid / severe, 기록 없으면 ""

    def __len__(self):
        return len(self.y)

    def subset(self, idx):
        idx = np.asarray(idx, dtype=int)
        return Split(self.X[idx], self.y[idx], [self.images[i] for i in idx],
                     [self.intensity[i] for i in idx])

    def with_images(self):
        return self.subset([i for i, p in enumerate(self.images) if p])

    def with_image_dir(self, name):
        """이미지 경로의 images 폴더를 다른 폴더로 바꾼다 (예: images_crop, learning/crop_images.py)
        경로 구조: <CSV 폴더>/images/<세션>/000123.jpg"""
        if name == "images":
            return self

        def swap(p):
            if not p:
                return p
            parts = list(Path(p).parts)
            parts[-3] = name
            return str(Path(*parts))
        return Split(self.X, self.y, [swap(p) for p in self.images], self.intensity)

    def missing_images(self):
        return [p for p in self.images if p and not Path(p).exists()]

    def fraction(self, frac, seed=0):
        """데이터 효율 실험용: 클래스 비율을 유지하며 일부만 사용"""
        if frac >= 1.0:
            return self
        rng = np.random.default_rng(seed)
        keep = []
        for c in np.unique(self.y):
            idx = np.flatnonzero(self.y == c)
            keep += rng.choice(idx, max(1, int(len(idx) * frac)), replace=False).tolist()
        return self.subset(sorted(keep))


def _concat(parts):
    if not parts:
        return Split(np.zeros((0, len(INPUTS)), np.float32), np.zeros(0, int), [], [])
    return Split(np.concatenate([p.X for p in parts]), np.concatenate([p.y for p in parts]),
                 sum((p.images for p in parts), []), sum((p.intensity for p in parts), []))


def label_runs(keys):
    """같은 (라벨, 강도)가 연속된 구간들 → [(시작, 끝)]
    강도까지 나눠야 '약하게→중간→심하게' 중 심하게만 검증으로 가는 일이 없다"""
    keys = list(keys)
    starts = [0] + [i for i in range(1, len(keys)) if keys[i] != keys[i - 1]]
    return list(zip(starts, starts[1:] + [len(keys)]))


def load_session(path):
    path = Path(path)
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    X = np.array([[float(r[k]) for k in INPUTS] for r in rows], dtype=np.float32)
    y = np.array([int(r["label"]) for r in rows], dtype=int)
    images = [str((path.parent / r["image"]).resolve()) if r.get("image") else "" for r in rows]
    intensity = [r.get("intensity", "") or "" for r in rows]    # 이전 형식 CSV는 빈칸
    return Split(X, y, images, intensity)


def relabel(split, scheme):
    return Split(split.X, convert(split.y, scheme), split.images, split.intensity)


def load_splits(data_dir, val_ratio=0.2, images_only=False, verbose=True, scheme="fine", val_sessions=None):
    """→ (train, val, test). 테스트 세션이 없으면 검증 세트를 테스트로 대신 사용
    scheme: 라벨 체계 (fine = 수집 라벨 그대로 5개, merged = 거북목·구부정 합침)
    val_sessions: 파일 이름에 이 문자열이 들어간 세션을 통째로 검증에 사용 (예: ["round_stripe"])"""
    files = sorted(Path(data_dir).glob("posture_session_*.csv"))
    if not files:
        raise SystemExit(f"{data_dir}/posture_session_*.csv 가 없습니다. "
                         f"python -m collect.posture_collect로 수집하세요.")
    held = [p for p in files if not p.stem.endswith("_test") and any(t in p.stem for t in val_sessions or [])]
    if val_sessions and not held:
        raise SystemExit(f"검증 세션 {val_sessions}에 맞는 세션이 없습니다: {[p.stem[16:] for p in files]}")
    train, val, test = [], [], []
    for path in files:
        s = load_session(path)
        if path.stem.endswith("_test"):
            test.append(s)
            continue
        if held:                                         # 세션 단위 검증
            (val if path in held else train).append(s)
            continue
        for a, b in label_runs(zip(s.y, s.intensity)):   # (라벨, 강도) 구간마다 끝부분을 검증으로
            cut = a + int((b - a) * (1 - val_ratio))
            train.append(s.subset(range(a, cut)))
            val.append(s.subset(range(cut, b)))
    train, val, test = (relabel(_concat(s), scheme) for s in (train, val, test))
    if images_only:
        train, val, test = train.with_images(), val.with_images(), test.with_images()

    if verbose:
        n_test = sum(1 for p in files if p.stem.endswith("_test"))
        print(f"세션 {len(files)}개 (테스트 {n_test}개) | 학습 {len(train)} / 검증 {len(val)} / "
              f"테스트 {len(test)} 프레임  [라벨 {scheme}]" + ("  [이미지 있는 프레임만]" if images_only else "")
              + (f"  [검증 세션: {', '.join(p.stem[16:] for p in held)}]" if held else ""))
    if len(test) == 0:
        print("주의: 테스트 세션이 없어 검증 세트로 테스트합니다 (--test로 따로 수집 권장)")
        test = val
    return train, val, test