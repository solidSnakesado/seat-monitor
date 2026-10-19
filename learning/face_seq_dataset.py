"""얼굴 시계열 데이터셋: 프레임마다 '최근 N프레임(기본 30, 약 1.2초)'의 수치 56개를 묶어 방식 T의 입력을 만든다

깜빡임과 졸음(눈 감김), 말하기와 하품처럼 한 순간의 모습은 같지만 시간에 따른 변화가 다른 상태를 구분하기 위함.
  - 라벨 = 묶음의 마지막 프레임 라벨 (지금 프레임을 과거 N프레임을 보고 판정)
  - 묶음은 같은 세션의 같은 구간(라벨·하위 태그가 이어진 행들) 안에서만 만든다
  - 구간 앞쪽처럼 이전 프레임이 모자라면 구간의 첫 프레임을 반복해 채운다
    → 모든 프레임에 T 판정이 생겨 A~E와 같은 프레임으로 비교할 수 있다
  - 묶음을 미리 만들지 않고 프레임 번호(N × win)만 저장해 학습 중에 꺼낸다 (메모리 절약)
입력 행은 learning.face_dataset.load_face_splits의 FaceSplit 그대로 (세션 순서·프레임 순서 유지, face=0 행 제외).
"""
import numpy as np
import torch
from torch.utils.data import Dataset

from learning.face_dataset import NUM_INPUTS

DEFAULT_WIN = 30


def segment_ids(split):
    """세션·라벨·하위 태그가 바뀌는 곳에서 새 구간 → (N,) 구간 번호"""
    seg = np.zeros(len(split), dtype=np.int64)
    for i in range(1, len(split)):
        same = (split.sessions[i] == split.sessions[i - 1] and split.y[i] == split.y[i - 1]
                and split.subtags[i] == split.subtags[i - 1])
        seg[i] = seg[i - 1] + (0 if same else 1)
    return seg


def window_index(split, win=DEFAULT_WIN):
    """→ (N, win) 프레임 번호. i번째 행 = [i − win + 1, …, i], 구간 시작보다 앞은 구간 첫 프레임으로"""
    n = len(split)
    if n == 0:
        return np.zeros((0, win), dtype=np.int64)
    seg = segment_ids(split)
    first = np.zeros(n, dtype=np.int64)                 # 각 행이 속한 구간의 첫 행 번호
    starts = np.flatnonzero(np.r_[True, seg[1:] != seg[:-1]])
    first[starts] = starts
    first = np.maximum.accumulate(first)
    idx = np.arange(n)[:, None] + np.arange(-win + 1, 1)[None, :]
    return np.maximum(idx, first[:, None])


class SeqDataset(Dataset):
    """→ (묶음 (win, 56) 정규화 수치, 라벨)"""

    def __init__(self, split, mean, std, win=DEFAULT_WIN):
        self.x = torch.from_numpy(((split.X - mean) / std).astype(np.float32))
        self.idx = torch.from_numpy(window_index(split, win))
        self.y = torch.from_numpy(np.asarray(split.y, dtype=np.int64))

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.x[self.idx[i]], self.y[i]


def num_stats(X):
    mean, std = X.mean(0), X.std(0)
    std[std < 1e-6] = 1.0
    return mean.astype(np.float32), std.astype(np.float32)


def angle_offset(std, scale_deg, offset_aug):
    """각도 오프셋 증강의 최대 크기 (정규화 단위, (56,)). 묶음마다 같은 오프셋을 더해 캘리브레이션 어긋남을 흉내"""
    off = np.array([scale_deg.get(k, 0.0) for k in NUM_INPUTS], dtype=np.float32) * offset_aug / std
    return torch.from_numpy(off.astype(np.float32))