"""실시간 자세 판정 (최종 구성 v6): B 수치 MLP + C 이미지 앙상블 → 가중 결합 → F2 각도 보정

  결합 확률 = (1 - w) × B 확률 + w × 이미지 모델들의 평균 확률   (w = 이미지 비중, 기본 0.6)
  F2: 목 꺾임(귀 기울기 - 어깨 기울기)이 10° 이상이면 고개 기울임

이미지는 학습 데이터와 같은 과정으로 만든다 (learning/crop_images.py의 함수를 그대로 사용):
  프레임을 수집 때 저장 크기(160×120)로 줄임 → 어깨 폭 기준 상반신 crop
  → 배경 제거(images_cropmask) 또는 실루엣(images_sil). 모델마다 학습한 이미지 종류에 맞춰 넣는다.
"""
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from common.posture_judge import Thresholds
from learning.crop_images import apply_mask, crop, crop_box, to_silhouette
from learning.image_model import load_image_model, make_transform
from learning.labels import classes, ko_names
from learning.posture_model import INPUTS, load_mlp, to_vector

ROOT = Path(__file__).resolve().parents[1]
V6 = ROOT / "models" / "v6"
DEFAULT_MLP = str(V6 / "posture_mlp.pt")
DEFAULT_CNNS = [str(V6 / f"cnn_{k}{s}.pt") for k in ("cropmask_reg", "sil") for s in ("", "_s1", "_s2")]
SMALL = (160, 120)                 # 수집 때 저장한 이미지 크기 (collect/session_recorder.py)
IMAGE_KINDS = ("images_cropmask", "images_sil")
SCHEME = "merged"


class PostureEnsemble:
    def __init__(self, mlp_path=DEFAULT_MLP, cnn_paths=DEFAULT_CNNS, img_weight=0.6, smooth=0.3, img_every=1):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.names = classes(SCHEME)
        self.labels = dict(zip(self.names, ko_names(SCHEME)))
        self.mlp, self.mean, self.std, mcls = load_mlp(mlp_path)
        self.cnns = []
        for p in cnn_paths:
            model, _, _, _, icls, idir = load_image_model(p, self.device)
            if list(icls) != self.names or idir not in IMAGE_KINDS:
                raise SystemExit(f"{p}: 라벨 {icls} / 이미지 {idir}는 실시간 구성에서 쓸 수 없습니다")
            self.cnns.append((model, idir))
        if list(mcls) != self.names:
            raise SystemExit(f"{mlp_path}: 라벨 {mcls}가 {self.names}와 다릅니다 (merged로 학습한 모델 필요)")
        self.tf = make_transform(train=False)
        self.w, self.smooth, self.img_every = img_weight, smooth, max(1, img_every)
        self.head_deg = Thresholds().head_tilt_deg
        self.reset()

    def reset(self):
        self._prob, self._pc, self._n = None, None, 0
        self.inputs = {}                    # 마지막으로 모델에 넣은 이미지 (화면 확인용, BGR)
        self.last = ("good", 1.0, False)    # (클래스, 확률, 각도 보정 여부)

    def _make_inputs(self, frame, lms, mask):
        """깨끗한 프레임 → {이미지 종류: BGR 160×120}. 어깨가 안 보이거나 윤곽이 없으면 {}"""
        small = cv2.resize(frame, SMALL, interpolation=cv2.INTER_AREA)
        box = crop_box(lms, *SMALL)
        if box is None or mask is None:
            return {}
        m = cv2.resize(np.squeeze(mask).astype(np.float32), SMALL, interpolation=cv2.INTER_LINEAR)
        return {"images_cropmask": crop(apply_mask(small, m), box), "images_sil": crop(to_silhouette(small, m), box)}

    @torch.no_grad()
    def _image_probs(self, frame, lms, mask):
        self.inputs = self._make_inputs(frame, lms, mask)
        if not self.inputs:
            return None
        x = {k: self.tf(Image.fromarray(cv2.cvtColor(v, cv2.COLOR_BGR2RGB)))[None].to(self.device)
             for k, v in self.inputs.items()}
        num = torch.zeros((1, len(INPUTS)), device=self.device)      # 이미지 모델은 수치 입력을 쓰지 않음
        return np.mean([torch.softmax(m(x[k], num), 1)[0].cpu().numpy() for m, k in self.cnns], axis=0)

    @torch.no_grad()
    def predict(self, deltas, frame, lms, mask):
        """→ (클래스, 확률, 각도 보정 여부). 이미지를 만들 수 없는 프레임은 B만 사용"""
        x = torch.tensor((to_vector(deltas) - self.mean) / self.std)[None]
        pb = torch.softmax(self.mlp(x), 1)[0].numpy()
        if self.cnns and self._n % self.img_every == 0:              # img_every 프레임마다 이미지 판정 갱신
            self._pc = self._image_probs(frame, lms, mask)
        self._n += 1
        p = pb if self._pc is None else (1 - self.w) * pb + self.w * self._pc
        self._prob = p if self._prob is None else (1 - self.smooth) * self._prob + self.smooth * p
        label = self.names[int(self._prob.argmax())]
        by_angle = abs(deltas["ear_tilt"] - deltas["shoulder_tilt"]) >= self.head_deg   # F2 목 꺾임 보정
        if by_angle:
            label = "head_tilt"
        self.last = (label, float(self._prob[self.names.index(label)]), by_angle)
        return self.last

    def issues(self, deltas, frame, lms, mask):
        """detect_issues()와 같은 형태 (빈 목록 = 바른 자세)"""
        label, _, _ = self.predict(deltas, frame, lms, mask)
        return [] if label == "good" else [label]

    def describe(self):
        """화면 표시용 짧은 설명"""
        if not self.cnns:
            return "수치 MLP만"
        return f"수치 + 이미지 {len(self.cnns)}개, 이미지 비중 {self.w:g} ({self.device})"