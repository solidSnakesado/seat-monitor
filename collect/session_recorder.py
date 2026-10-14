"""세션 기록: 프레임마다 수치 + 라벨 + 강도를 CSV로, N프레임마다 같은 프레임 이미지를 저장

CSV 열: t, label, intensity, ear_shoulder, head_ratio, shoulder_y, shoulder_tilt, ear_tilt, image
  - intensity: none(바른 자세) / mild / mid / severe, 수동 녹화는 빈칸
  - image: CSV 기준 상대 경로, 저장하지 않은 프레임은 빈칸
"""
import csv
from datetime import datetime
from pathlib import Path

import cv2

from learning.posture_model import CLASSES, INPUTS

IMAGE_SIZE = (160, 120)      # 저장 이미지 크기 (가로, 세로)
KO = {"good": "바른 자세", "forward_head": "거북목", "slouch": "구부정",
      "head_tilt": "고개 기울임", "shoulder_tilt": "어깨 비대칭"}


class Recorder:
    def __init__(self, out_dir, img_every, test, tag):
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = f"posture_session_{stamp}" + (f"_{tag}" if tag else "") + ("_test" if test else "")
        self.root = Path(out_dir)
        self.path = self.root / f"{name}.csv"
        self.img_dir = Path("images") / name
        self.img_every = img_every
        self.counts = {c: 0 for c in CLASSES}
        self.n_images = 0
        self._file = self._writer = None
        self._row = 0
        self.root.mkdir(parents=True, exist_ok=True)

    @property
    def is_test(self):
        return self.path.stem.endswith("_test")

    def write(self, t, label, intensity, d, clean_frame):
        if self._writer is None:
            self._file = open(self.path, "w", newline="")
            self._writer = csv.writer(self._file)
            self._writer.writerow(["t", "label", "intensity", *INPUTS, "image"])
        image = ""
        if self.img_every > 0 and self._row % self.img_every == 0:
            image = self._save_image(clean_frame)
        self._writer.writerow([f"{t:.3f}", label, intensity,
                               *[f"{d[k]:.5f}" for k in INPUTS], image])
        self.counts[CLASSES[label]] += 1
        self._row += 1

    def _save_image(self, frame):
        rel = self.img_dir / f"{self._row:06d}.jpg"
        (self.root / rel).parent.mkdir(parents=True, exist_ok=True)
        small = cv2.resize(frame, IMAGE_SIZE, interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(self.root / rel), small, [cv2.IMWRITE_JPEG_QUALITY, 90])
        self.n_images += 1
        return rel.as_posix()

    def close(self):
        if self._file:
            self._file.close()
            print(f"저장: {self.path}  (이미지 {self.n_images}장)")
            for c, n in self.counts.items():
                print(f"  {KO[c]:<8} {n:6d} 프레임")