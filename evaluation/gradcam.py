"""Grad-CAM: 이미지 모델이 판정할 때 이미지의 어느 부분을 근거로 삼았는지 히트맵으로 확인

같은 프레임에 대해 여러 이미지 모델(원본 / crop / 배경 제거)의 히트맵을 나란히 붙여 저장한다.

프로젝트 루트에서 실행:
  python -m evaluation.gradcam
  python -m evaluation.gradcam --per-class 4
  python -m evaluation.gradcam --models models/v3/cnn_crop.pt models/v3/cnn_cropmask.pt
결과: eval/gradcam/gradcam_<테스트 폴더 이름>.jpg (얼굴 이미지 포함 → git 제외 폴더)
  행 = 프레임 (자세별 N장, 약하게 → 심하게 순서로 고르게)
  열 = 원본 이미지(정답) + 모델별 히트맵 (예측과 확률, 맞으면 초록 / 틀리면 빨강)
히트맵은 MobileNetV3 마지막 특징 지도(약 4×5칸)를 키운 것이라 대략적인 위치만 나타낸다.
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from learning.image_model import FrameDataset, load_image_model
from learning.labels import DEFAULT, SCHEMES, classes
from learning.posture_dataset import load_splits

CELL_W, CELL_H, BAR = 160, 120, 14
GREEN, RED, WHITE = (0, 200, 0), (0, 0, 230), (255, 255, 255)
SHORT = {"good": "good", "forward_lean": "lean", "forward_head": "fhead", "slouch": "slouch",
         "head_tilt": "head", "shoulder_tilt": "shoulder"}


def parse_args():
    root = Path(__file__).resolve().parents[1]
    v3 = root / "models" / "v3"
    ap = argparse.ArgumentParser(description="이미지 모델 Grad-CAM 비교")
    ap.add_argument("--data", nargs="+", default=[str(root / "data"), str(root / "eval" / "new_env")],
                    help="테스트 세션이 있는 폴더들 (폴더마다 결과 이미지 1장)")
    ap.add_argument("--models", nargs="+",
                    default=[str(v3 / f"cnn_{n}.pt") for n in ("images", "crop", "cropmask")])
    ap.add_argument("--labels", choices=list(SCHEMES), default=DEFAULT)
    ap.add_argument("--per-class", type=int, default=3, help="자세별로 고를 프레임 수")
    ap.add_argument("--out", default=str(root / "eval" / "gradcam"))
    return ap.parse_args()


class GradCAM:
    """마지막 특징 지도(활성값)와 그 기울기로 판정 근거 위치를 계산"""

    def __init__(self, model):
        self.model, self.acts, self.grads = model, None, None
        model.features[0][-1].register_forward_hook(self._keep)   # 특징 추출부의 마지막 블록 (576채널)

    def _keep(self, module, inputs, output):
        self.acts = output
        output.register_hook(lambda g: setattr(self, "grads", g))

    def __call__(self, img, num):
        """→ (히트맵 0~1, 예측 클래스, 확률)"""
        self.model.zero_grad()
        out = self.model(img[None], num[None])
        prob = torch.softmax(out, 1)[0]
        cls = int(prob.argmax())
        out[0, cls].backward()
        w = self.grads.mean(dim=(2, 3), keepdim=True)              # 채널별 중요도
        cam = torch.relu((w * self.acts).sum(1))[0].detach().cpu().numpy().astype(np.float32)
        cam = cam / cam.max() if cam.max() > 0 else cam
        return cam, cls, float(prob[cls])


def title_bar(text, color=WHITE):
    bar = np.zeros((BAR, CELL_W, 3), np.uint8)
    cv2.putText(bar, text, (3, 11), cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1, cv2.LINE_AA)
    return bar


def read_cell(path):
    img = cv2.imread(path)
    return np.zeros((CELL_H, CELL_W, 3), np.uint8) if img is None else cv2.resize(img, (CELL_W, CELL_H))


def overlay(path, cam):
    heat = cv2.applyColorMap((cv2.resize(cam, (CELL_W, CELL_H)) * 255).astype(np.uint8), cv2.COLORMAP_JET)
    return cv2.addWeighted(read_cell(path), 0.55, heat, 0.45, 0)


def pick_frames(y, n_cls, per_class):
    """자세별로 프레임을 시간 순서에서 고르게 고름 (수집 순서상 약하게 → 심하게가 고루 섞임)"""
    picks = []
    for c in range(n_cls):
        idx = np.flatnonzero(y == c)
        if len(idx):
            picks += idx[np.linspace(0, len(idx) - 1, min(per_class, len(idx))).astype(int)].tolist()
    return picks


def load_models(paths, names, device):
    loaded = []
    for path in paths:
        if not Path(path).is_file():
            print(f"{path}: 파일이 없어 제외합니다")
            continue
        model, _, mean, std, mcls, idir = load_image_model(path, device)
        if list(mcls) != list(names):
            print(f"{path}: 모델 라벨 {mcls}가 지금 라벨 체계와 달라 제외합니다")
            continue
        loaded.append({"name": Path(path).stem.replace("cnn_", ""), "cam": GradCAM(model),
                       "mean": mean, "std": std, "image_dir": idir})
    if not loaded:
        raise SystemExit("사용할 이미지 모델이 없습니다.")
    return loaded


def render_session(test, picks, loaded, names, device):
    """→ (결과 이미지, {모델 이름: 맞힌 수})"""
    sets = {m["name"]: FrameDataset(test.with_image_dir(m["image_dir"]), m["mean"], m["std"], train=False)
            for m in loaded}
    rows = [np.hstack([title_bar("original")] + [title_bar(m["name"]) for m in loaded])]
    correct = {m["name"]: 0 for m in loaded}
    for i in picks:
        truth = int(test.y[i])
        cells = [np.vstack([title_bar(f"truth: {SHORT.get(names[truth], names[truth])}"),
                            read_cell(test.images[i])])]
        for m in loaded:
            ds = sets[m["name"]]
            img, num, _ = ds[i]
            cam, cls, p = m["cam"](img.to(device), num.to(device))
            ok = cls == truth
            correct[m["name"]] += ok
            bar = title_bar(f"{SHORT.get(names[cls], names[cls])} {p:.2f}", GREEN if ok else RED)
            cells.append(np.vstack([bar, overlay(ds.split.images[i], cam)]))
        rows.append(np.hstack(cells))
    return np.vstack(rows), correct


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    names = classes(args.labels)
    loaded = load_models(args.models, names, device)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    for d in args.data:
        _, _, test = load_splits(d, images_only=True, scheme=args.labels, verbose=False)
        missing = [m["name"] for m in loaded if test.with_image_dir(m["image_dir"]).missing_images()]
        if missing:
            print(f"{d}: {missing} 모델의 입력 이미지가 없어 건너뜁니다 (python -m learning.crop_images 필요)")
            continue
        picks = pick_frames(test.y, len(names), args.per_class)
        image, correct = render_session(test, picks, loaded, names, device)
        path = out / f"gradcam_{Path(d).name}.jpg"
        cv2.imwrite(str(path), image, [cv2.IMWRITE_JPEG_QUALITY, 92])
        summary = "  ".join(f"{k} {v}/{len(picks)}" for k, v in correct.items())
        print(f"저장: {path}  (고른 프레임 {len(picks)}장 중 맞힌 수: {summary})")


if __name__ == "__main__":
    main()