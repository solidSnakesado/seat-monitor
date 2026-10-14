"""이미지 모델: 방식 C (이미지만 → CNN), 방식 D (이미지 + 수치 → 융합)

백본: MobileNetV3-Small (ImageNet 사전학습, 약 250만 파라미터)
  → 직접 수집한 적은 이미지로도 미세조정이 가능하고, 4GB VRAM에서 충분히 학습된다.
좌우 반전 같은 기하 변형은 쓰지 않는다 (기울기 방향·위치 정보가 바뀌기 때문). 밝기·대비 변형만 사용.
"""
import numpy as np
import torch
import torch.nn as nn
import torchvision
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T

from learning.posture_model import CLASSES, INPUTS

IMG_HW = (120, 160)                       # 수집 시 저장 크기 (세로, 가로)
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
KINDS = ("cnn", "fusion")


def backbone(pretrained):
    weights = torchvision.models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
    net = torchvision.models.mobilenet_v3_small(weights=weights)
    return nn.Sequential(net.features, net.avgpool, nn.Flatten()), 576   # (특징 추출기, 출력 차원)


class PostureCNN(nn.Module):
    """방식 C: 이미지 → 자세"""

    def __init__(self, pretrained=True, n_cls=len(CLASSES)):
        super().__init__()
        self.features, d = backbone(pretrained)
        self.head = nn.Sequential(nn.Dropout(0.2), nn.Linear(d, n_cls))

    def forward(self, img, num=None):
        return self.head(self.features(img))


class FusionNet(nn.Module):
    """방식 D: 이미지 특징(64) + 수치 특징(32) 이어 붙이기 → 자세 (intermediate fusion)"""

    def __init__(self, pretrained=True, n_num=len(INPUTS), n_cls=len(CLASSES)):
        super().__init__()
        self.features, d = backbone(pretrained)
        self.img_proj = nn.Sequential(nn.Linear(d, 64), nn.ReLU())
        self.num_proj = nn.Sequential(nn.Linear(n_num, 32), nn.ReLU(), nn.Linear(32, 32), nn.ReLU())
        self.head = nn.Sequential(nn.Dropout(0.2), nn.Linear(64 + 32, n_cls))

    def forward(self, img, num):
        return self.head(torch.cat([self.img_proj(self.features(img)), self.num_proj(num)], 1))


def build(kind, pretrained=True, n_cls=len(CLASSES)):
    if kind not in KINDS:
        raise ValueError(f"kind는 {KINDS} 중 하나")
    return PostureCNN(pretrained, n_cls=n_cls) if kind == "cnn" else FusionNet(pretrained, n_cls=n_cls)


def make_transform(train):
    aug = [T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2)] if train else []
    return T.Compose([T.Resize(IMG_HW), *aug, T.ToTensor(), T.Normalize(MEAN, STD)])


class FrameDataset(Dataset):
    """Split → (이미지 텐서, 정규화된 수치, 라벨)"""

    def __init__(self, split, num_mean, num_std, train):
        self.split = split
        self.num = ((split.X - num_mean) / num_std).astype(np.float32)
        self.tf = make_transform(train)

    def __len__(self):
        return len(self.split)

    def __getitem__(self, i):
        img = Image.open(self.split.images[i]).convert("RGB")
        return self.tf(img), torch.from_numpy(self.num[i]), int(self.split.y[i])


def save_image_model(path, kind, model, num_mean, num_std, classes=CLASSES, image_dir="images"):
    """image_dir: 학습에 쓴 이미지 폴더 (images = 원본, images_crop = 사람 영역 crop)"""
    torch.save({"kind": kind, "state": model.state_dict(), "classes": list(classes),
                "inputs": list(INPUTS), "num_mean": num_mean.tolist(), "num_std": num_std.tolist(),
                "image_dir": image_dir}, path)


def load_image_model(path, device="cpu"):
    """→ (model, kind, num_mean, num_std, classes, image_dir). 사전학습 가중치는 다시 받지 않는다"""
    ck = torch.load(path, map_location=device)
    model = build(ck["kind"], pretrained=False, n_cls=len(ck["classes"]))
    model.load_state_dict(ck["state"])
    model.to(device).eval()
    return (model, ck["kind"], np.array(ck["num_mean"], dtype=np.float32),
            np.array(ck["num_std"], dtype=np.float32), ck["classes"], ck.get("image_dir", "images"))


@torch.no_grad()
def image_probs(model, loader, device="cpu"):
    """DataLoader 전체 추론 → (N, 클래스 수) 확률"""
    model.eval()
    out = []
    for img, num, _ in loader:
        out.append(torch.softmax(model(img.to(device), num.to(device)), 1).cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, len(CLASSES)))