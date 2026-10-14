"""자세 분류 딥러닝 모델 (MLP)

입력: 기준 자세 대비 변화량 5개 (posture_judge.compute_deltas 결과)
출력: 바른 자세 / 거북목 / 구부정 / 고개 기울임 / 어깨 비대칭

규칙(detect_issues)의 임계값 대신, 직접 수집한 데이터로 경계를 학습한다.
"""
import numpy as np
import torch
import torch.nn as nn

CLASSES = ["good", "forward_head", "slouch", "head_tilt", "shoulder_tilt"]
INPUTS = ("ear_shoulder", "head_ratio", "shoulder_y", "shoulder_tilt", "ear_tilt")


class PostureMLP(nn.Module):
    def __init__(self, n_in=len(INPUTS), n_cls=len(CLASSES), hidden=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, hidden), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_cls),
        )

    def forward(self, x):
        return self.net(x)


def to_vector(deltas):
    return np.array([deltas[k] for k in INPUTS], dtype=np.float32)


def save_model(path, model, mean, std, classes=CLASSES):
    """가중치 + 정규화 값 + 클래스를 한 파일에 저장 (학습·추론 전처리 일치)
    classes: 학습에 쓴 라벨 체계의 클래스 목록 (learning/labels.py)"""
    torch.save({"state": model.state_dict(), "mean": mean.tolist(), "std": std.tolist(),
                "classes": list(classes), "inputs": list(INPUTS)}, path)


def load_mlp(path):
    """→ (model, mean, std, classes)"""
    ck = torch.load(path, map_location="cpu")
    if tuple(ck["inputs"]) != INPUTS:
        raise ValueError("모델 입력 항목이 현재 코드와 다릅니다. 다시 학습하세요.")
    model = PostureMLP(len(INPUTS), len(ck["classes"]))
    model.load_state_dict(ck["state"])
    model.eval()
    return (model, np.array(ck["mean"], dtype=np.float32),
            np.array(ck["std"], dtype=np.float32), ck["classes"])


def mlp_probs(model, mean, std, X):
    """여러 프레임 한 번에 추론 (평가용) → (N, 클래스 수) 확률"""
    with torch.no_grad():
        x = torch.tensor((np.asarray(X, dtype=np.float32) - mean) / std)
        return torch.softmax(model(x), 1).numpy()


class PosturePredictor:
    """실시간 추론: 확률을 지수평활해 프레임 간 흔들림을 줄인다"""

    def __init__(self, path, smooth=0.3):
        self.model, self.mean, self.std, self.classes = load_mlp(path)
        self.smooth = smooth
        self.reset()

    def reset(self):
        self._prob = None
        self.last = ("good", 1.0)

    def predict(self, deltas):
        """→ (클래스, 확률)"""
        x = torch.tensor((to_vector(deltas) - self.mean) / self.std)[None]
        with torch.no_grad():
            p = torch.softmax(self.model(x), 1)[0].numpy()
        self._prob = p if self._prob is None else (1 - self.smooth) * self._prob + self.smooth * p
        i = int(self._prob.argmax())
        self.last = (self.classes[i], float(self._prob[i]))
        return self.last

    def issues(self, deltas):
        """detect_issues()와 같은 형태로 반환 (빈 목록 = 바른 자세)"""
        label, _ = self.predict(deltas)
        return [] if label == "good" else [label]