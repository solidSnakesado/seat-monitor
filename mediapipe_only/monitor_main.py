"""착석자 모니터링: 자세(거북목·구부정·기울임) + 얼굴(졸음·하품·전방 주시)

  python monitor_main.py
  python monitor_main.py --source http://<ESP32-IP>:81/stream --posture-hold 30
  python monitor_main.py --posture-model models/posture_mlp.pt   # 학습 모델로 자세 판정

필요 파일(같은 폴더): pose_landmarker_lite.task, face_landmarker.task
키: c 기준 자세 다시 저장, ESC 종료
"""
import argparse
import time
from pathlib import Path

import cv2

from face_features import FaceDetector, compute_face_features, draw_face_points
from face_judge import FACE_LABELS, FaceThresholds, detect_face_issues, face_deltas, make_face_filter
from korean_text import TextRenderer
from posture_features import FEATURE_NAMES, PoseDetector, Smoother, compute_features, draw_key_points
from posture_judge import (LABELS, Calibrator, DurationFilter, Thresholds,
                           compute_deltas, detect_issues)

GREEN, RED, YELLOW, WHITE = (0, 255, 0), (0, 0, 255), (0, 255, 255), (255, 255, 255)
CALIB_KEYS = FEATURE_NAMES + ("yaw", "pitch")      # 자세 기준 + 정면 고개 각도


def parse_args():
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description="착석자 모니터링")
    ap.add_argument("--source", default="0", help="웹캠 번호 또는 스트림 URL")
    ap.add_argument("--pose-model", default=str(here / "pose_landmarker_lite.task"))
    ap.add_argument("--face-model", default=str(here / "face_landmarker.task"))
    ap.add_argument("--calib-sec", type=float, default=5.0)
    ap.add_argument("--posture-hold", type=float, default=Thresholds.hold_sec,
                    help="자세 문제 확정 시간(초), 실사용 30 이상 권장")
    ap.add_argument("--posture-model", default="",
                    help="학습한 자세 모델(models/posture_mlp.pt). 비우면 규칙으로 판정")
    return ap.parse_args()


def load_posture_model(path):
    if not path:
        return None
    from posture_model import PosturePredictor     # torch는 모델을 쓸 때만 필요
    return PosturePredictor(path)


def status_line(title, confirmed, labels):
    if confirmed:
        return (f"{title}: " + ", ".join(labels[k] for k in confirmed), RED)
    return (f"{title}: 정상" if title == "얼굴" else f"{title}: 바른 자세", GREEN)


def pending_text(candidates, filt, labels, now):
    return [f"{labels[k]} {filt.elapsed(k, now):.1f}초" for k in candidates if k not in filt.confirmed]


def value_lines(f, base, d):
    lines = []
    if f is not None:
        fd = face_deltas(f, base)
        lines.append((f"얼굴  눈 감김 {f['eye_closed']:.2f}  입 벌림 {f['jaw_open']:.2f}  "
                      f"시선 {f['gaze']:.2f}  고개 좌우 {fd['yaw']:+.0f}°  상하 {fd['pitch']:+.0f}°", WHITE))
    if d is not None:
        lines.append((f"자세  귀-어깨 {d['ear_shoulder']:+.0%}  머리 비율 {d['head_ratio']:+.0%}  "
                      f"어깨 높이 {d['shoulder_y']:+.3f}", WHITE))
        lines.append((f"      어깨 기울기 {d['shoulder_tilt']:+.1f}°  귀 기울기 {d['ear_tilt']:+.1f}°", WHITE))
    return lines


class Monitor:
    """프레임 1장 처리: 검출 → 특징 → (캘리브레이션 | 판정) → 표시할 줄"""

    def __init__(self, args):
        self.pose = PoseDetector(args.pose_model)
        self.face = FaceDetector(args.face_model)
        self.smoother = Smoother(alpha=0.3)
        self.calib = Calibrator(seconds=args.calib_sec, keys=CALIB_KEYS)
        self.pth = Thresholds(hold_sec=args.posture_hold)
        self.fth = FaceThresholds()
        self.posture_filter = DurationFilter(self.pth.hold_sec, self.pth.release_sec)
        self.face_filter = make_face_filter(self.fth)
        self.posture_model = load_posture_model(args.posture_model)
        self.base = None

    def judge_posture(self, d):
        """학습 모델이 있으면 모델로, 없으면 규칙으로 자세 판정"""
        if self.posture_model is None:
            return detect_issues(d, self.pth)
        return self.posture_model.issues(d)

    def mode_line(self):
        if self.posture_model is None:
            return ("자세 판정 방식: 규칙", WHITE)
        label, p = self.posture_model.last
        return (f"자세 판정 방식: 학습 모델  ({label} {p:.2f})", WHITE)

    def recalibrate(self, now):
        self.smoother.reset()
        self.posture_filter.reset()
        self.face_filter.reset()
        if self.posture_model is not None:
            self.posture_model.reset()
        self.calib.start(now)

    def step(self, frame, now):
        h, w = frame.shape[:2]
        pose_lms = self.pose.detect(frame)
        face_res = self.face.detect(frame)
        p_raw = compute_features(pose_lms, w, h)
        p = self.smoother.update(p_raw)
        f = compute_face_features(face_res)
        draw_key_points(frame, pose_lms)
        draw_face_points(frame, face_res)

        if p_raw is None and f is None:                 # 사람 없음
            self.posture_filter.reset()
            self.face_filter.reset()
            if self.calib.active:
                self.calib.add(None, now)
            return [("자리 비움: 사람이 화면에 없습니다", YELLOW)]

        if self.calib.active:
            sample = {**p, "yaw": f["yaw"], "pitch": f["pitch"]} if (p_raw and f) else None
            b = self.calib.add(sample, now)
            if b is not None:
                self.base = b
            lines = [(f"기준 자세 저장 중: 허리를 펴고 화면 정면을 보세요  "
                      f"{self.calib.remaining(now):.1f}초", YELLOW)]
            if self.calib.message:
                lines.append((self.calib.message, RED))
            return lines

        face_issues = detect_face_issues(f, self.base, self.fth)
        face_conf = self.face_filter.update(face_issues, now)
        d = compute_deltas(p, self.base) if p_raw is not None else None
        posture_issues = self.judge_posture(d) if d is not None else []
        posture_conf = self.posture_filter.update(posture_issues, now)

        lines = [status_line("얼굴", face_conf, FACE_LABELS),
                 status_line("자세", posture_conf, LABELS)
                 if p_raw is not None else ("자세: 어깨가 보이지 않습니다", YELLOW)]
        pending = (pending_text(face_issues, self.face_filter, FACE_LABELS, now)
                   + pending_text(posture_issues, self.posture_filter, LABELS, now))
        if pending:
            lines.append(("확인 중: " + ", ".join(pending), YELLOW))
        return lines + value_lines(f, self.base, d) + [self.mode_line()]

    def close(self):
        self.pose.close()
        self.face.close()


def main():
    args = parse_args()
    text = TextRenderer(size=18)
    monitor = Monitor(args)
    src = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise SystemExit(f"카메라를 열 수 없습니다: {args.source}")
    monitor.recalibrate(time.monotonic())

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            now = time.monotonic()
            lines = monitor.step(frame, now)
            lines.append(("c: 기준 자세 다시 저장  |  ESC: 종료", WHITE))
            text.draw_lines(frame, lines)
            cv2.imshow("seat monitor", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break
            if key == ord("c"):
                monitor.recalibrate(now)
    finally:
        cap.release()
        monitor.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()