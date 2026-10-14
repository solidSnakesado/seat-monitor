"""거북목·자세 체크 실행

  python posture_main.py
  python posture_main.py --source http://<ESP32-IP>:81/stream --hold-sec 30

키: c 재캘리브레이션, ESC 종료
화면: 1~3단계 특징값 → 5단계 변화량 → 6단계 후보 → 7단계 확정 판정
한글 표시에 Pillow와 한글 폰트 필요: pip install pillow / sudo apt install fonts-nanum
"""
import argparse
import time
from pathlib import Path

import cv2

from korean_text import TextRenderer
from posture_features import PoseDetector, Smoother, compute_features, draw_key_points
from posture_judge import LABELS, Calibrator, DurationFilter, Thresholds, compute_deltas, detect_issues

GREEN, RED, YELLOW, WHITE = (0, 255, 0), (0, 0, 255), (0, 255, 255), (255, 255, 255)


def parse_args():
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description="거북목·자세 체크")
    ap.add_argument("--source", default="0", help="웹캠 번호 또는 스트림 URL")
    ap.add_argument("--model", default=str(here / "pose_landmarker_lite.task"))
    ap.add_argument("--calib-sec", type=float, default=5.0)
    ap.add_argument("--hold-sec", type=float, default=Thresholds.hold_sec)
    return ap.parse_args()


def feature_lines(f):
    return [
        (f"특징  귀-어깨 {f['ear_shoulder']:.3f}  머리 비율 {f['head_ratio']:.3f}  "
         f"어깨 높이 {f['shoulder_y']:.3f}", WHITE),
        (f"      어깨 기울기 {f['shoulder_tilt']:+.1f}°  귀 기울기 {f['ear_tilt']:+.1f}°", WHITE),
    ]


def delta_lines(d):
    return [
        (f"변화  귀-어깨 {d['ear_shoulder']:+.0%}  머리 비율 {d['head_ratio']:+.0%}  "
         f"어깨 높이 {d['shoulder_y']:+.3f}", WHITE),
        (f"      어깨 기울기 {d['shoulder_tilt']:+.1f}°  귀 기울기 {d['ear_tilt']:+.1f}°", WHITE),
    ]


def result_lines(confirmed, candidates, durations, now):
    if confirmed:
        lines = [("자세: " + ", ".join(LABELS[k] for k in confirmed), RED)]
    else:
        lines = [("자세: 바른 자세", GREEN)]
    pending = [k for k in candidates if k not in confirmed]
    if pending:
        text = ", ".join(f"{LABELS[k]} {durations.elapsed(k, now):.1f}초" for k in pending)
        lines.append(("확인 중: " + text, YELLOW))
    return lines


def main():
    args = parse_args()
    th = Thresholds(hold_sec=args.hold_sec)
    text = TextRenderer(size=18)
    detector = PoseDetector(args.model)
    smoother = Smoother(alpha=0.3)
    calib = Calibrator(seconds=args.calib_sec)
    durations = DurationFilter(th.hold_sec, th.release_sec)
    base = None

    src = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise SystemExit(f"카메라를 열 수 없습니다: {args.source}")
    calib.start(time.monotonic())

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            now = time.monotonic()
            h, w = frame.shape[:2]

            lms = detector.detect(frame)                       # 2단계
            raw = compute_features(lms, w, h)                  # 3단계
            feats = smoother.update(raw)                       # 프레임 간 흔들림 완화
            draw_key_points(frame, lms)

            if raw is None:
                lines = [("사람이 없거나 귀·어깨가 화면에 보이지 않습니다", RED)]
                durations.reset()
                if calib.active:
                    calib.add(None, now)
            elif calib.active:                                 # 4단계
                b = calib.add(feats, now)
                if b is not None:
                    base = b
                    durations.reset()
                lines = [(f"기준 자세 저장 중: 허리를 펴고 정면을 보세요  {calib.remaining(now):.1f}초",
                          YELLOW), *feature_lines(feats)]
                if calib.message:
                    lines.append((calib.message, RED))
            else:
                d = compute_deltas(feats, base)                # 5단계
                candidates = detect_issues(d, th)              # 6단계
                confirmed = durations.update(candidates, now)  # 7단계
                lines = result_lines(confirmed, candidates, durations, now)
                lines += feature_lines(feats) + delta_lines(d)

            lines.append(("c: 기준 자세 다시 저장  |  ESC: 종료", WHITE))
            text.draw_lines(frame, lines)
            cv2.imshow("posture check", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break
            if key == ord("c"):
                smoother.reset()
                durations.reset()
                calib.start(now)
    finally:
        cap.release()
        detector.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()