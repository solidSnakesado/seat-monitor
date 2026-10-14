"""자세 학습 데이터 수집 (수치 + 강도 + 같은 프레임 이미지)

프로젝트 루트(~/dev_ws/mediaPipe)에서 실행:
  python -m collect.posture_collect --tag near          # 학습 세션
  python -m collect.posture_collect --test              # 테스트 세션 (학습에 사용 안 함)

[안내 모드, 권장] Space 키 → 화면 안내에 따라 자세·강도를 바꾸면 자동으로 녹화된다
    Space 시작 / 일시정지·계속   n 다음 단계로   s 안내 중단
[수동 모드] 1~5 라벨 선택, r 녹화 시작·정지 (강도는 기록 안 됨)
공통: c 기준 자세 다시 저장, ESC 종료
"""
import argparse
import time
from pathlib import Path

import cv2

from collect.collect_guide import Guide, build_steps
from collect.session_recorder import KO, Recorder
from common.korean_text import TextRenderer
from common.posture_features import PoseDetector, Smoother, compute_features, draw_key_points
from common.posture_judge import Calibrator, compute_deltas
from learning.posture_model import CLASSES

KEYS = {ord(str(i + 1)): i for i in range(len(CLASSES))}
GREEN, RED, YELLOW, WHITE = (0, 255, 0), (0, 0, 255), (0, 255, 255), (255, 255, 255)
HELP = "Space 안내 시작·일시정지 | n 다음 | s 안내 중단 | 1~5·r 수동 | c 기준 | ESC"


def parse_args():
    root = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser(description="자세 학습 데이터 수집")
    ap.add_argument("--source", default="0")
    ap.add_argument("--model", default=str(root / "assets" / "pose_landmarker_lite.task"))
    ap.add_argument("--out", default=str(root / "data"))
    ap.add_argument("--calib-sec", type=float, default=5.0)
    ap.add_argument("--img-every", type=int, default=3,
                    help="N프레임마다 이미지 저장 (30fps 기준 3 → 초당 10장), 0이면 저장 안 함")
    ap.add_argument("--test", action="store_true", help="테스트 세션으로 저장 (파일 이름에 _test)")
    ap.add_argument("--tag", default="", help="조건 태그 (예: near, far_night)")
    ap.add_argument("--good-sec", type=float, default=120, help="안내 모드: 바른 자세 녹화 시간")
    ap.add_argument("--step-sec", type=float, default=30, help="안내 모드: 자세·강도별 녹화 시간")
    ap.add_argument("--no-shuffle", action="store_true", help="안내 모드: 자세 순서를 섞지 않음")
    return ap.parse_args()


def value_line(d):
    """강도를 맞출 때 참고하는 현재 변화량 (값을 맞추라는 뜻은 아님)"""
    return (f"참고  귀-어깨 {d['ear_shoulder']:+.0%}  머리 비율 {d['head_ratio']:+.0%}  "
            f"어깨 높이 {d['shoulder_y']:+.3f}  어깨 {d['shoulder_tilt']:+.0f}°  "
            f"귀 {d['ear_tilt']:+.0f}°", WHITE)


def status_lines(rec, guide, label, manual_rec):
    total = sum(rec.counts.values())
    kind = "테스트" if rec.is_test else "학습"
    rec_on = guide.recording or manual_rec
    lines = [(f"{'● 녹화 중' if rec_on else '■ 정지'}  [{kind}]  누적 {total} 프레임  "
              f"이미지 {rec.n_images}장", RED if rec_on else WHITE)]
    if guide.running or guide.state == "done":
        lines += guide.lines(KO)
    else:
        lines += [(f"수동 라벨: {label + 1}. {KO[CLASSES[label]]}  "
                   f"(안내 모드: Space, 약 {guide.total_seconds() / 60:.0f}분)", GREEN)]
    lines.append(("  ".join(f"{KO[c]} {n}" for c, n in rec.counts.items()), WHITE))
    return lines


def handle_key(key, guide, state, calib, smoother, now):
    """→ False면 종료. state: {'label', 'manual_rec', 'base'}"""
    if key == 27:
        return False
    if key == ord(" ") and state["base"] is not None:
        state["manual_rec"] = False
        if guide.running:
            guide.toggle_pause()
        else:
            guide.start()
    elif key == ord("n") and guide.running:
        guide.skip()
    elif key == ord("s"):
        guide.stop()
    elif not guide.running:                    # 수동 모드 키는 안내 중에는 무시
        if key in KEYS:
            state["label"] = KEYS[key]
        elif key == ord("r") and state["base"] is not None:
            state["manual_rec"] = not state["manual_rec"]
    if key == ord("c"):
        guide.stop()
        state["manual_rec"] = False
        smoother.reset()
        calib.start(now)
    return True


def main():
    args = parse_args()
    text = TextRenderer(size=18)
    detector = PoseDetector(args.model)
    smoother = Smoother(alpha=0.3)
    calib = Calibrator(seconds=args.calib_sec)
    rec = Recorder(args.out, args.img_every, args.test, args.tag)
    guide = Guide(build_steps(args.good_sec, args.step_sec, shuffle=not args.no_shuffle))
    state = {"label": 0, "manual_rec": False, "base": None}

    src = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise SystemExit(f"카메라를 열 수 없습니다: {args.source}")
    t0 = t_prev = time.monotonic()
    calib.start(t0)

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            clean = frame.copy()                 # 점·글자를 그리기 전 원본 (이미지 저장용)
            now = time.monotonic()
            dt, t_prev = now - t_prev, now
            h, w = frame.shape[:2]
            lms = detector.detect(frame)
            raw = compute_features(lms, w, h)
            feats = smoother.update(raw)
            draw_key_points(frame, lms)
            guide.update(dt, active=raw is not None and not calib.active)

            if raw is None:
                lines = [("귀·어깨가 화면에 보이지 않습니다 (기록·안내 시간 멈춤)", RED)]
                if calib.active:
                    calib.add(None, now)
            elif calib.active:
                b = calib.add(feats, now)
                if b is not None:
                    state["base"] = b
                lines = [(f"기준 자세 저장 중: 허리를 펴고 정면을 보세요  {calib.remaining(now):.1f}초", YELLOW)]
                if calib.message:
                    lines.append((calib.message, RED))
            else:
                d = compute_deltas(feats, state["base"])
                if guide.recording:
                    rec.write(now - t0, guide.step.label, guide.step.intensity, d, clean)
                elif state["manual_rec"]:
                    rec.write(now - t0, state["label"], "", d, clean)
                lines = status_lines(rec, guide, state["label"], state["manual_rec"]) + [value_line(d)]
            lines.append((HELP, WHITE))
            text.draw_lines(frame, lines)
            cv2.imshow("posture collect", frame)

            if not handle_key(cv2.waitKey(1) & 0xFF, guide, state, calib, smoother, now):
                break
    finally:
        cap.release()
        detector.close()
        rec.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()