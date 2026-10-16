"""얼굴 학습 데이터 수집 (영역 캘리브레이션 + 안내 녹화: 라벨 + 각도 + blendshape + 얼굴 crop 이미지)

프로젝트 루트(~/dev_ws/mediaPipe)에서 실행:
  python -m collect.face_collect --tag home_day --glasses 1
  python -m collect.face_collect --tag home_day --glasses 0 --no-lecture   # 강의 공간 없음 (경계 A·B, 2단계 생략)
  python -m collect.face_collect --test --glasses 1                        # 테스트 세션 (학습에 사용 안 함)

시작하면 영역 캘리브레이션(화면 → 경계 A → 경계 B → 책상)이 자동으로 진행된다.
한 단계가 끝나면 대기하고 Space를 눌러야 다음 단계로 간다.
키: Space 안내 시작 / 다음 단계 시작 / 일시정지·계속, n 단계 건너뛰기, s 안내 중단, z 영역 다시 측정, ESC 종료
"""
import argparse
import time
from pathlib import Path

import cv2

from collect.face_guide import FACE_KO, FaceGuide, build_face_steps
from collect.face_recorder import FaceRecorder
from collect.zone_calib import ZoneCalib
from common.face_features import FaceDetector, compute_face_features, draw_face_points
from common.korean_text import TextRenderer

GREEN, RED, YELLOW, WHITE = (0, 255, 0), (0, 0, 255), (0, 255, 255), (255, 255, 255)
HELP = "Space 시작·다음·일시정지 | n 건너뛰기 | s 안내 중단 | z 영역 다시 | ESC 종료"


def parse_args():
    root = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser(description="얼굴 학습 데이터 수집")
    ap.add_argument("--source", default="0")
    ap.add_argument("--model", default=str(root / "assets" / "face_landmarker.task"))
    ap.add_argument("--out", default=str(root / "data" / "face"))
    ap.add_argument("--glasses", type=int, choices=(0, 1), required=True, help="안경 착용 1, 미착용 0")
    ap.add_argument("--no-lecture", action="store_true", help="강의 공간 없음: 경계 A·B와 강의 공간 단계 생략")
    ap.add_argument("--img-every", type=int, default=3,
                    help="N프레임마다 얼굴 이미지 저장 (30fps 기준 3 → 초당 10장), 0이면 저장 안 함")
    ap.add_argument("--test", action="store_true", help="테스트 세션으로 저장 (파일 이름에 _test)")
    ap.add_argument("--tag", default="", help="조건 태그 (예: home_day, cafe_night)")
    ap.add_argument("--zone-sec", type=float, default=3.0, help="영역별 측정 시간")
    ap.add_argument("--base-sec", type=float, default=60, help="화면·강의 공간·책상 단계 녹화 시간")
    ap.add_argument("--step-sec", type=float, default=30, help="상태·강도·헷갈리는 동작별 녹화 시간")
    ap.add_argument("--no-shuffle", action="store_true", help="영역 밖·졸음·하품 순서를 섞지 않음")
    return ap.parse_args()


def value_line(f, zones):
    """현재 값 (화면 방향 대비 각도). 값을 맞추라는 뜻은 아님"""
    base = zones.get("screen", {"yaw": 0.0, "pitch": 0.0})
    return (f"눈 감김 {f['eye_closed']:.2f}  입 벌림 {f['jaw_open']:.2f}  시선 {f['gaze']:.2f}  "
            f"화면 대비 좌우 {f['yaw'] - base['yaw']:+.0f}°  상하 {f['pitch'] - base['pitch']:+.0f}°", WHITE)


def status_lines(rec, guide, calib):
    total = sum(n for n, _ in rec.counts.values())
    kind = "테스트" if rec.is_test else "학습"
    glasses = "안경 착용" if rec.glasses else "안경 미착용"
    on = guide.recording
    lines = [(f"{'● 녹화 중' if on else '■ 정지'}  [{kind} · {glasses}]  누적 {total} 프레임  "
              f"이미지 {rec.n_images}장", RED if on else WHITE)]
    if guide.running or guide.waiting or guide.state == "done":
        lines += guide.lines()
    else:
        lines.append((f"Space: 안내 시작 (약 {guide.total_seconds() / 60:.0f}분)", GREEN))
    lines += calib.summary_lines()
    lines.append(("  ".join(f"{FACE_KO[c]} {n}" for c, (n, _) in rec.counts.items()), WHITE))
    return lines


def handle_key(key, guide, calib, now):
    """→ False면 종료"""
    if key == 27:
        return False
    if key == ord(" ") and calib.ready:
        if guide.waiting:                       # 단계 완료 후 대기 → 다음 단계 시작
            guide.advance()
        elif guide.running:
            guide.toggle_pause()
        else:
            guide.start()
    elif key == ord("n") and (guide.running or guide.waiting):
        guide.skip()
    elif key == ord("s"):
        guide.stop()
    elif key == ord("z"):
        guide.stop()
        calib.start(now)
    return True


def main():
    args = parse_args()
    lecture = not args.no_lecture
    text = TextRenderer(size=18)
    detector = FaceDetector(args.model)
    calib = ZoneCalib(seconds=args.zone_sec, lecture=lecture)
    rec = FaceRecorder(args.out, args.img_every, args.test, args.tag, args.glasses)
    guide = FaceGuide(build_face_steps(args.base_sec, args.step_sec, lecture=lecture,
                                       glasses=bool(args.glasses), shuffle=not args.no_shuffle))

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
            clean = frame.copy()                 # 점·글자를 그리기 전 원본 (얼굴 crop 저장용)
            now = time.monotonic()
            dt, t_prev = now - t_prev, now
            res = detector.detect(frame)
            f = compute_face_features(res)
            draw_face_points(frame, res)
            guide.update(dt, active=not calib.active)  # 얼굴을 놓쳐도 안내 시간은 계속

            if calib.active:
                if calib.add(f, now):
                    rec.save_zones(calib.zones, calib.warnings)
                lines = calib.lines(now) if calib.active else calib.summary_lines()
                if f is None:
                    lines.append(("얼굴이 보이지 않습니다", RED))
            elif f is None:
                if guide.recording:                  # 얼굴 없음으로 기록 (뒤돌아보기, 컵으로 가림 등)
                    rec.write(now - t0, guide.step, None, clean)
                lines = status_lines(rec, guide, calib)
                lines.append(("얼굴 미검출 (시간 계속, 얼굴 없음으로 기록)", RED))
            else:
                if guide.recording:
                    rec.write(now - t0, guide.step, res, clean)
                lines = status_lines(rec, guide, calib) + [value_line(f, calib.zones)]
            lines.append((HELP, WHITE))
            text.draw_lines(frame, lines)
            cv2.imshow("face collect", frame)

            if not handle_key(cv2.waitKey(1) & 0xFF, guide, calib, now):
                break
    finally:
        cap.release()
        detector.close()
        rec.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()