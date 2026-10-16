"""영역 캘리브레이션: 화면 → 강의 공간 양 끝 → 책상을 차례로 보게 하고 각 방향(yaw·pitch 중앙값)을 저장

자리마다 강의 공간의 위치(정면·왼쪽·오른쪽)가 달라도 실제로 본 각도를 저장하므로 같은 방식으로 동작한다.
각 영역: 대기(Space를 눌러야 시작) → 2초 시선 옮기기(측정 안 함) → 3초 측정.
안내를 보지 못한 채 다음 영역으로 넘어가지 않도록 영역마다 Space를 기다린다.
얼굴이 잡힌 프레임이 60% 미만이면 그 영역을 다시 대기 상태로 돌린다.
화면 대비 yaw 차이가 45°를 넘으면 옆모습 한계 경고를 남긴다 (얼굴 각도·blendshape가 부정확해짐).
"""
import statistics

ZONES = (("screen", "내 화면을 보세요"),
         ("edge_a", "강의 공간의 한쪽 끝을 보세요"),
         ("edge_b", "강의 공간의 반대쪽 끝을 보세요"),
         ("desk", "책상 위 노트를 보세요"))
ZONE_KO = {"screen": "화면", "edge_a": "경계 A", "edge_b": "경계 B", "desk": "책상"}
MOVE_SEC = 2.0
MIN_VALID = 0.6
SIDE_LIMIT_DEG = 45.0
GREEN, RED, YELLOW, CYAN = (0, 255, 0), (0, 0, 255), (0, 255, 255), (255, 255, 0)


class ZoneCalib:
    def __init__(self, seconds=3.0, lecture=True):
        self.seconds = seconds
        self.order = [z for z in ZONES if lecture or not z[0].startswith("edge")]
        self.zones, self.warnings = {}, []
        self.active, self.waiting, self.message, self.last, self.idx = False, False, "", "", 0

    @property
    def ready(self):
        """모든 영역 측정이 끝났는가"""
        return not self.active and len(self.zones) == len(self.order)

    def start(self, now):
        """처음 영역의 대기부터 다시 시작"""
        self.zones, self.warnings, self.message, self.last, self.idx = {}, [], "", "", 0
        self.active, self.waiting = True, True

    def go(self, now):
        """대기 중 Space: 지금 영역의 시선 옮기기 → 측정 시작"""
        if self.active and self.waiting:
            self.waiting, self.message = False, ""
            self._begin(now)

    def _begin(self, now):
        self.t_start, self.samples, self.frames = now, [], 0

    def _phase(self, now):
        el = now - self.t_start
        if el < MOVE_SEC:
            return "move", MOVE_SEC - el
        return "measure", MOVE_SEC + self.seconds - el

    def add(self, feats, now):
        """feats: compute_face_features 결과 또는 None → 모든 영역이 끝난 프레임에 True"""
        if not self.active or self.waiting:
            return False
        kind, left = self._phase(now)
        if kind == "move":
            return False
        if left > 0:
            self.frames += 1
            if feats is not None:
                self.samples.append((feats["yaw"], feats["pitch"]))
            return False
        return self._finish_zone(now)

    def _finish_zone(self, now):
        name = self.order[self.idx][0]
        if self.frames == 0 or len(self.samples) / self.frames < MIN_VALID:
            self.message = f"{ZONE_KO[name]}: 얼굴이 잘 잡히지 않았습니다. 다시 보고 Space"
            self.waiting = True
            return False
        yaw = statistics.median(s[0] for s in self.samples)
        pitch = statistics.median(s[1] for s in self.samples)
        self.zones[name] = {"yaw": round(yaw, 2), "pitch": round(pitch, 2)}
        self.message = ""
        base = self.zones["screen"]
        self.last = (f"{ZONE_KO[name]} 측정: 화면 대비 좌우 {yaw - base['yaw']:+.0f}° / "
                     f"상하 {pitch - base['pitch']:+.0f}°" if name != "screen"
                     else f"화면 측정: yaw {yaw:+.0f}° / pitch {pitch:+.0f}°")
        if name != "screen":
            diff = yaw - self.zones["screen"]["yaw"]
            if abs(diff) > SIDE_LIMIT_DEG:
                self.warnings.append(f"{ZONE_KO[name]}: 화면 대비 좌우 {diff:+.0f}° "
                                     f"(옆모습 한계 초과, 이 방향 판정이 부정확할 수 있음)")
        self.idx += 1
        if self.idx >= len(self.order):
            self.active = False
            return True
        self.waiting = True
        return False

    def lines(self, now):
        """측정 중 화면 안내 줄"""
        name, how = self.order[self.idx]
        lines = [(f"[영역 {self.idx + 1}/{len(self.order)}] {ZONE_KO[name]}: {how}", CYAN)]
        if self.waiting:
            lines.append(("그 방향을 보면서 Space를 누르세요 (2초 뒤 3초 측정)", YELLOW))
            if self.last:
                lines.append((self.last, GREEN))
            if self.message:
                lines.append((self.message, RED))
            return lines
        kind, left = self._phase(now)
        if kind == "move":
            lines.append((f"시선을 옮기세요  {left:.0f}초 후 측정", YELLOW))
        else:
            lines.append((f"● 측정 중  {max(left, 0):.1f}초 (눈을 뜬 채 그대로 보세요)", RED))
        if self.message:
            lines.append((self.message, RED))
        return lines

    def summary_lines(self):
        """측정이 끝난 뒤 화면 대비 각 영역 각도와 경고"""
        if not self.ready:
            return []
        base = self.zones["screen"]
        parts = [f"{ZONE_KO[k]} {v['yaw'] - base['yaw']:+.0f}°/{v['pitch'] - base['pitch']:+.0f}°"
                 for k, v in self.zones.items() if k != "screen"]
        lines = [("영역 (화면 대비 좌우/상하): " + "  ".join(parts), GREEN)]
        return lines + [(w, RED) for w in self.warnings]