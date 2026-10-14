"""OpenCV 화면에 한글 출력

cv2.putText는 한글을 그리지 못하므로 Pillow로 그린 뒤 프레임에 다시 넣는다.
한글 폰트가 없으면: sudo apt install fonts-nanum
"""
import subprocess
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
    "/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/unfonts-core/UnDotum.ttf",
]


def find_korean_font():
    for path in FONT_CANDIDATES:
        if Path(path).exists():
            return path
    try:   # 목록에 없으면 시스템에 설치된 한글 폰트 검색
        out = subprocess.run(["fc-list", ":lang=ko", "file"],
                             capture_output=True, text=True, timeout=3).stdout
        for line in out.splitlines():
            path = line.split(":")[0].strip()
            if path:
                return path
    except (OSError, subprocess.SubprocessError):
        pass
    raise FileNotFoundError("한글 폰트가 없습니다. 설치: sudo apt install fonts-nanum")


class TextRenderer:
    def __init__(self, size=18, font_path=None):
        self.font = ImageFont.truetype(font_path or find_korean_font(), size)
        self.line_h = int(size * 1.4)

    def draw_lines(self, frame, lines, x=10, y0=8):
        """lines: [(문자열, BGR 색)] — 검은 테두리를 둘러 배경과 무관하게 읽히게 한다"""
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img)
        for i, (text, (b, g, r)) in enumerate(lines):
            draw.text((x, y0 + i * self.line_h), text, font=self.font,
                      fill=(r, g, b), stroke_width=2, stroke_fill=(0, 0, 0))
        frame[:] = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)