"""ANSI SGR → HTML 转换器 (零依赖)

支持:
  - 16 色 (30-37, 40-47, 90-97, 100-107)
  - 256 色 (38;5;n / 48;5;n)
  - 24-bit truecolor (38;2;r;g;b / 48;2;r;g;b)
  - bold / dim / italic / underline / strikethrough
  - 跨行状态追踪 (Rich 进度条等场景)
"""

import html
import re
from typing import Optional

# 匹配所有 ANSI 转义序列 (CSI + OSC + 裸 \r)
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]|\x1b\][^\x07]*\x07|\r")

# 标准 16 色调色板 (0-7 普通, 8-15 高亮)
_BASE_COLORS = [
    "#000000", "#cd0000", "#00cd00", "#cdcd00",
    "#0000ee", "#cd00cd", "#00cdcd", "#e5e5e5",
    "#7f7f7f", "#ff0000", "#00ff00", "#ffff00",
    "#5c5cff", "#ff00ff", "#00ffff", "#ffffff",
]


def _color_256(n: int) -> Optional[str]:
    """将 256 色索引转为 #rrggbb"""
    if n < 0 or n > 255:
        return None
    if n < 16:
        return _BASE_COLORS[n]
    if n < 232:
        # 6x6x6 色块
        n -= 16
        b = n % 6
        n //= 6
        g = n % 6
        r = n // 6
        return f"#{r * 51:02x}{g * 51:02x}{b * 51:02x}"
    # 灰阶 232-255
    v = 8 + (n - 232) * 10
    return f"#{v:02x}{v:02x}{v:02x}"


def strip_ansi(text: str) -> str:
    """剥离所有 ANSI 转义码，返回纯文本"""
    return _ANSI_RE.sub("", text)


class AnsiToHtmlConverter:
    """有状态的 ANSI → HTML 转换器，跨行追踪样式"""

    def __init__(self):
        self.bold = False
        self.dim = False
        self.italic = False
        self.underline = False
        self.strikethrough = False
        self.fg: Optional[str] = None  # #rrggbb or None
        self.bg: Optional[str] = None

    def reset(self):
        """重置所有状态"""
        self.__init__()

    def _apply_sgr(self, params: list[int]):
        """解析 SGR 参数列表并更新内部状态"""
        i = 0
        while i < len(params):
            p = params[i]
            if p == 0:
                self.reset()
            elif p == 1:
                self.bold = True
            elif p == 2:
                self.dim = True
            elif p == 3:
                self.italic = True
            elif p == 4:
                self.underline = True
            elif p == 9:
                self.strikethrough = True
            elif p == 22:
                self.bold = False
                self.dim = False
            elif p == 23:
                self.italic = False
            elif p == 24:
                self.underline = False
            elif p == 29:
                self.strikethrough = False
            elif 30 <= p <= 37:
                self.fg = _BASE_COLORS[p - 30]
            elif 40 <= p <= 47:
                self.bg = _BASE_COLORS[p - 40]
            elif 90 <= p <= 97:
                self.fg = _BASE_COLORS[p - 90 + 8]
            elif 100 <= p <= 107:
                self.bg = _BASE_COLORS[p - 100 + 8]
            elif p == 39:
                self.fg = None
            elif p == 49:
                self.bg = None
            elif p == 38:
                # 前景: 256 色 / truecolor
                if i + 1 < len(params):
                    mode = params[i + 1]
                    if mode == 5 and i + 2 < len(params):
                        self.fg = _color_256(params[i + 2])
                        i += 2
                    elif mode == 2 and i + 4 < len(params):
                        r, g, b = params[i + 2], params[i + 3], params[i + 4]
                        self.fg = f"#{r:02x}{g:02x}{b:02x}"
                        i += 4
                    else:
                        i += 1
            elif p == 48:
                # 背景: 256 色 / truecolor
                if i + 1 < len(params):
                    mode = params[i + 1]
                    if mode == 5 and i + 2 < len(params):
                        self.bg = _color_256(params[i + 2])
                        i += 2
                    elif mode == 2 and i + 4 < len(params):
                        r, g, b = params[i + 2], params[i + 3], params[i + 4]
                        self.bg = f"#{r:02x}{g:02x}{b:02x}"
                        i += 4
                    else:
                        i += 1
            i += 1

    def _build_style(self) -> str:
        """根据当前状态生成 CSS style 字符串

        所有属性使用 !important 以覆盖 Quasar/主题的全局 CSS 规则。
        """
        parts = []
        if self.fg:
            parts.append(f"color:{self.fg} !important")
        if self.bg:
            parts.append(f"background-color:{self.bg} !important")
        if self.bold:
            parts.append("font-weight:bold !important")
        if self.dim:
            parts.append("opacity:0.6")
        if self.italic:
            parts.append("font-style:italic")
        decorations = []
        if self.underline:
            decorations.append("underline")
        if self.strikethrough:
            decorations.append("line-through")
        if decorations:
            parts.append(f"text-decoration:{' '.join(decorations)}")
        return ";".join(parts)

    def _has_style(self) -> bool:
        return bool(self.fg or self.bg or self.bold or self.dim
                     or self.italic or self.underline or self.strikethrough)

    def convert_line(self, line: str) -> str:
        """将单行 ANSI 文本转为 HTML（先 escape 文本再包 span）"""
        # 用正则分割出 ANSI 序列和纯文本
        sgr_re = re.compile(r"\x1b\[([0-9;]*)m")
        result = []
        last_end = 0

        for m in sgr_re.finditer(line):
            # 序列之前的纯文本
            text_before = line[last_end:m.start()]
            if text_before:
                # 剥离非 SGR 的其他 ANSI 序列
                text_before = _ANSI_RE.sub("", text_before)
                if text_before:
                    escaped = html.escape(text_before)
                    style = self._build_style()
                    if style:
                        result.append(f'<span style="{style}">{escaped}</span>')
                    else:
                        result.append(escaped)

            # 解析 SGR 参数
            raw_params = m.group(1)
            if raw_params:
                params = [int(x) for x in raw_params.split(";") if x.isdigit()]
            else:
                params = [0]
            self._apply_sgr(params)

            last_end = m.end()

        # 剩余文本
        remaining = line[last_end:]
        if remaining:
            remaining = _ANSI_RE.sub("", remaining)
            if remaining:
                escaped = html.escape(remaining)
                style = self._build_style()
                if style:
                    result.append(f'<span style="{style}">{escaped}</span>')
                else:
                    result.append(escaped)

        return "".join(result)

    def convert(self, text: str) -> str:
        """转换多行文本，行间用 <br> 分隔"""
        lines = text.split("\n")
        html_lines = [self.convert_line(line) for line in lines]
        return "<br>".join(html_lines)
