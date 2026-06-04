import platform

import matplotlib.pyplot as plt
from matplotlib import font_manager


def configure_korean_font() -> None:
    preferred_fonts = {
        "Windows": ["Malgun Gothic"],
        "Darwin": ["AppleGothic"],
        "Linux": ["NanumGothic", "NanumBarunGothic", "Noto Sans CJK KR", "Noto Sans KR"],
    }
    installed_fonts = {font.name for font in font_manager.fontManager.ttflist}
    candidates = preferred_fonts.get(platform.system(), []) + [
        "NanumGothic",
        "Noto Sans CJK KR",
        "Noto Sans KR",
    ]

    for font_name in candidates:
        if font_name in installed_fonts:
            plt.rcParams["font.family"] = font_name
            break


configure_korean_font()
plt.rcParams["axes.unicode_minus"] = False


COLORS = {
    "primary": "#2563eb",
    "secondary": "#16a34a",
    "danger": "#dc2626",
    "muted": "#64748b",
    "grid": "#e2e8f0",
}


def style_axis(ax, title: str | None = None) -> None:
    if title:
        ax.set_title(title)
    ax.grid(True, color=COLORS["grid"], linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def new_figure(figsize: tuple[int, int] = (10, 5)):
    fig, ax = plt.subplots(figsize=figsize)
    return fig, ax
