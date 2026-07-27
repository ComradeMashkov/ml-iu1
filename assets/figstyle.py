"""Единый стиль графиков курса (академический, в духе ETH/Beamer-metropolis).

Использование в начале лекции (скрытая ячейка):

    #| echo: false
    import sys; sys.path.append("../assets")
    from figstyle import set_style, INK, ACCENT, ACCENT2, GREEN, AMBER, MUTED
    set_style()
"""
import matplotlib as mpl

# --- палитра курса ---
INK     = "#20232A"   # текст и оси
ACCENT  = "#215CAF"   # основной акцент (ETH-синий)
ACCENT2 = "#B7352D"   # ошибка/опасность
GREEN   = "#2C7D59"
AMBER   = "#C28A00"
MUTED   = "#9AA0A6"
GRID    = "#E7E9EC"


def set_style():
    mpl.rcParams.update({
        # фон
        "figure.facecolor": "white",
        "axes.facecolor":   "white",
        "savefig.facecolor": "white",
        # шрифты
        "font.family":      "sans-serif",
        "font.sans-serif":  ["Fira Sans", "Helvetica Neue", "Arial", "DejaVu Sans"],
        "font.size":        15,
        "axes.titlesize":   16,
        "axes.titleweight": "medium",
        "axes.labelsize":   14,
        "legend.fontsize":  13,
        # оси: тонкие, без верх/правой рамки (metropolis-вид)
        "axes.edgecolor":   "#3A3F45",
        "axes.linewidth":   1.0,
        "axes.spines.top":  False,
        "axes.spines.right": False,
        # сетка
        "axes.grid":        True,
        "axes.axisbelow":   True,
        "grid.color":       GRID,
        "grid.linewidth":   0.9,
        # цвета элементов
        "text.color":       INK,
        "axes.labelcolor":  INK,
        "xtick.color":      INK,
        "ytick.color":      INK,
        "xtick.labelsize":  12,
        "ytick.labelsize":  12,
        # линии и порядок цветов
        "lines.linewidth":  2.2,
        "axes.prop_cycle":  mpl.cycler(color=[ACCENT, ACCENT2, GREEN, AMBER, MUTED]),
        # легенда
        "legend.frameon":   False,
        # рендер
        "figure.dpi":       150,
        "figure.constrained_layout.use": True,
    })
