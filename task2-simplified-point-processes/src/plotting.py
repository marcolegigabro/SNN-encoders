"""Shared chart tokens and axis styling for every figure of this task.

Dataviz reference palette, light surface. Categorical slots are assigned in this
fixed order and never cycled; slots 1-3 pass the all-pairs CVD and normal-vision
checks of validate_palette.js. Slot 3 (aqua) is below 3:1 contrast on the
surface, so a series drawn in it always gets a direct label.
"""

SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
SLOTS = ("#2a78d6", "#eb6834", "#1baf7a")


def style(ax):
    """Recessive chrome: hairline solid grid, no top/right spines, muted ticks."""
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
        ax.spines[side].set_linewidth(1)
    ax.grid(True, color=GRID, linewidth=0.8, which="major")
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, labelcolor=INK2, labelsize=9)
