from __future__ import annotations

from enum import IntEnum


class Command(IntEnum):
    DEFAULT = 0
    LEFT = 1
    STRAIGHT = 2
    RIGHT = 3


COMMAND_NAMES = ["default", "left", "straight", "right"]

COMMAND_COLORS_BGR = {
    Command.DEFAULT: (180, 180, 180),
    Command.LEFT: (255, 80, 0),
    Command.STRAIGHT: (0, 200, 0),
    Command.RIGHT: (0, 80, 255),
}
