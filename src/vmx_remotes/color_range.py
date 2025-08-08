from typing import Literal

type ColorRangeParam = Literal["H_min", "H_max", "S_min", "S_max", "V_min", "V_max"]
COLOR_RANGE_PARAMS: tuple[ColorRangeParam, ...] = (
    "H_min",
    "H_max",
    "S_min",
    "S_max",
    "V_min",
    "V_max",
)


type ColorRange = dict[ColorRangeParam, float]
"""HSV色空間での色の範囲"""
