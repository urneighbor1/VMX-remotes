import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from color_range import ColorRange

PREFERENCES_FILE = Path(__file__).parent / "color_squares_preferences.json"
"設定ファイルのパス"


type ColorName = Literal["Red", "Blue", "Yellow"]
COLORS: tuple[ColorName, ...] = ("Red", "Blue", "Yellow")


class ColorConfig(BaseModel):
    """色検出の設定"""

    min_area: float = Field(gt=0)
    max_area: float = Field(gt=0)
    epsilon_factor: float = Field(gt=0, lt=1)  # 輪郭近似の精度
    max_cosine_limit: float = Field(gt=0, lt=1)  # 角度の最大コサイン値
    color_ranges: dict[ColorName, ColorRange]


def load_preferences() -> ColorConfig:
    """保存した設定を読み込む"""
    try:
        with PREFERENCES_FILE.open() as f:
            data = json.load(f)
        return ColorConfig(**data)
    except FileNotFoundError:
        # デフォルト設定を返す
        return ColorConfig(
            min_area=100.0,
            max_area=10000.0,
            epsilon_factor=0.04,
            max_cosine_limit=0.3,
            color_ranges={
                "Red": {
                    "H_min": 0,
                    "H_max": 10,
                    "S_min": 100,
                    "S_max": 255,
                    "V_min": 100,
                    "V_max": 255,
                },
                "Blue": {
                    "H_min": 100,
                    "H_max": 130,
                    "S_min": 100,
                    "S_max": 255,
                    "V_min": 100,
                    "V_max": 255,
                },
                "Yellow": {
                    "H_min": 20,
                    "H_max": 30,
                    "S_min": 100,
                    "S_max": 255,
                    "V_min": 100,
                    "V_max": 255,
                },
            },
        )


if __name__ == "__main__":
    if not PREFERENCES_FILE.exists():
        PREFERENCES_FILE.write_text(load_preferences().model_dump_json(indent=4))
