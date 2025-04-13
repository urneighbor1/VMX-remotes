"""NetworkTables上のカメラ設定を保存するスクリプト"""

import json
import time
from pathlib import Path
from typing import TypedDict

from init import NetworkTables, get_server_ip, init_network_tables

# 設定ファイルのパス
PREFERENCES_FILE = "color_squares_preferences.json"


class Preferences(TypedDict):
    min_area: float
    max_area: float
    color_ranges: dict[str, dict[str, float]]


def save_preferences() -> None:
    """NetworkTablesの設定をJSONファイルに保存する"""
    # NetworkTablesに接続
    init_network_tables(get_server_ip())

    # 接続できるまで待つ
    for _ in range(5):
        time.sleep(1)
        if NetworkTables.isConnected():
            break
    else:
        # 接続出来なかった(滅多にない)
        msg = "NetworkTablesサーバーに接続できませんでした"
        raise ConnectionError(msg)

    # 設定を取得
    table = NetworkTables.getTable("ColorSquares")

    if "Config" not in table.getSubTables():
        # まだConfigが作成されていない
        print("NetworkTablesに設定が見つかりませんでした。")
        return

    config_table = table.getSubTable("Config")
    color_ranges_table = config_table.getSubTable("ColorRanges")

    # 保存する設定
    preferences: Preferences = {
        "min_area": config_table.getNumber("Minimum area", 0.0),
        "max_area": config_table.getNumber("Maximum area", 0.0),
        "color_ranges": {},
    }

    # 各色の設定を取得
    for color in ["Red", "Blue", "Yellow"]:
        color_table = color_ranges_table.getSubTable(color)
        preferences["color_ranges"][color] = {
            "H_min": color_table.getNumber("H_min", 0.0),
            "H_max": color_table.getNumber("H_max", 0.0),
            "S_min": color_table.getNumber("S_min", 0.0),
            "S_max": color_table.getNumber("S_max", 0.0),
            "V_min": color_table.getNumber("V_min", 0.0),
            "V_max": color_table.getNumber("V_max", 0.0),
        }

    # JSONファイルに保存
    with Path(PREFERENCES_FILE).open("w") as f:
        json.dump(preferences, f, indent=4)

    print(f"設定を {PREFERENCES_FILE} に保存しました。")


if __name__ == "__main__":
    save_preferences()
