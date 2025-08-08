"""NetworkTables上のカメラ設定を保存するスクリプト"""

import json
import time

from config import PREFERENCES_FILE, ColorConfig
from init import NetworkTables, get_server_ip, init_network_tables


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
        # 滅多に起こらない
        msg = "NetworkTablesサーバーに接続できませんでした"
        raise ConnectionError(msg)

    # 設定を取得
    table = NetworkTables.getTable("ColorSquares")

    if "Config" not in table.getSubTables():
        print("NetworkTablesに設定が見つかりませんでした。")
        return

    config_table = table.getSubTable("Config")
    color_ranges_table = config_table.getSubTable("ColorRanges")

    # 保存する設定
    config = ColorConfig(
        min_area=config_table.getNumber("Minimum area", 0.0),
        max_area=config_table.getNumber("Maximum area", 0.0),
        epsilon_factor=config_table.getNumber("Epsilon factor", 0.04),
        max_cosine_limit=config_table.getNumber("Max cosine limit", 0.3),
        # 各色の設定を取得
        color_ranges={
            color: {
                "H_min": color_table.getNumber("H_min", 0.0),
                "H_max": color_table.getNumber("H_max", 0.0),
                "S_min": color_table.getNumber("S_min", 0.0),
                "S_max": color_table.getNumber("S_max", 0.0),
                "V_min": color_table.getNumber("V_min", 0.0),
                "V_max": color_table.getNumber("V_max", 0.0),
            }
            for color, color_table in (  # type: ignore
                (color, color_ranges_table.getSubTable(color))
                for color in ("Red", "Blue", "Yellow")
            )
        },
    )

    # JSONファイルに保存
    with PREFERENCES_FILE.open("w") as f:
        json.dump(config.model_dump(), f, indent=4)

    print(f"設定を {PREFERENCES_FILE.as_posix()} に保存しました。")


if __name__ == "__main__":
    save_preferences()
