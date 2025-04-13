import logging
import sys
from typing import NoReturn

import cv2
from cscore import CameraServer, CvSink
from ntcore import DoubleSubscriber, NetworkTable

from color_range import COLOR_RANGE_PARAMS, ColorRangeParam
from color_square_detector import ColorSquareDetector, ImageMask
from config import COLORS, ColorName, load_preferences
from init import NetworkTables, get_server_ip, init_camera_sink, init_network_tables

WIDTH = 160
HEIGHT = 120
FPS = 5


def init() -> CvSink:
    logging.basicConfig(level=logging.DEBUG)
    CameraServer.enableLogging()

    server_ip = get_server_ip()

    init_network_tables(server_ip)
    return init_camera_sink(server_ip, WIDTH, HEIGHT, FPS)


def get_double_subscriber(
    table: NetworkTable,
    name: str,
    default: float,
) -> DoubleSubscriber:
    """初期値を設定しつつsubscriberを作る"""
    table.setDefaultNumber(name, default)
    return table.getDoubleTopic(name).subscribe(default)


def main() -> NoReturn:
    """メインループ"""
    sink = init()

    detector = ColorSquareDetector(HEIGHT, WIDTH, load_preferences())
    input_img = detector.color_detector.get_input_buffer()

    # NetworkTablesのトピックを設定
    table = NetworkTables.getTable("ColorSquares")
    config_table = table.getSubTable("Config")
    color_ranges_table = config_table.getSubTable("ColorRanges")

    # 面積の範囲を取得
    min_area = get_double_subscriber(config_table, "Minimum area", detector.config.min_area)
    max_area = get_double_subscriber(config_table, "Maximum area", detector.config.max_area)

    # 色の範囲を取得

    color_tables: tuple[tuple[ColorName, NetworkTable], ...] = tuple(
        (color, color_ranges_table.getSubTable(color)) for color in COLORS
    )
    params: dict[ColorName, dict[ColorRangeParam, DoubleSubscriber]] = {
        color: {
            param: get_double_subscriber(
                color_table,
                param,
                detector.config.color_ranges[color][param],
            )
            for param in COLOR_RANGE_PARAMS
        }
        for color, color_table in color_tables
    }

    while True:
        grabbed_time, _ = sink.grabFrame(input_img)
        if grabbed_time == 0:
            continue

        # NetworkTablesから取得したパラメータを更新
        detector.config.min_area = min_area.get()
        detector.config.max_area = max_area.get()

        detector.config.color_ranges.items()
        # 色の範囲を更新
        for color_name, color_params in params.items():
            detector.config.color_ranges[color_name] = {
                param: color_params[param].get() for param in COLOR_RANGE_PARAMS
            }

        # 色付き四角形を検出
        colored_squares = detector.detect_colored_squares(input_img)

        # 結果を描画して表示
        result_img = detector.visualizer.draw_colored_squares(input_img, colored_squares)
        cv2.imshow("Colored Squares", result_img)

        # 各色のフィルター後の画像を表示
        color_masks: dict[ColorName, ImageMask] = {
            color_name: detector.color_detector.create_color_mask(input_img, color_range)
            for color_name, color_range in detector.config.color_ranges.items()
        }
        masked_imgs = detector.visualizer.draw_color_masks(input_img, color_masks)
        for color_name, mask_img in masked_imgs.items():
            cv2.imshow(f"{color_name.capitalize()} Mask", mask_img)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    sys.exit(0)


if __name__ == "__main__":
    main()
