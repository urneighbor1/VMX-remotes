import datetime
import logging
import sys
from pathlib import Path
from typing import NoReturn, TypedDict

import cv2
from cscore import CameraServer, CvSink
from ntcore import (
    BooleanSubscriber,
    DoublePublisher,
    DoubleSubscriber,
    NetworkTable,
    StringPublisher,
)

from .color_range import COLOR_RANGE_PARAMS, ColorRangeParam
from .color_square_detector import ColorImage, ColorSquareDetector, ImageMask, Rectangle
from .config import COLORS, ColorName, load_preferences
from .init import NetworkTables, get_server_ip, init_camera_sink, init_network_tables

WIDTH = 160
HEIGHT = 120
FPS = 5

# 画像を保存するディレクトリ
SAVED_IMAGES_DIR = Path("saved_images")


# 型定義
type ColorParams = dict[ColorName, dict[ColorRangeParam, DoubleSubscriber]]
"""色のパラメータの辞書型"""


class SubscribersDict(TypedDict):
    """NetworkTablesの購読者の辞書型"""

    process_camera: BooleanSubscriber
    min_area: DoubleSubscriber
    max_area: DoubleSubscriber
    color_params: ColorParams


class PublishersDict(TypedDict):
    """NetworkTablesの発行者の辞書型"""

    detected_color: StringPublisher
    detected_area_x: DoublePublisher
    detected_area_y: DoublePublisher


class NetworkTablesDict(TypedDict):
    """NetworkTablesのエントリー辞書型"""

    subscribers: SubscribersDict
    publishers: PublishersDict


def init() -> CvSink:
    # 画像を保存するディレクトリが存在しない場合は作成
    SAVED_IMAGES_DIR.mkdir(exist_ok=True)

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


def save_image(image: ColorImage) -> None:
    """画像を保存する"""
    timestamp = datetime.datetime.now(datetime.UTC).strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp}.jpg"
    filepath = SAVED_IMAGES_DIR.joinpath(filename)
    cv2.imwrite(filepath.as_posix(), image)
    logger = logging.getLogger(__name__)
    logger.info(f"画像を保存しました: {filepath}")  # noqa: G004


def initialize_system() -> tuple[CvSink, ColorSquareDetector]:
    """システムを初期化し、カメラシンクと検出器を返す"""
    sink = init()
    detector = ColorSquareDetector(HEIGHT, WIDTH, load_preferences())
    return sink, detector


def setup_networktables(detector: ColorSquareDetector) -> NetworkTablesDict:
    """NetworkTablesのテーブルを設定し、購読者と発行者を返す"""
    # NetworkTablesのトピックを設定
    table = NetworkTables.getTable("ColorSquares")
    config_table = table.getSubTable("Config")
    color_ranges_table = config_table.getSubTable("ColorRanges")

    # 入力エントリー
    process_camera_sub = table.getBooleanTopic("processCamera").subscribe(defaultValue=False)
    table.setDefaultBoolean("processCamera", defaultValue=False)

    # 出力エントリー
    detected_color_pub = table.getStringTopic("DetectedColor").publish()
    detected_area_x_pub = table.getDoubleTopic("DetectedAreaX").publish()
    detected_area_y_pub = table.getDoubleTopic("DetectedAreaY").publish()
    detected_color_pub.set("NONE")
    detected_area_x_pub.set(-1.0)
    detected_area_y_pub.set(-1.0)

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

    # 購読者と発行者をまとめて返す
    return NetworkTablesDict(
        subscribers=SubscribersDict(
            process_camera=process_camera_sub,
            min_area=min_area,
            max_area=max_area,
            color_params=params,
        ),
        publishers=PublishersDict(
            detected_color=detected_color_pub,
            detected_area_x=detected_area_x_pub,
            detected_area_y=detected_area_y_pub,
        ),
    )


def update_config_from_networktables(
    detector: ColorSquareDetector, nt_subscribers: SubscribersDict
) -> None:
    """NetworkTablesから設定値を取得し、detectorの設定を更新する"""
    # 面積の範囲を更新
    detector.config.min_area = nt_subscribers["min_area"].get()
    detector.config.max_area = nt_subscribers["max_area"].get()

    # 色の範囲を更新
    color_params = nt_subscribers["color_params"]
    detector.config.color_ranges = {
        color_name: {param: color_params_dict[param].get() for param in COLOR_RANGE_PARAMS}
        for color_name, color_params_dict in color_params.items()
    }


def select_best_square(
    colored_squares: dict[ColorName, list[tuple[Rectangle, float]]],
) -> tuple[ColorName, Rectangle, float] | None:
    """検出された四角形から最適なものを選択する"""
    # 全ての色の四角形を一つのリストにまとめる
    all_squares: list[tuple[ColorName, Rectangle, float]] = [
        (color, rect, area)
        for color, squares_with_area in colored_squares.items()
        for rect, area in squares_with_area
    ]

    if not all_squares:
        return None

    # 最も左にある四角形を選択
    def get_center_x(item: tuple[ColorName, Rectangle, float]) -> float:
        rect = item[1]
        return rect[:, 0].mean()

    return min(all_squares, key=get_center_x)


def publish_results(
    best_result: tuple[ColorName, Rectangle, float] | None,
    nt_publishers: PublishersDict,
) -> None:
    """検出結果をNetworkTablesに発行する"""
    if not best_result:
        # 検出されなかった場合
        nt_publishers["detected_color"].set("NONE")
        nt_publishers["detected_area_x"].set(-1.0)
        nt_publishers["detected_area_y"].set(-1.0)
    else:
        # 検出された場合
        best_color, best_rect, _ = best_result  # areaは無視
        # 4つの点のx座標とy座標それぞれの平均を計算
        center_x: float = best_rect[:, 0].mean()
        center_y: float = best_rect[:, 1].mean()

        nt_publishers["detected_color"].set(best_color.upper())  # "RED" or "BLUE" or "YELLOW"
        nt_publishers["detected_area_x"].set(center_x)
        nt_publishers["detected_area_y"].set(center_y)


def display_results(
    input_img: ColorImage,
    detector: ColorSquareDetector,
    colored_squares: dict[ColorName, list[tuple[Rectangle, float]]],
) -> None:
    """検出結果を画像に描画して表示する"""
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


def handle_keyboard_input(input_img: ColorImage) -> bool:
    """キーボード入力を処理し、続行するかどうかを返す"""
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        return False
    if key == ord("s"):
        # 元の画像を保存
        save_image(input_img)
    return True


def process_single_frame(
    sink: CvSink,
    detector: ColorSquareDetector,
    nt_entries: NetworkTablesDict,
) -> bool:
    """1フレームの処理を行う"""
    # process_cameraが再度有効になった時に、"準備中"状態にする
    # subscriberに出力のキャッシュ機能があるが、ロボット側で上書きされた時に、
    # 変化が起こるまでそれを上書き出来ない
    process_camera_queue = nt_entries["subscribers"]["process_camera"].readQueue()
    if process_camera_queue and process_camera_queue[-1].value:
        nt_entries["publishers"]["detected_color"].set("PREPARING")
    # processCameraフラグを確認
    if not nt_entries["subscribers"]["process_camera"].get():
        # Falseなら待機
        cv2.waitKey(100)  # CPU負荷軽減のため少し待機
        return True

    input_img = detector.color_detector.get_input_buffer()
    grabbed_time, _ = sink.grabFrame(input_img)
    if grabbed_time == 0:
        return True

    # NetworkTablesから設定値を更新
    update_config_from_networktables(detector, nt_entries["subscribers"])

    # 色付き四角形を検出
    colored_squares = detector.detect_colored_squares(input_img)

    # 最適な四角形を選択
    best_square = select_best_square(colored_squares)

    # 結果をNetworkTablesに発行
    publish_results(best_square, nt_entries["publishers"])

    # 結果を画像に描画して表示
    display_results(input_img, detector, colored_squares)

    # キーボード入力を処理
    return handle_keyboard_input(input_img)


def run_main_loop(
    sink: CvSink,
    detector: ColorSquareDetector,
    nt_entries: NetworkTablesDict,
) -> None:
    """メインループを実行する"""
    while True:
        # 1フレームの処理を行い、結果に応じて継続するかどうかを決定
        if not process_single_frame(sink, detector, nt_entries):
            break


def main() -> NoReturn:
    """メインループ"""
    sink, detector = initialize_system()
    nt_entries = setup_networktables(detector)
    run_main_loop(sink, detector, nt_entries)
    sys.exit(0)


if __name__ == "__main__":
    main()
