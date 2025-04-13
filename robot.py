import logging
import math
from typing import ClassVar, NoReturn, TypedDict

import cv2
import cv2.typing as cv2t
import numpy as np
from cscore import CameraServer, CvSink
from ntcore import DoubleSubscriber, NetworkTable

from init import NetworkTables, get_server_ip, init_camera_sink, init_network_tables


class ColorSquareDetector:
    """赤、青、黄色の四角形を検出するクラス"""

    # 色の定義（HSV色空間）
    COLOR_RANGES: ClassVar = {
        "red": np.array(
            ((0, 100, 100), (10, 255, 255), (160, 100, 100), (180, 255, 255)),
            dtype=np.uint8,
        ),  # 赤は2つの範囲
        "blue": np.array(
            ((100, 100, 100), (130, 255, 255)),
            dtype=np.uint8,
        ),
        "yellow": np.array(
            ((20, 100, 100), (30, 255, 255)),
            dtype=np.uint8,
        ),
    }

    # 色の表示用BGR値
    COLOR_BGR: ClassVar = {"red": (0, 0, 255), "blue": (255, 0, 0), "yellow": (0, 255, 255)}

    def __init__(self, image_height: int, image_width: int) -> None:
        self.img_width = image_width
        self.img_height = image_height

        # メモリ領域を使い回すためのバッファー
        self._hsv_img = np.zeros((image_height, image_width, 3), dtype=np.uint8)
        self._mask_img = np.zeros((image_height, image_width), dtype=np.uint8)
        self._contour_img = np.zeros((image_height, image_width), dtype=np.uint8)

        # 四角形検出のパラメータ
        self.min_area = 100
        self.max_area = image_height * image_width
        self.epsilon_factor = 0.02  # 輪郭近似の精度
        self.max_cosine_limit = 0.3  # 角度の最大コサイン値

        # 色の範囲を動的に変更するための変数
        self._color_ranges = {
            "red": {
                "range1": {
                    "lower": np.array([0, 100, 100], dtype=np.uint8),
                    "upper": np.array([10, 255, 255], dtype=np.uint8),
                },
                "range2": {
                    "lower": np.array([160, 100, 100], dtype=np.uint8),
                    "upper": np.array([180, 255, 255], dtype=np.uint8),
                },
            },
            "blue": {
                "range1": {
                    "lower": np.array([100, 100, 100], dtype=np.uint8),
                    "upper": np.array([130, 255, 255], dtype=np.uint8),
                },
            },
            "yellow": {
                "range1": {
                    "lower": np.array([20, 100, 100], dtype=np.uint8),
                    "upper": np.array([30, 255, 255], dtype=np.uint8),
                },
            },
        }

    def update_parameters(self, min_area: float, max_area: float) -> None:
        """NetworkTablesから取得したパラメータを更新"""
        self.min_area = min_area
        self.max_area = max_area

    def update_color_range(
        self,
        color_name: str,
        range_index: int,
        h_min: int,
        h_max: int,
        s_min: int,
        s_max: int,
        v_min: int,
        v_max: int,
    ) -> None:
        """指定した色のHSV範囲を更新"""
        if color_name not in self._color_ranges:
            return

        range_key = f"range{range_index}"
        if range_key not in self._color_ranges[color_name]:
            return

        self._color_ranges[color_name][range_key]["lower"] = np.array(
            [h_min, s_min, v_min],
            dtype=np.uint8,
        )
        self._color_ranges[color_name][range_key]["upper"] = np.array(
            [h_max, s_max, v_max],
            dtype=np.uint8,
        )

    def get_input_buffer(self) -> cv2.Mat:
        """入力画像用のバッファを取得"""
        return cv2.Mat(np.zeros((self.img_width, self.img_height, 3), dtype=np.uint8))

    @staticmethod
    def _angle(pt1: cv2t.MatLike, pt2: cv2t.MatLike, pt0: cv2t.MatLike) -> float:
        """3点間の角度のコサイン値を計算"""
        dx1 = float(pt1[0, 0] - pt0[0, 0])
        dy1 = float(pt1[0, 1] - pt0[0, 1])
        dx2 = float(pt2[0, 0] - pt0[0, 0])
        dy2 = float(pt2[0, 1] - pt0[0, 1])
        v = math.sqrt((dx1 * dx1 + dy1 * dy1) * (dx2 * dx2 + dy2 * dy2))
        return (dx1 * dx2 + dy1 * dy2) / v

    def _is_square(self, contour: np.ndarray) -> bool:
        """輪郭が四角形かどうかを判定"""
        # 輪郭を近似
        arc_len = cv2.arcLength(contour, closed=True)
        approx = cv2.approxPolyDP(contour, arc_len * self.epsilon_factor, closed=True)

        # 面積を計算
        area = abs(cv2.contourArea(approx))

        # 四角形の条件をチェック
        if (
            approx.shape[0] == 4  # 頂点が4つ
            and self.min_area < area < self.max_area  # 面積が適切
            and cv2.isContourConvex(approx)  # 凸多角形
        ):
            # 角度をチェック
            max_cosine = 0
            for j in range(2, 5):
                cosine = abs(self._angle(approx[j % 4], approx[j - 2], approx[j - 1]))
                max_cosine = max(max_cosine, cosine)

            return max_cosine < self.max_cosine_limit

        return False

    def _create_color_mask(self, img: cv2t.MatLike, color_name: str) -> cv2t.MatLike:
        """指定した色のマスクを作成"""
        # BGRからHSVに変換
        cv2.cvtColor(img, cv2.COLOR_BGR2HSV, self._hsv_img)

        # 色の範囲でマスクを作成
        mask = np.zeros((self.img_height, self.img_width), dtype=np.uint8)

        # 赤色の場合は2つの範囲を結合
        if color_name == "red":
            lower1 = self._color_ranges["red"]["range1"]["lower"]
            upper1 = self._color_ranges["red"]["range1"]["upper"]
            lower2 = self._color_ranges["red"]["range2"]["lower"]
            upper2 = self._color_ranges["red"]["range2"]["upper"]

            mask1 = cv2.inRange(self._hsv_img, lower1, upper1)
            mask2 = cv2.inRange(self._hsv_img, lower2, upper2)
            mask = cv2.bitwise_or(mask1, mask2)
        else:
            lower = self._color_ranges[color_name]["range1"]["lower"]
            upper = self._color_ranges[color_name]["range1"]["upper"]
            mask = cv2.inRange(self._hsv_img, lower, upper)

        # ノイズ除去と穴埋め
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        return cv2.Mat(mask)

    def detect_colored_squares(
        self,
        img: cv2t.MatLike,
    ) -> dict[str, list[tuple[np.ndarray, float]]]:
        """画像から色付き四角形を検出"""
        result = {}

        for color_name in self.COLOR_RANGES:
            # 色のマスクを作成
            mask = self._create_color_mask(img, color_name)

            # 輪郭を検出
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # 四角形を検出
            squares = []
            for contour in contours:
                if self._is_square(contour):
                    # 四角形の頂点を取得
                    arc_len = cv2.arcLength(contour, closed=True)
                    approx = cv2.approxPolyDP(contour, arc_len * self.epsilon_factor, closed=True)
                    rectangle = approx.reshape(-1, 2)

                    # 面積を計算
                    area = abs(cv2.contourArea(approx))

                    squares.append((rectangle, area))

            result[color_name] = squares

        return result

    def draw_colored_squares(
        self,
        img: cv2t.MatLike,
        colored_squares: dict[str, list[tuple[np.ndarray, float]]],
    ) -> cv2t.MatLike:
        """検出した色付き四角形を描画"""
        result_img = img.copy()

        for color_name, squares in colored_squares.items():
            color_bgr = self.COLOR_BGR[color_name]

            for rectangle, area in squares:
                # 矩形を閉じるために最初の点を最後に追加
                pts = np.vstack((rectangle, rectangle[0]))

                # 矩形を描画
                cv2.polylines(result_img, [pts], isClosed=True, color=color_bgr, thickness=2)

                # 矩形の中心点を計算
                center_x = int(np.mean(pts[:, 0]))
                center_y = int(np.mean(pts[:, 1]))

                # 色名と面積を表示
                text = f"{color_name}: {area:.1f}"
                cv2.putText(
                    result_img,
                    text,
                    (center_x, center_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color_bgr,
                    1,
                )

        return result_img

    def draw_color_masks(self, img: cv2t.MatLike) -> dict[str, cv2t.MatLike]:
        """各色のフィルター後の画像を生成"""
        masks = {}
        for color_name in self.COLOR_RANGES:
            # 色のマスクを作成
            mask = self._create_color_mask(img, color_name)
            # マスクを3チャンネルに変換
            mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            # 元の画像とマスクを合成
            result = cv2.bitwise_and(img, mask_3ch)
            masks[color_name] = result
        return masks


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
    sink = init()

    detector = ColorSquareDetector(HEIGHT, WIDTH)
    input_img = detector.get_input_buffer()

    # NetworkTablesのトピックを設定
    table = NetworkTables.getTable("ColorSquares")
    red_squares_topic = table.getDoubleTopic("Red squares").publish()
    blue_squares_topic = table.getDoubleTopic("Blue squares").publish()
    yellow_squares_topic = table.getDoubleTopic("Yellow squares").publish()

    # 設定
    config_table = table.getSubTable("Config")
    min_area = get_double_subscriber(config_table, "Minimum area", 100.0)
    max_area = get_double_subscriber(config_table, "Maximum area", HEIGHT * WIDTH)

    # 色の範囲設定
    color_ranges_table = config_table.getSubTable("ColorRanges")

    # 色の範囲パラメータを格納する辞書
    color_params: dict[str, list[dict[str, DoubleSubscriber]]] = {}

    class ColorRange(TypedDict):
        ranges: int
        defaults: list[dict[str, float]]

    # 色の定義
    colors: dict[str, ColorRange] = {
        "Red": {
            "ranges": 2,
            "defaults": [
                {
                    "H_min": 0.0,
                    "H_max": 10.0,
                    "S_min": 100.0,
                    "S_max": 255.0,
                    "V_min": 100.0,
                    "V_max": 255.0,
                },
                {
                    "H_min": 160.0,
                    "H_max": 180.0,
                    "S_min": 100.0,
                    "S_max": 255.0,
                    "V_min": 100.0,
                    "V_max": 255.0,
                },
            ],
        },
        "Blue": {
            "ranges": 1,
            "defaults": [
                {
                    "H_min": 100.0,
                    "H_max": 130.0,
                    "S_min": 100.0,
                    "S_max": 255.0,
                    "V_min": 100.0,
                    "V_max": 255.0,
                },
            ],
        },
        "Yellow": {
            "ranges": 1,
            "defaults": [
                {
                    "H_min": 20.0,
                    "H_max": 30.0,
                    "S_min": 100.0,
                    "S_max": 255.0,
                    "V_min": 100.0,
                    "V_max": 255.0,
                },
            ],
        },
    }

    # 各色のパラメータをサブスクライブ
    for color_name, color_info in colors.items():
        color_params[color_name] = []
        for range_idx in range(1, color_info["ranges"] + 1):
            range_params: dict[str, DoubleSubscriber] = {}
            range_table = color_ranges_table.getSubTable(color_name).getSubTable(
                f"Range{range_idx}",
            )
            defaults = color_info["defaults"][range_idx - 1]

            for param_name, default_value in defaults.items():
                range_params[param_name] = get_double_subscriber(
                    range_table,
                    param_name,
                    default_value,
                )

            color_params[color_name].append(range_params)

    while True:
        grabbed_time, input_img = CvSink.grabFrame(sink, input_img)
        if grabbed_time == 0:
            continue

        # NetworkTablesから取得したパラメータを更新
        detector.update_parameters(min_area.get(), max_area.get())

        # 色の範囲を更新
        for color_name, ranges in color_params.items():
            color_name_lower = color_name.lower()
            for range_idx, params in enumerate(ranges, 1):
                detector.update_color_range(
                    color_name_lower,
                    range_idx,
                    int(params["H_min"].get()),
                    int(params["H_max"].get()),
                    int(params["S_min"].get()),
                    int(params["S_max"].get()),
                    int(params["V_min"].get()),
                    int(params["V_max"].get()),
                )

        # 色付き四角形を検出
        colored_squares = detector.detect_colored_squares(input_img)

        # 検出結果をNetworkTablesに送信
        red_squares_topic.set(len(colored_squares["red"]))
        blue_squares_topic.set(len(colored_squares["blue"]))
        yellow_squares_topic.set(len(colored_squares["yellow"]))

        # 結果を描画して表示
        result_img = detector.draw_colored_squares(input_img, colored_squares)
        cv2.imshow("Colored Squares", result_img)

        # 各色のフィルター後の画像を表示
        color_masks = detector.draw_color_masks(input_img)
        for color_name, mask_img in color_masks.items():
            cv2.imshow(f"{color_name.capitalize()} Mask", mask_img)

        cv2.waitKey(1)


if __name__ == "__main__":
    main()
