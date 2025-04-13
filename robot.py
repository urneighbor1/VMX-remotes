import logging
import math
from typing import ClassVar, NoReturn

import cv2
import cv2.typing as cv2t
import numpy as np
from cscore import CameraServer, CvSink

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
            lower1, upper1 = self.COLOR_RANGES["red"][0], self.COLOR_RANGES["red"][1]
            lower2, upper2 = self.COLOR_RANGES["red"][2], self.COLOR_RANGES["red"][3]

            mask1 = cv2.inRange(self._hsv_img, lower1, upper1)
            mask2 = cv2.inRange(self._hsv_img, lower2, upper2)
            mask = cv2.bitwise_or(mask1, mask2)
        else:
            lower, upper = self.COLOR_RANGES[color_name]
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


WIDTH = 160
HEIGHT = 120
FPS = 5


def init() -> CvSink:
    logging.basicConfig(level=logging.DEBUG)
    CameraServer.enableLogging()

    server_ip = get_server_ip()

    init_network_tables(server_ip)
    return init_camera_sink(server_ip, WIDTH, HEIGHT, FPS)


def main() -> NoReturn:
    sink = init()

    detector = ColorSquareDetector(HEIGHT, WIDTH)
    input_img = detector.get_input_buffer()

    # NetworkTablesのトピックを設定
    table = NetworkTables.getTable("ColorSquares")
    red_squares_topic = table.getIntegerTopic("Red squares").publish()
    blue_squares_topic = table.getIntegerTopic("Blue squares").publish()
    yellow_squares_topic = table.getIntegerTopic("Yellow squares").publish()

    while True:
        grabbed_time, input_img = CvSink.grabFrame(sink, input_img)
        if grabbed_time == 0:
            continue

        # 色付き四角形を検出
        colored_squares = detector.detect_colored_squares(input_img)

        # 検出結果をNetworkTablesに送信
        red_squares_topic.set(len(colored_squares["red"]))
        blue_squares_topic.set(len(colored_squares["blue"]))
        yellow_squares_topic.set(len(colored_squares["yellow"]))

        # 結果を描画して表示
        result_img = detector.draw_colored_squares(input_img, colored_squares)
        cv2.imshow("Colored Squares", result_img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()
