import math
from typing import ClassVar, Literal

import cv2
import cv2.typing as cv2t
import numpy as np

from color_range import ColorRange
from config import ColorConfig, ColorName

type ColorImage = np.ndarray[tuple[int, int, Literal[3]], np.dtype[np.uint8]]
type ImageMask = np.ndarray[tuple[int, int], np.dtype[np.uint8]]


class ColorDetector:
    """色の検出を行う"""

    def __init__(self, image_height: int, image_width: int) -> None:
        self.img_width = image_width
        self.img_height = image_height

        self._hsv_img = np.zeros((image_height, image_width, 3), dtype=np.uint8)
        self._kernel = np.ones((5, 5), np.uint8)

    def get_input_buffer(self) -> ColorImage:
        """入力画像用のバッファを取得"""
        return np.zeros((self.img_height, self.img_width, 3), dtype=np.uint8)

    def create_color_mask(self, img: ColorImage, color_range: ColorRange) -> ImageMask:
        cv2.cvtColor(img, cv2.COLOR_BGR2HSV, self._hsv_img)
        """指定した色のマスクを作成"""
        lower = np.array(
            [color_range["H_min"], color_range["S_min"], color_range["V_min"]],
            dtype=np.uint8,
        )
        upper = np.array(
            [color_range["H_max"], color_range["S_max"], color_range["V_max"]],
            dtype=np.uint8,
        )
        mask = cv2.inRange(self._hsv_img, lower, upper)

        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self._kernel)
        return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self._kernel)  # type: ignore


class SquareDetector:
    """四角形の検出を行う"""

    def __init__(self, config: ColorConfig) -> None:
        self.config = config

    @staticmethod
    def _angle(pt1: cv2t.MatLike, pt2: cv2t.MatLike, pt0: cv2t.MatLike) -> float:
        """3点間の角度のコサイン値を計算"""
        dx1 = float(pt1[0, 0] - pt0[0, 0])
        dy1 = float(pt1[0, 1] - pt0[0, 1])
        dx2 = float(pt2[0, 0] - pt0[0, 0])
        dy2 = float(pt2[0, 1] - pt0[0, 1])
        v = math.sqrt((dx1 * dx1 + dy1 * dy1) * (dx2 * dx2 + dy2 * dy2))
        return (dx1 * dx2 + dy1 * dy2) / v

    def is_square(self, contour: cv2t.MatLike) -> bool:
        """輪郭が四角形かどうかを判定"""
        arc_len = cv2.arcLength(contour, closed=True)
        approx = cv2.approxPolyDP(contour, arc_len * self.config.epsilon_factor, closed=True)
        area = abs(cv2.contourArea(approx))

        if (
            approx.shape[0] == 4  # noqa: PLR2004
            and self.config.min_area < area < self.config.max_area
            and cv2.isContourConvex(approx)
        ):
            max_cosine = 0
            for j in range(2, 5):
                cosine = abs(self._angle(approx[j % 4], approx[j - 2], approx[j - 1]))
                max_cosine = max(max_cosine, cosine)

            return max_cosine < self.config.max_cosine_limit

        return False


type Rectangle = np.ndarray[tuple[Literal[4], Literal[2]], np.dtype[np.int32]]


class Visualizer:
    """検出結果の可視化を行う"""

    COLOR_BGR: ClassVar = {"Red": (0, 0, 255), "Blue": (255, 0, 0), "Yellow": (0, 255, 255)}

    @staticmethod
    def draw_colored_squares(
        img: cv2t.MatLike,
        colored_squares: dict[ColorName, list[tuple[Rectangle, float]]],
    ) -> cv2t.MatLike:
        """検出した色付き四角形を描画"""
        result_img = img.copy()

        for color_name, squares in colored_squares.items():
            color_bgr = Visualizer.COLOR_BGR[color_name]

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

    @staticmethod
    def draw_color_masks(
        img: cv2t.MatLike,
        masks: dict[ColorName, ImageMask],
    ) -> dict[ColorName, ColorImage]:
        """各色のフィルター後の画像を生成"""
        return {
            color_name: cv2.bitwise_and(img, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR))  # type: ignore
            for color_name, mask in masks.items()
        }


class ColorSquareDetector:
    """色付き四角形の検出を統合するクラス"""

    def __init__(self, image_height: int, image_width: int, config: ColorConfig) -> None:
        self.color_detector = ColorDetector(image_height, image_width)
        self.square_detector = SquareDetector(config)
        self.visualizer = Visualizer()
        self.config = config

    def detect_colored_squares(
        self,
        img: ColorImage,
    ) -> dict[ColorName, list[tuple[Rectangle, float]]]:
        """画像から色付き四角形を検出"""
        result: dict[ColorName, list[tuple[Rectangle, float]]] = {}

        for color_name, color_range in self.config.color_ranges.items():
            mask = self.color_detector.create_color_mask(img, color_range)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            squares: list[tuple[Rectangle, float]] = []
            for contour in contours:
                if self.square_detector.is_square(contour):
                    arc_len = cv2.arcLength(contour, closed=True)
                    approx = cv2.approxPolyDP(
                        contour,
                        arc_len * self.config.epsilon_factor,
                        closed=True,
                    )
                    rectangle: Rectangle = approx.reshape(4, 2)  # type: ignore
                    area = abs(cv2.contourArea(approx))
                    squares.append((rectangle, area))

            result[color_name] = squares

        return result
