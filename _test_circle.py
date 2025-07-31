"""円検出を試すだけのテストスクリプト"""

import typing

import cv2
from cv2.typing import Point2f

from color_square_detector import ColorDetector, ColorImage, ColorSquareDetector
from config import ColorConfig, load_preferences
from robot import HEIGHT, WIDTH

# 使用する画像ファイルのパス
IMG_SRC = "./saved_images/img.jpg"


def get_circle(
    frame: ColorImage,
    color_detector: ColorDetector,
    config: ColorConfig,
) -> list[tuple[Point2f, float]]:
    """See: https://blog.ashija.net/2017/11/28/post-2549/"""
    # # HSVによる画像情報に変換
    # hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # # ガウシアンぼかしを適用して、認識精度を上げる
    blur = cv2.GaussianBlur(frame, (3, 3), 0)

    color = color_detector.create_color_mask(
        typing.cast("ColorImage", blur),
        config.color_ranges["Yellow"],
    )

    # 輪郭抽出
    contours, _hierarchy = cv2.findContours(color, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(f"{len(contours)} contours.")

    cv2.imshow("blur", blur)
    cv2.imshow("masked", color)

    return [cv2.minEnclosingCircle(contour) for contour in contours]


def main() -> None:
    img_src = typing.cast("ColorImage", cv2.imread(IMG_SRC, 1))

    detector = ColorSquareDetector(HEIGHT, WIDTH, load_preferences())

    img_dst = img_src.copy()

    circles = get_circle(img_src, detector.color_detector, detector.config)
    for center, radius in circles:
        # 見つかった円の上に青い円を描画
        center_int = (round(center[0]), round(center[1]))
        cv2.circle(img_dst, center_int, int(radius), (255, 0, 0), 2)
        print(center, radius)

    cv2.imshow("result(YELLOW)", img_dst)

    cv2.waitKey()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
