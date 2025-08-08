"""円検出を試すだけのテストスクリプト"""

import typing

import cv2
import cv2.typing as cv2t
import numpy as np

from vmx_remotes.color_square_detector import ColorDetector, ColorImage, ColorSquareDetector
from vmx_remotes.config import ColorConfig, load_preferences
from vmx_remotes.robot_app import HEIGHT, WIDTH

# 使用する画像ファイルのパス
IMG_SRC = "./saved_images/img.jpg"


# 円のセンターマークを描画
def draw_center_mark(image: ColorImage, point: cv2t.Point2f, radius: float) -> None:
    npt = (int(point[0]), int(point[1]))
    r = int(radius)
    cv2.circle(image, npt, r, (255, 0, 0), 2)

    cv2.drawMarker(image, npt, (0, 0, 0), markerType=cv2.MARKER_CROSS, markerSize=10, thickness=1)


def get_circle(
    frame: ColorImage,
    color_detector: ColorDetector,
    config: ColorConfig,
) -> list[tuple[cv2t.Point2f, float]]:
    """
    円検出をする

    See:
    https://blog.ashija.net/2017/11/28/post-2549/
    https://emotionexplorer.blog.fc2.com/blog-entry-228.html
    """
    MIN_RADIUS = 10  # noqa: N806
    MAX_RADIUS = 23  # noqa: N806

    # # HSVによる画像情報に変換
    # hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # # ガウシアンぼかしを適用して、認識精度を上げる
    blur = cv2.GaussianBlur(frame, (3, 3), 0)

    color = color_detector.create_color_mask(
        typing.cast("ColorImage", blur),
        config.color_ranges["Yellow"],
    )

    cv2.imshow("blur", blur)
    cv2.imshow("masked", color)

    # ハフ変換
    circles: cv2t.MatLike | None = cv2.HoughCircles(
        color,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=color.shape[0] / 3,
        # param1, param2はよくわからない
        param1=40,
        param2=7,
        minRadius=MIN_RADIUS,
        maxRadius=MAX_RADIUS,
    )
    if circles:
        circles = np.around(circles).astype(np.uint32)
        return [((center_x, center_y), radius) for (center_x, center_y, radius) in circles[0, :]]
    return []


def main() -> None:
    img_src = typing.cast("ColorImage", cv2.imread(IMG_SRC, 1))

    detector = ColorSquareDetector(HEIGHT, WIDTH, load_preferences())

    img_dst = img_src.copy()

    circles = get_circle(img_src, detector.color_detector, detector.config)
    for center, radius in circles:
        # 見つかった円の上に青い円を描画
        draw_center_mark(img_dst, center, radius)
        print(center, radius)

    cv2.imshow("result(YELLOW)", img_dst)

    cv2.waitKey()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
