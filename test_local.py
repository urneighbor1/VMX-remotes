"""ローカルのテスト用スクリプト"""
import cv2

from color_square_detector import ColorImage, ColorSquareDetector
from config import load_preferences
from robot import HEIGHT, SAVED_IMAGES_DIR, WIDTH, display_results


def load_images() -> list[ColorImage]:
    return [cv2.imread(image_path.as_posix()) for image_path in SAVED_IMAGES_DIR.glob("*.jpg")]  # type: ignore


def main() -> None:
    images = load_images()
    detector = ColorSquareDetector(HEIGHT, WIDTH, load_preferences())

    while True:
        for input_img in images:
            # 色付き四角形を検出
            colored_squares = detector.detect_colored_squares(input_img)

            # 結果を画像に描画して表示
            display_results(input_img, detector, colored_squares)

            cv2.waitKey(100)


if __name__ == "__main__":
    main()
