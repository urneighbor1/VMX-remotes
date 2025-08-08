"""ローカルのテスト用スクリプト"""

from typing import cast

import cv2

from vmx_remotes.color_square_detector import ColorImage, ColorSquareDetector
from vmx_remotes.config import load_preferences
from vmx_remotes.robot_app import HEIGHT, WIDTH, display_results, save_image


def main() -> None:
    detector = ColorSquareDetector(HEIGHT, WIDTH, load_preferences())
    cap = cv2.VideoCapture(1)
    while True:
        ret, frame = cap.read()
        frame = cast("ColorImage", cv2.resize(frame, (WIDTH, HEIGHT)))

        # 色付き四角形を検出
        colored_squares = detector.detect_colored_squares(frame)

        # 結果を画像に描画して表示
        display_results(frame, detector, colored_squares)

        key = cv2.waitKey(100)
        if key == ord("s"):
            # 元の画像を保存
            save_image(frame)


if __name__ == "__main__":
    main()
