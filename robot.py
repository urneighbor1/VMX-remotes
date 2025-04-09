import logging
import math
from typing import Literal, NoReturn

import cv2
import cv2.typing as cv2t
import ntcore
import numpy as np
from cscore import CameraServer, CvSink, HttpCamera, VideoMode

NetworkTables = ntcore.NetworkTableInstance.getDefault()
SERVERS = ["127.0.0.1", "10.12.34.2"]
PORT = NetworkTables.kDefaultPort3


def find_server(server_ips: list[str]) -> str:
    import socket

    """接続できるNetworkTablesサーバーを探す"""
    for server in server_ips:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(1.0)
        try:
            s.connect((server, PORT))
            s.close()
        except OSError as e:
            print(e)
        else:
            return server
        finally:
            s.close()
    msg = "SERVERSのどれにも接続できませんでした"
    raise RuntimeError(msg)


SERVER_IP = find_server(SERVERS)


class ImageProcess:
    img_width: int
    img_height: int

    # メモリ領域を使い回すためのバッファー
    _gray_img: cv2.Mat
    _bin_img: cv2.Mat
    _rectangles_img: cv2.Mat

    def __init__(self, image_width: int, image_height: int) -> None:
        self.img_width = image_width
        self.img_height = image_height

        self._gray_img = cv2.Mat(np.zeros((image_width, image_height), dtype=np.uint8))
        self._bin_img = cv2.Mat(np.zeros((image_width, image_height), dtype=np.uint8))
        self._rectangles_img = cv2.Mat(np.zeros((image_width, image_height, 3), dtype=np.uint8))

    def get_input_buffer(
        self,
    ) -> cv2.Mat:
        return cv2.Mat(np.zeros((self.img_width, self.img_height, 3), dtype=np.uint8))

    type Rectangle = np.ndarray[tuple[Literal[4], Literal[2]], np.dtype[np.int32]]

    @staticmethod
    def _angle(pt1: cv2t.MatLike, pt2: cv2t.MatLike, pt0: cv2t.MatLike) -> float:
        dx1 = float(pt1[0, 0] - pt0[0, 0])
        dy1 = float(pt1[0, 1] - pt0[0, 1])
        dx2 = float(pt2[0, 0] - pt0[0, 0])
        dy2 = float(pt2[0, 1] - pt0[0, 1])
        v = math.sqrt((dx1 * dx1 + dy1 * dy1) * (dx2 * dx2 + dy2 * dy2))
        return (dx1 * dx2 + dy1 * dy2) / v

    def find_squares(
        self,
        img: cv2t.MatLike,
    ) -> list[Rectangle]:
        """imgから四角形を検出する"""
        # 改変元: https://emotionexplorer.blog.fc2.com/blog-entry-281.html
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY, self._gray_img)
        cv2.imshow("gray img", gray)
        _, bin_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU, self._bin_img)
        cv2.imshow("bin img", bin_img)

        # TODO:  意味のある値にする, NetworkTablesで調整出来るようにする
        find_area_min = 100
        find_area_max = 500

        found_rectangles: list[ImageProcess.Rectangle] = []

        contours, _ = cv2.findContours(bin_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            arc_len = cv2.arcLength(contour, closed=True)
            approx = cv2.approxPolyDP(contour, arc_len * 0.02, closed=True)

            area = abs(cv2.contourArea(approx))
            if (
                # 四角形(==頂点の数が4)であるか
                approx.shape[0] == 4  # noqa: PLR2004
                # 面積の範囲
                and find_area_min < area < find_area_max
                # 凸多角形である(==凹みがない)か
                and cv2.isContourConvex(approx)
            ):
                max_cosine = 0
                max_cosine_limit = 0.3

                for j in range(2, 5):
                    # 辺間の角度の最大コサインを算出
                    cosine = abs(self._angle(approx[j % 4], approx[j - 2], approx[j - 1]))
                    max_cosine = max(max_cosine, cosine)

                if max_cosine < max_cosine_limit:
                    # 四角形を見つけた
                    rectangle: ImageProcess.Rectangle = approx.reshape(-1, 2)  # type: ignore
                    found_rectangles.append(rectangle)
        return found_rectangles


def main() -> NoReturn:
    logging.basicConfig(level=logging.DEBUG)
    CameraServer.enableLogging()

    NetworkTables.startClient3("Py Image Processor")
    NetworkTables.setServer(SERVER_IP)
    NetworkTables.startDSClient()

    WIDTH = 160
    HEIGHT = 120

    camera = HttpCamera("Camera", f"http://{SERVER_IP}:1181/?action=stream")
    camera.setVideoMode(VideoMode.PixelFormat(VideoMode.PixelFormat.kMJPEG), WIDTH, HEIGHT, 5)

    sink = CameraServer.getVideo(camera)

    im_process = ImageProcess(WIDTH, HEIGHT)
    input_img = im_process.get_input_buffer()
    table = NetworkTables.getTable("PyTest")
    found_squares_amount = table.getIntegerTopic("Found squares").publish()

    while True:
        grabbed_time, input_img = CvSink.grabFrame(sink, input_img)
        if grabbed_time == 0:
            continue

        found_squares = im_process.find_squares(input_img)

        found_squares_amount.set(len(found_squares))

        cv2.imshow("image", input_img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()
