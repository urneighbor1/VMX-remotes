import logging

from cscore import CameraServer, CvSink, HttpCamera, VideoMode
from networktables import NetworkTables
import numpy as np
import cv2


logging.basicConfig(level=logging.DEBUG)


NetworkTables.setServerTeam(1234)
NetworkTables.initialize(server="127.0.0.1")
NetworkTables.startDSClient()

CameraServer.enableLogging()


def main():
    camera = HttpCamera("Camera", "http://127.0.0.1:1181/?action=stream")
    camera.setVideoMode(VideoMode.PixelFormat(VideoMode.PixelFormat.kMJPEG), 160, 120, 10)

    sink = CameraServer.getVideo(camera)

    print(camera.isConnected())

    mode_value = NetworkTables.getEntry("/Py Mode")
    mode_value.setString("none")
    prev_mode = mode_value.getString("none")

    input_img = cv2.Mat(np.empty((160, 120, 3)))
    while True:
        mode = mode_value.getString("none")
        if mode != prev_mode:
            print(f"mode: {mode}")
        prev_mode = mode

        grabbed_time, input_img = CvSink.grabFrame(sink, input_img)

        cv2.imshow("image", input_img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()
