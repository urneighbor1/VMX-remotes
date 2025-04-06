import time

# from cscore import CameraServer, CvSink
from networktables import NetworkTables
# import numpy as np
# import cv2

NetworkTables.setServerTeam(1234)
NetworkTables.initialize(server="127.0.0.1")

# CameraServer.enableLogging()


def main():
    # camera = CameraServer.startAutomaticCapture()
    # camera.setResolution(360, 680)
    # sink = CameraServer.getVideo()

    # print(camera.isConnected())

    mode_value = NetworkTables.getEntry("/Py Mode")
    mode_value.setString("none")
    prev_mode = mode_value.getString("none")

    # input_img = cv2.Mat(np.empty((360, 680, 3)))
    while True:
        time.sleep(0.5)
        mode = mode_value.getString("none")
        if mode != prev_mode:
            print(f"mode: {mode}")
        prev_mode = mode

        # grabbed_time, input_img = CvSink.grabFrame(sink, input_img)

        # cv2.imshow("image", input_img)
        # cv2.waitKey(1)


if __name__ == "__main__":
    main()
