import ntcore
from cscore import CameraServer, CvSink, HttpCamera, VideoMode

__all__ = [
    "NetworkTables",
    "get_server_ip",
    "init_camera_sink",
    "init_network_tables",
]

NetworkTables = ntcore.NetworkTableInstance.getDefault()
SERVERS = ["127.0.0.1", "10.12.34.2"]


def find_network_tables_server(server_ips: list[str]) -> str:
    """接続できるNetworkTablesサーバーを探す"""
    import socket

    port = ntcore.NetworkTableInstance.kDefaultPort3
    for server in server_ips:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(1.0)
        try:
            s.connect((server, port))
            s.close()
        except OSError as e:
            print(e)
        else:
            return server
        finally:
            s.close()
    msg = "SERVERSのどれにも接続できませんでした"
    raise RuntimeError(msg)


def get_server_ip() -> str:
    return find_network_tables_server(SERVERS)


def init_network_tables(server_ip: str) -> ntcore.NetworkTableInstance:
    NetworkTables.startClient3("Py Image Processor")
    NetworkTables.setServer(server_ip)
    NetworkTables.startDSClient()

    return NetworkTables


def init_camera_sink(server_ip: str, width: int, height: int, fps: int) -> CvSink:
    camera = HttpCamera("Camera", f"http://{server_ip}:1181/?action=stream")
    camera.setVideoMode(VideoMode.PixelFormat(VideoMode.PixelFormat.kMJPEG), width, height, fps)

    return CameraServer.getVideo(camera)
