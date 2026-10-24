import socket


def find_random_open_port() -> int:
    """Find a random open port on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        port = s.getsockname()[1]
    return port  # noqa: RET504
