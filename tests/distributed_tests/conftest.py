from pathlib import Path

default_exec_file = Path(__file__).absolute().parents[2] / "lightgbm"


def pytest_addoption(parser):
    parser.addoption("--execfile", action="store", default=str(default_exec_file))
