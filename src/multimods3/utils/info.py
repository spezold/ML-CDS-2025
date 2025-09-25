from pathlib import Path
import socket
import subprocess
import sys


def base_dir() -> Path:
    """
    :return: The base directory of this Python project
    """
    return Path(__file__).resolve().parents[3]


def _output_of(cmd: list[str]) -> str:
    try:
        result = subprocess.check_output(cmd, cwd=base_dir(), text=True, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as e:
        result = e.output if e.output else str(e)
    return result.strip("\n")


def git_status() -> str:
    """
    :return: the result of calling ``git status --porcelain`` on the project's base dir
    """
    return _output_of(["git",  "status", "--porcelain"])


def git_commit() -> str:
    """
    :return: the result of calling ``git rev-parse --verify HEAD`` on the project's base dir
    """
    return _output_of(["git", "rev-parse", "--verify", "HEAD"])


def machine() -> str:
    """
    :return: the current machine's name
    """
    return socket.gethostname()


def environment() -> str:
    """
    :return: the result of calling ``pip list`` to provide an overview of the current environment (note that on a conda
        environment, this does neither necessarily include all installed packages, nor do the conda package names
        necessarily correspond to the listed pip package names)
    """
    return _output_of(["pip", "list"])


def cmd() -> str:
    """
    :return: the launch command of this run
    """
    return " ".join(sys.argv)
