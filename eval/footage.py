"""Where the test footage lives. The footage itself is private and never committed; every test that needs a clip
skips when it is missing.

Environment overrides (all optional):
  STILLPOINT_FOOTAGE_DIR   base folder for the defaults below (default: ~/Desktop)
  STILLPOINT_O3_DIR        DJI O3 / Avata originals, DJI_00XX.MP4        (default: <base>/untitled folder 4)
  STILLPOINT_GYROFLOW_DIR  Gyroflow renders of those clips, for eval     (default: <base>/untitled folder 5)
  STILLPOINT_OA4_DIR       Osmo Action 4 clips, DJI_2026..._D.MP4        (default: <base>)
  STILLPOINT_SD_DIR        a mounted SD card's DCIM folder               (default: /Volumes/Untitled/DCIM/DJI_001)
"""
from __future__ import annotations

import os


def _env_dir(name: str, default: str) -> str:
    v = os.environ.get(name, '')
    return os.path.abspath(os.path.expanduser(v if v else default))


BASE = _env_dir('STILLPOINT_FOOTAGE_DIR', '~/Desktop')
O3_DIR = _env_dir('STILLPOINT_O3_DIR', os.path.join(BASE, 'untitled folder 4'))
GYROFLOW_DIR = _env_dir('STILLPOINT_GYROFLOW_DIR', os.path.join(BASE, 'untitled folder 5'))
OA4_DIR = _env_dir('STILLPOINT_OA4_DIR', BASE)
SD_DIR = _env_dir('STILLPOINT_SD_DIR', '/Volumes/Untitled/DCIM/DJI_001')


def o3(name: str) -> str:
    return os.path.join(O3_DIR, name)


def oa4(name: str) -> str:
    return os.path.join(OA4_DIR, name)
