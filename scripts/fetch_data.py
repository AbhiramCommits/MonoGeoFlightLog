#!/usr/bin/env python
"""Attempt to fetch a real dataset, or explain exactly how to place it locally.

The repository is designed to run fully offline on the synthetic renderer, so
this script never *requires* a download. It tries the network, and if that
fails (or the dataset needs manual registration) it prints the directory layout
:mod:`monogeo.data` expects and exits 0 with a clear message.

Usage::

    python scripts/fetch_data.py --dataset kitti_odom --out data/kitti_odom
    python scripts/fetch_data.py --dataset euroc --out data/euroc
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

KITTI_NOTE = """\
KITTI odometry ground truth + velodyne data needs a one-time browser download
from https://www.cvlibs.net/datasets/kitti/eval_odometry.php (free, no login).
Place the unzipped files so that `monogeo.data.load_kitti_odometry` finds:

    {root}/sequences/<seq>/image_0/*.png
    {root}/sequences/<seq>/calib.txt
    {root}/sequences/<seq>/poses.txt          # optional GT poses
    {root}/sequences/<seq>/velodyne_points/data/*.bin   # optional GT depth

Then run:  monogeo evaluate --sequence kitti:00
"""

EUROC_NOTE = """\
EuRoC MAV needs a one-time browser download from
https://projects.asl.ethz.ch/datasets/doku.php?id=kmavvisualinertialdatasets
(e.g. MH_01_easy.zip, ASL format, CC BY 4.0). Place it so that
`monogeo.data.load_euroc` finds:

    {root}/<seq>/mav0/cam0/data/*.png
    {root}/<seq>/mav0/cam0/sensor.yaml
    {root}/<seq>/mav0/state_groundtruth_estimate0/data.csv

Then run:  monogeo evaluate --sequence euroc:MH_01_easy
"""

URLS = {
    # Small, scriptable subsets. Full KITTI/EuRoC need the browser flow above.
    "kitti_odom": [
        "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_odometry_calib.zip",
    ],
    "euroc": [
        "https://zenodo.org/record/3524720/files/MH_01_easy.zip",
    ],
}


def try_download(url: str, dest: Path, timeout: int = 20) -> bool:
    """Attempt a single download with curl or urllib.

    Args:
        url: Source URL.
        dest: Destination file path.
        timeout: Network timeout in seconds.

    Returns:
        ``True`` if the file was written, ``False`` otherwise.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    curl = shutil.which("curl")
    if curl:
        res = subprocess.run(
            [curl, "-fL", "--max-time", str(timeout), "-o", str(dest), url],
            capture_output=True,
            text=True,
        )
        return res.returncode == 0 and dest.exists() and dest.stat().st_size > 0

    try:  # pragma: no cover - curl is present in CI and locally
        import urllib.request

        with urllib.request.urlopen(url, timeout=timeout) as resp:
            dest.write_bytes(resp.read())
        return True
    except Exception:
        return False


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Always exits 0: absence of network is not an error."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", choices=["kitti_odom", "euroc"], required=True)
    ap.add_argument("--out", default=None, help="destination root (default: data/<dataset>)")
    ap.add_argument("--timeout", type=int, default=20)
    ap.add_argument("--force", action="store_true", help="re-download even if present")
    args = ap.parse_args(argv)

    root = Path(args.out) if args.out else Path("data") / args.dataset
    marker = root / ".downloaded"

    if marker.exists() and not args.force:
        print(f"[fetch_data] {root} already marked as downloaded; nothing to do.")
        return 0

    print(f"[fetch_data] dataset={args.dataset} root={root}")
    ok_any = False
    for url in URLS.get(args.dataset, []):
        name = url.rsplit("/", 1)[-1]
        dest = root / name
        if dest.exists() and not args.force:
            print(f"[fetch_data]   already present: {dest}")
            ok_any = True
            continue
        print(f"[fetch_data]   trying {url} ...")
        if try_download(url, dest, timeout=args.timeout):
            print(f"[fetch_data]   downloaded {dest} ({dest.stat().st_size} bytes)")
            ok_any = True
            print(
                "[fetch_data]   NOTE: unzip it and check the layout below before running "
                "the pipeline."
            )
        else:
            print(f"[fetch_data]   could not download {url}")

    note = (KITTI_NOTE if args.dataset == "kitti_odom" else EUROC_NOTE).format(root=root)
    if ok_any:
        print("\n[fetch_data] partial download succeeded. Remaining manual steps:\n")
    else:
        print(
            "\n[fetch_data] no download succeeded (offline, or the host requires a browser).\n"
            "              This is expected and fine: MonoGeoFlightLog runs on the deterministic\n"
            "              synthetic renderer by default. To use real data:\n"
        )
    print(note)
    print(
        "\n[fetch_data] The synthetic fallback needs no data at all:\n"
        "    python -c \"from monogeo.data import load_sequence; print(load_sequence('forward')[0].shape)\"\n"
    )
    root.mkdir(parents=True, exist_ok=True)
    if ok_any:
        marker.write_text("partial\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
