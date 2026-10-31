"""Usage: python scripts/download_datasets.py [ims]   (no argument: list all datasets)"""
import sys, urllib.request, zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent / "datasets"
AUTO = {"ims": ("https://phm-datasets.s3.amazonaws.com/NASA/4.+Bearings.zip", "ims/NASA_IMS_Bearings.zip")}
MANUAL = {
    "cwru": ("Bearing fault classification", "https://engineering.case.edu/bearingdatacenter"),
    "paderborn": ("Bearing faults, varied conditions", "https://mb.uni-paderborn.de/en/kat/research/bearing-datacenter/data-sets-and-download"),
    "paderborn_run_to_failure": ("Degradation / RUL (large, pick a subset)", "https://zenodo.org/records/10868257"),
    "mimii": ("Acoustic anomaly detection", "https://zenodo.org/records/3384388"),
    "mimii_due": ("Acoustic, domain shift", "https://zenodo.org/records/4740101"),
    "cmapss": ("RUL methodology (turbofan, not compressor data)", "https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/"),
}


def fetch(name):
    url, rel = AUTO[name]
    dest = ROOT / rel
    dest.parent.mkdir(parents=True, exist_ok=True)
    print("Downloading", url)
    urllib.request.urlretrieve(url, dest, lambda b, bs, t: print(f"\r{b * bs / 1e6:.0f} MB", end=""))
    print("\nExtracting to", dest.parent / "extracted")
    zipfile.ZipFile(dest).extractall(dest.parent / "extracted")
    print("Done. The archive contains a nested IMS .7z: extract it, then run: python -m ml.ims_demo <test-set folder>")


if __name__ == "__main__":
    args = sys.argv[1:]
    for a in args:
        fetch(a) if a in AUTO else print("Unknown or manual-only dataset:", a)
    if not args:
        print("Auto-download: ims")
        for k, (use, link) in MANUAL.items():
            print(f"Manual: {k:26s} {use}\n        {link}")
        print(f"\nSave manual downloads under {ROOT}/<name>/")
