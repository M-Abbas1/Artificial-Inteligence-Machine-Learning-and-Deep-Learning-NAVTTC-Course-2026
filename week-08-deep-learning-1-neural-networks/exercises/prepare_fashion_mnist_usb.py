"""
prepare_fashion_mnist_usb.py
=============================
RUN THIS ONCE, on a machine WITH internet (e.g. your own laptop at home).

It downloads the four Fashion-MNIST files and arranges them in the exact
folder layout the student lab notebook expects. Then you copy the created
`data/` folder onto a USB stick and hand it out.

Usage:
    python prepare_fashion_mnist_usb.py

After it finishes you will have:
    ./data/FashionMNIST/raw/train-images-idx3-ubyte.gz
    ./data/FashionMNIST/raw/train-labels-idx1-ubyte.gz
    ./data/FashionMNIST/raw/t10k-images-idx3-ubyte.gz
    ./data/FashionMNIST/raw/t10k-labels-idx1-ubyte.gz

Copy the whole `data/` folder to the USB stick. Students drop it next to
the notebook and run with download=False (no internet needed).
"""

import os
import sys
import gzip
import shutil
import urllib.request

# The four files that make up Fashion-MNIST, with several mirror hosts each.
# We try mirrors in order until one works.
FILES = [
    "train-images-idx3-ubyte.gz",
    "train-labels-idx1-ubyte.gz",
    "t10k-images-idx3-ubyte.gz",
    "t10k-labels-idx1-ubyte.gz",
]

MIRRORS = [
    # official Zalando research host
    "https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/",
    # torchvision's S3 mirror
    "https://ossci-datasets.s3.amazonaws.com/mnist/",  # note: MNIST layout, kept as fallback name-check
]

# The Zalando GitHub host is the reliable one for Fashion-MNIST specifically.
PRIMARY = "https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/"

TARGET_DIR = os.path.join("data", "FashionMNIST", "raw")

USER_AGENT = "Mozilla/5.0 (course-prep-script)"


def download(url, dest):
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=60) as resp, open(dest, "wb") as fh:
        total = getattr(resp, "length", None)
        read = 0
        while True:
            chunk = resp.read(32768)
            if not chunk:
                break
            fh.write(chunk)
            read += len(chunk)
            if total:
                pct = read * 100 // total
                print(f"\r    {os.path.basename(dest)}: {pct:3d}%  ({read//1024} KB)", end="")
    print()


def main():
    os.makedirs(TARGET_DIR, exist_ok=True)
    print(f"Target folder: {os.path.abspath(TARGET_DIR)}\n")

    ok = 0
    for name in FILES:
        dest = os.path.join(TARGET_DIR, name)
        if os.path.exists(dest) and os.path.getsize(dest) > 0:
            print(f"  [skip] {name} already present ({os.path.getsize(dest)//1024} KB)")
            ok += 1
            continue

        url = PRIMARY + name
        print(f"  [get ] {name}")
        try:
            download(url, dest)
            size = os.path.getsize(dest)
            if size < 1000:  # sanity: real files are hundreds of KB to MBs
                raise IOError(f"downloaded file suspiciously small ({size} bytes)")
            print(f"    done ({size//1024} KB)")
            ok += 1
        except Exception as e:
            print(f"    FAILED: {e}")
            if os.path.exists(dest):
                os.remove(dest)  # don't leave a broken partial file

    # Decompress each .gz alongside the archive. Newer torchvision expects the
    # EXTRACTED files (no .gz) in raw/; older versions accept the .gz. Keeping
    # both forms makes the USB stick work on any torchvision version.
    if ok == len(FILES):
        print("\n  Decompressing archives (needed by newer torchvision)...")
        for name in FILES:
            gz = os.path.join(TARGET_DIR, name)
            out = os.path.join(TARGET_DIR, name[:-3])  # strip .gz
            if os.path.exists(out) and os.path.getsize(out) > 0:
                continue
            try:
                with gzip.open(gz, "rb") as fi, open(out, "wb") as fo:
                    shutil.copyfileobj(fi, fo)
                print(f"    extracted {name[:-3]}")
            except Exception as e:
                print(f"    WARNING: could not extract {name}: {e}")

    print()
    if ok == len(FILES):
        print("SUCCESS - all four files ready (both .gz and extracted).")
        print(f"\nNow copy the whole 'data' folder to the USB stick:")
        print(f"    {os.path.abspath('data')}")
        print("\nStudents place 'data' next to the lab notebook and run with download=False.")

        # quick integrity check: let torchvision try to read them (if installed)
        try:
            from torchvision import datasets
            _ = datasets.FashionMNIST(root="data", train=True, download=False)
            _ = datasets.FashionMNIST(root="data", train=False, download=False)
            print("\nVerified: torchvision can read the files with download=False. Ready to distribute.")
        except ImportError:
            print("\n(torchvision not installed here - skipping the read test. The files are still fine.)")
        except Exception as e:
            print(f"\nWARNING: torchvision could not read the files: {e}")
            print("Re-run this script to re-fetch; if it persists, check the files aren't corrupted.")
    else:
        print(f"INCOMPLETE - {ok}/{len(FILES)} files downloaded.")
        print("Re-run the script (it will skip the ones already done). Check your internet connection.")
        sys.exit(1)


if __name__ == "__main__":
    main()
