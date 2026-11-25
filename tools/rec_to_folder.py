"""Convert an InsightFace MXNet RecordIO pack (train.rec) to  out/<label>/<n>.jpg  without needing mxnet.

  python tools/rec_to_folder.py --rec data/faces_webface_112x112/train.rec --out data/casia_112

RecordIO layout: [magic u32 0xced7230a][lrec u32: cflag<<29 | length][payload][pad to 4 bytes].
Image payload: IRHeader(flag u32, label f32, id u64, id2 u64) + (flag extra f32 labels if flag>0) + JPEG/PNG bytes.
Records that carry no image (the pack's metadata record) are skipped.
NOTE: written from the format spec; sanity-check the first output folders (a few identities, face crops).
"""
import argparse
import os
import struct

MAGIC = 0xCED7230A


def records(path):
    with open(path, "rb") as f:
        while True:
            head = f.read(8)
            if len(head) < 8:
                return
            magic, lrec = struct.unpack("<II", head)
            if magic != MAGIC:
                raise ValueError("bad RecordIO magic; wrong file?")
            length = lrec & ((1 << 29) - 1)
            data = f.read(length)
            f.read((4 - length % 4) % 4)
            yield data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rec", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    counts, n_skip = {}, 0
    for data in records(a.rec):
        flag, label = struct.unpack("<If", data[:8])
        off = 24 + (4 * flag if flag > 0 else 0)
        img = data[off:]
        if flag > 0:                                  # array label: use the first value
            label = struct.unpack("<f", data[24:28])[0]
        if not (img[:2] == b"\xff\xd8" or img[:4] == b"\x89PNG"):
            n_skip += 1                               # metadata record, not an image
            continue
        lab = int(label)
        d = os.path.join(a.out, f"{lab:06d}")
        os.makedirs(d, exist_ok=True)
        counts[lab] = counts.get(lab, 0) + 1
        ext = ".jpg" if img[:2] == b"\xff\xd8" else ".png"
        with open(os.path.join(d, f"{counts[lab]:04d}{ext}"), "wb") as f:
            f.write(img)
    print(f"wrote {sum(counts.values())} images, {len(counts)} identities ({n_skip} non-image records skipped)")


if __name__ == "__main__":
    main()
