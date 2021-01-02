import os
import sys
import re
import argparse
import msgpack
import imageio


def parse_args():
    parser = argparse.ArgumentParser(description="")

    parser.add_argument("-v", "--verbose", action="store_true", help="verbose output")
    parser.add_argument("-m", "--msgpack_path", help="verbose output")
    args = parser.parse_args()
    return args


def list_files(path, patter=r".*?\.rec"):
    patter_re = re.compile(patter)
    return [os.path.join(path, x) for x in os.listdir(path) if re.match(patter_re, x)]


def main():
    args = parse_args()
    shard_re = r"shard_(\d+).msg"

    shards_paths = []
    for p in [args.msgpack_path]:
        shards_paths.extend([os.path.join(p, x) for x in os.listdir(p) if re.match(shard_re, x)])
    count_a = 0
    count_sum = 0
    count = 0
    for path in shards_paths:
        with open(path, "rb") as f:
            packer = msgpack.Unpacker(f, max_buffer_size=1024 * 1024 * 1024, raw=True)
            for p in packer:
                if b"additional" in p:
                    count_a += 1
                    count_sum += len(p[b"additional"])
                count += 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
