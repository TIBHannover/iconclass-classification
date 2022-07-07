import os
import sys
import re
import argparse
import json
import multiprocessing as mp
from rectpack import newPacker
import imageio
import numpy as np
import cv2

from PIL import Image

Image.warnings.simplefilter("error", Image.DecompressionBombError)  # turn off Decompression bomb error
Image.warnings.simplefilter("error", Image.DecompressionBombWarning)  # turn off Decompression bomb warning
Image.MAX_IMAGE_PIXELS = 1000000000  # set max pixel up high


def parse_args():
    parser = argparse.ArgumentParser(description="")

    parser.add_argument("-v", "--verbose", action="store_true", help="verbose output")
    parser.add_argument("-i", "--input_path", nargs="+", help="verbose output")

    parser.add_argument("--image_path", help="verbose output")
    parser.add_argument("-o", "--output_path", help="verbose output")
    args = parser.parse_args()
    return args


def pack(args):
    def pack_iter(image_size, rects):
        packer = newPacker(rotation=False)
        packer.add_bin(*image_size)

        num_rects = 0
        for i, rect in enumerate(rects):
            num_rects += 1
            # print(rect)
            packer.add_rect(rect[2] - rect[0], rect[3] - rect[1], i)
        packer.pack()

        for abin in packer:
            if len(abin) != num_rects:
                return None
            rects = []
            for r in abin:
                rects.append([r.rid, r.x, r.y, r.x + r.width, r.y + r.height])
            rects = sorted(rects, key=lambda x: x[0])
            return [r[1:] for r in rects]

        return None

    # print(args)
    image_size = 24
    for x in range(80):
        packing_result = pack_iter([image_size, image_size], args["bbox"])
        if packing_result is not None:
            return {**args, "packed_bbox": packing_result}
        image_size *= 1.1

    return None


def read_image(path, max_dim=None):
    try:
        image = imageio.imread(path)
    except:
        return None
    if len(image.shape) == 2:
        image = np.stack([image] * 3, axis=-1)

    if image.shape[-1] == 4:
        image = image[:, :, 0:3]

    if max_dim is not None:

        shape = np.asarray(image.shape[:-1], dtype=np.float32)
        long_dim = max(shape)
        scale = min(1, max_dim / long_dim)

        new_shape = np.asarray(shape * scale, dtype=np.int32)
        scale_image = cv2.resize(image, tuple(new_shape[::-1].tolist()))

        return scale_image
    return image


def generate_boxes_image(args):
    try:
        args["data"]["bbox"] = [b for b in args["data"]["bbox"] if b[2] - b[0] > 64 and b[3] - b[1] > 64]

        data = pack(args["data"])
        if data is None:
            print(f"packing not working {data}")
            return None
        image_path = os.path.join(args["args"].image_path, data["path"])
        image = read_image(image_path)
        if image is None:

            print(f"image not loaded {data}")
            return
        max_boxes = np.max(np.asarray(data["packed_bbox"])[:, 2:], axis=0)
        new_image = np.zeros([max_boxes[1], max_boxes[0], 3], dtype=np.uint8)

        for old_rect, new_rect in zip(data["bbox"], data["packed_bbox"]):
            new_image[new_rect[1] : new_rect[3], new_rect[0] : new_rect[2]] = image[
                old_rect[1] : old_rect[3], old_rect[0] : old_rect[2]
            ]
        output_path = os.path.join(args["args"].output_path, data["path"])
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        imageio.imsave(output_path, new_image)
    except KeyboardInterrupt as e:
        raise e

    except Exception as e:
        print(f"{e} {data}")
        return None


# def add_boxes_path(boxes):
#     result = []
#     for x in boxes:
#         m = re.match(r"^(\d{0,4}/[0-9a-f]+)\.(.*?)$", x["path"])
#         if not m:
#             print(x)
#             continue
#         paths = []
#         for i, box in enumerate(x["bbox"]):
#             path = f"{m.group(1)}_bbbox{i}.jpg"
#             paths.append(path)
#         result.append({**x, "paths": paths})
#     return result


def main():
    args = parse_args()

    data = []
    for file in args.input_path:
        with open(file) as f:
            for line in f:
                data.append(json.loads(line))
    # data = add_boxes_path(data)

    pool = mp.Pool(32)

    for i, x in enumerate(pool.imap(generate_boxes_image, [{"data": d, "args": args} for d in data])):
        # if i % 1000 == 0:
        print(f"{i}/{len(data)}", flush=True)

    # print(pack({**data[0]}))
    # print(len(data))

    return 0


if __name__ == "__main__":
    sys.exit(main())