import os
import sys
import re
import argparse

from hierarchical.tools.utils import read_jsonl

def parse_args():
    parser = argparse.ArgumentParser(description='')

    parser.add_argument('-v', '--verbose', action='store_true', help='verbose output')    
    parser.add_argument('-a', '--annotation_path', help='verbose output')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    yolo_count = 0
    flat_count = 0
    unique = 0
    data = read_jsonl(args.annotation_path)
    for line in data:
        yolo_count+= line["count"]["yolo"]
        flat_count+= line["count"]["flat"]
        if int(line["count"]["flat"]) >= 1:
            unique+= 1


    print(yolo_count)
    print(flat_count)
    print(unique)
    return 0

if __name__ == '__main__':
    sys.exit(main())