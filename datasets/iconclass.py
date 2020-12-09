import argparse

from datasets.pipeline import MsgPackPipeline
from datasets.datasets import DatasetsManager


@DatasetsManager.export("iconclass")
class IconClassDataloader:
    def __init__(self):
        pass

    def train(self):
        pass

    def val(self):
        pass

    def test(self):
        pass

    @classmethod
    def add_args(cls, parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--train_path", type=int, default=12)
        parser.add_argument("--val_path", type=str, default="/some/path")
        return parser
