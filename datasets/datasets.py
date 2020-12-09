import argparse


class DatasetsManager:
    _datasets = {}

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @classmethod
    def export(cls, name):
        def export_helper(plugin):
            cls._datasets[name] = plugin
            return plugin

        return export_helper

    @classmethod
    def add_args(cls, parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)

        parser.add_argument("-d", "--dataset", help="Dasaset to be loaded")
        for dataset, c in cls._datasets.items():
            add_args = getattr(c, "add_args", None)
            if callable(add_args):
                parser = add_args(parser)
        return parser

    def list_datasets(self):
        return self._datasets

    def build_dataset(self, name, **kwargs):

        assert name in self._datasets, f"Dataset {name} is unknown"
        return self._datasets[name](**kwargs)
