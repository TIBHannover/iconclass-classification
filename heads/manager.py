import argparse
import logging


class HeadsManager:
    _heads = {}

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @classmethod
    def export(cls, name):
        def export_helper(plugin):
            cls._heads[name] = plugin
            return plugin

        return export_helper

    @classmethod
    def add_args(cls, parent_parser):

        logging.info("Add HeadsManager args")
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False, conflict_handler="resolve")

        parser.add_argument("--heads", nargs="+", help="Heads that should be trained")
        args, _ = parser.parse_known_args()

        for head, c in cls._heads.items():
            if head not in args.heads:
                continue
            add_args = getattr(c, "add_args", None)
            if callable(add_args):
                parser = add_args(parser)
        return parser

    def list_heads(self):
        return self._heads

    def build_heads(self, names, **kwargs):
        heads = []
        for name in names:
            assert name in self._heads, f"Head {name} is unknown"
            heads.append(self._heads[name](**kwargs))
        
        return heads


