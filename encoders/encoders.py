import argparse
import logging


class EncodersManager:
    _encoders = {}

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @classmethod
    def export(cls, name):
        def export_helper(plugin):
            cls._encoders[name] = plugin
            return plugin

        return export_helper

    @classmethod
    def add_args(cls, parent_parser):

        logging.info("Add EncodersManager args")
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False, conflict_handler="resolve")

        parser.add_argument("-e", "--encoder", help="Encoder that should be trained")
        args, _ = parser.parse_known_args()

        for encoder, c in cls._encoders.items():
            if encoder != args.encoder:
                continue
            add_args = getattr(c, "add_args", None)
            if callable(add_args):
                parser = add_args(parser)
        return parser

    def list_encoders(self):
        return self._encoders

    def build_encoder(self, name, **kwargs):

        assert name in self._encoders, f"Encoder {name} is unknown"
        return self._encoders[name](**kwargs)
