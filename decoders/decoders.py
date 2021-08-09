import argparse
import logging


class DecodersManager:
    _decoders = {}

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @classmethod
    def export(cls, name):
        def export_helper(plugin):
            cls._decoders[name] = plugin
            return plugin

        return export_helper

    @classmethod
    def add_args(cls, parent_parser):
        logging.info("Add DecodersManager args")
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False, conflict_handler="resolve")

        parser.add_argument("-d", "--decoder", help="Decoder that should be trained")
        args, _ = parser.parse_known_args()

        for decoder, c in cls._decoders.items():
            if decoder != args.decoder:
                continue
            add_args = getattr(c, "add_args", None)
            if callable(add_args):
                parser = add_args(parser)
        return parser

    def list_decoders(self):
        return self._decoders

    def build_decoder(self, name, **kwargs):

        assert name in self._decoders, f"Decoder {name} is unknown"
        return self._decoders[name](**kwargs)
