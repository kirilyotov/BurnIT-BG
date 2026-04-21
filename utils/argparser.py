"""Helpers for constructing dataclass instances from command-line arguments."""

import argparse
from dataclasses import MISSING
from typing import TypeVar, Type

Dataclass = TypeVar('Dataclass')


def init_dataclass_from_args(dataclass_type: Type[Dataclass]) -> Dataclass:
    """Instantiate a dataclass by mapping parsed CLI arguments to dataclass fields."""

    parser = argparse.ArgumentParser(
        description=f'Initialize {dataclass_type.__name__} from command line arguments.'
        )

    for field in dataclass_type.__dataclass_fields__.values():
        has_default = field.default is not MISSING or field.default_factory is not MISSING
        parser.add_argument(
            f'--{field.name.replace("_", "-")}',
            type=field.type,
            required=not has_default,
            default=None if not has_default else field.default,
            dest=field.name,
            help=f'{field.name} of type {field.type}'
        )

    args, _ = parser.parse_known_args()
    args_dict = {
        f: getattr(args, f)
        for f in dataclass_type.__dataclass_fields__
    }

    return dataclass_type(**args_dict)