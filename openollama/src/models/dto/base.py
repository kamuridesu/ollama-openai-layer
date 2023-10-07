import json
from typing import Callable, Any, Unpack, TypedDict


class FieldParser(TypedDict):
    property: str
    function: Callable[[dict[str, Any]], Any]


class Base:
    @classmethod
    def parse(cls, data: dict, **kwargs: Unpack[FieldParser]):
        if data is None:
            return
        _kwargs = {}
        for _field in cls.__dict__.get("__match_args__"):
            value = data.get(_field)
            if (func := kwargs.get(_field)) is not None and value is not None:
                value = func(value)
            _kwargs[_field] = value
        return cls(**_kwargs)

    def to_json(self):
        return json.dumps(self, default=lambda o: o.__dict__, sort_keys=True)

    def to_dict(self):
        return json.loads(self.to_json())
