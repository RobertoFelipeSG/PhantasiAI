from __future__ import annotations
from typing import Any, List, Type, Union, get_args, get_origin
import json
from pydantic import BaseModel


def _unwrap_optional(t: Any) -> tuple[Any, bool]:
    """
    Return (inner_type, is_optional) where `is_optional` is True when
    the original annotation allowed `None`.
    """
    if get_origin(t) is Union:
        args = get_args(t)
        non_none = [a for a in args if a is not type(None)]
        if len(non_none) == 1:
            return _unwrap_optional(non_none[0])[0], True
    return t, False


def _type_label(t: Any, optional: bool) -> str:
    """
    Produce a string label for primitive types.  Adds '|null' if optional.
    """
    base = {
        str: "str",
        bytes: "bytes",
        int: "int",
        float: "float",
        bool: "bool",
    }.get(t, "any")
    return f"{base}|null" if optional else base


class PromptOutputFormatter:
    """
    Produce a JSON skeleton where each leaf is a *string describing the
    expected type*, e.g. `"str|null"`, `"int"`, etc.
    """

    def __init__(self, model_cls: Type[BaseModel]):
        self.model_cls = model_cls

    def _build(self, ann: Any) -> Any:
        inner, opt = _unwrap_optional(ann)
        origin = get_origin(inner)

        if origin in (list, List):
            return [self._build(get_args(inner)[0])]

        if hasattr(inner, "model_fields"):
            return {n: self._build(f.annotation) for n, f in inner.model_fields.items()}

        return _type_label(inner, opt)

    def generate_json_structure(self) -> str:
        skeleton = self._build(self.model_cls)
        json_block = json.dumps(skeleton, indent=2, ensure_ascii=False)
        return f"```json\n{json_block}\n```"
