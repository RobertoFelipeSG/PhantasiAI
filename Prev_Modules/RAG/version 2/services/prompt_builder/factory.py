from __future__ import annotations
from typing import Any, Dict, List, Optional, Type, Union, get_args, get_origin


def _inner_type(t: Any) -> Any:
    """Unwrap Optional / Union[..., None] shells (same helper as above)."""
    origin = get_origin(t)
    if origin is Union:
        non_none = [a for a in get_args(t) if a is not type(None)]
        return _inner_type(non_none[0]) if non_none else t
    return t

def _unwrap_list(t):
    """Return (inner_type, is_list)."""
    if get_origin(t) in (list, List):
        return get_args(t)[0], True
    return t, False


class PromptSnippets: 
    intro = (
        "You are a document analysis expert. I will provide you with a document.\n"
        "Your task is to extract specific information from the document and return it in JSON format.\n"
    )
    extract_instruction = "Please extract the following information:\n"
    model_structure_intro = "\nImportant: The response must follow these model structures:\n"
    output_requirements = (
        "\nReturn the results in valid JSON format with exactly these field names and structures.\n"
        "If a field is not found, use null as the value.\n"
        "Only return the JSON object, no other text."
    )


class Factory:
    """
    Builds an extraction prompt from a Pydantic model.  Now aware of Optional/
    Union wrappers, so nested descriptions are rendered correctly.
    """

    def __init__(self, model_cls: Type[Any], snippets: PromptSnippets = PromptSnippets):
        self.model_cls = model_cls
        self.snippets = snippets

    def _describe_field(self, name: str, field) -> str:
        desc = getattr(field, "description", None)
        return f"- {name}: {desc}" if desc else ""

    def _collect_nested_lines(self, t: Any) -> List[str]:
        """
        Return description lines for nested sub-fields (including inside List[…]).
        """
        lines: List[str] = []
        t = _inner_type(t)
        t, _ = _unwrap_list(t)

        origin = get_origin(t)
        if origin in (list, List):
            t = _inner_type(get_args(t)[0])

        if hasattr(t, "model_fields"):
            lines.append(f"\n{t.__name__} structure:")
            for n, sub in t.model_fields.items():
                if sub.description:
                    lines.append(f"- {n}: {sub.description}")
        return lines

    def assemble_prompt(self, keys: Optional[List[str]] = None) -> str:
        keys = keys or list(self.model_cls.model_fields)
        field_lines: List[str] = []
        nested_lines: List[str] = []

        for key in keys:
            field = self.model_cls.model_fields.get(key)
            if not field:
                continue
            field_lines.append(self._describe_field(key, field))
            nested_lines.extend(self._collect_nested_lines(field.annotation))

        parts = [
            self.snippets.intro,
            self.snippets.extract_instruction,
            "\n".join(field_lines)
        ]
        if nested_lines:
            parts.extend([self.snippets.model_structure_intro, "\n".join(nested_lines)])
        parts.append(self.snippets.output_requirements)
        return "\n".join(parts)
