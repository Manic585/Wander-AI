import re


def ensure_text(value) -> str:
    if isinstance(value, str):
        return value

    if isinstance(value, list):
        return "\n".join(ensure_text(item) for item in value)

    if isinstance(value, dict):
        text = value.get("text") or value.get("content")
        if text is not None:
            return ensure_text(text)
        return "\n".join(f"{key}: {ensure_text(val)}" for key, val in value.items())

    if value is None:
        return ""

    return str(value)


def strip_thinking(value: str) -> str:
    """Remove reasoning blocks, including an unfinished block at the end."""
    text = re.sub(r"&lt;(/?think\b.*?)&gt;", r"<\1>", ensure_text(value), flags=re.I)
    text = re.sub(r"<think\b[^>]*>[\s\S]*?(?:</think\s*>|$)", "", text, flags=re.I)
    return re.sub(r"</think\s*>", "", text, flags=re.I).strip()


def visible_model_response(response) -> str:
    text = strip_thinking(response.content)
    if not text:
        return "No travel plan was returned. Please generate the plan again."
    if response.response_metadata.get("finish_reason") == "length":
        text += (
            "\n\n> This response reached the model's output limit and is incomplete. "
            "Please generate a plan for fewer days or request fewer details."
        )
    return text
