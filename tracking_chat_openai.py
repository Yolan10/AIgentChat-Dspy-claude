from __future__ import annotations

try:  # pragma: no cover - fallback only used in misconfigured envs
    from langchain_openai import ChatOpenAI
except ModuleNotFoundError:  # noqa: D207 - keep short message
    try:  # pragma: no cover - only executed when fallback needed
        # Older langchain versions bundle the OpenAI integration
        from langchain.chat_models import ChatOpenAI
    except Exception as exc:  # pragma: no cover - manual error message
        raise ModuleNotFoundError(
            "Missing 'langchain_openai'. Install it with 'pip install langchain-openai'"
        ) from exc



try:  # pragma: no cover - optional dependency
    from openai import OpenAIError  # type: ignore
except Exception:  # pragma: no cover - when openai is missing
    try:  # pragma: no cover - support older openai versions
        from openai.error import OpenAIError  # type: ignore
    except Exception:  # pragma: no cover - fallback when not installed
        OpenAIError = Exception


class TrackingChatOpenAI(ChatOpenAI):
    """ChatOpenAI subclass that records token usage."""

    def __init__(self, *args, token_tracker_instance=None, **kwargs):
        super().__init__(*args, **kwargs)
        if token_tracker_instance is None:
            from token_tracker import token_tracker as default_tracker
            object.__setattr__(self, "token_tracker", default_tracker)
        else:
            object.__setattr__(self, "token_tracker", token_tracker_instance)

    def invoke(self, *args, **kwargs):  # type: ignore[override]
        try:
            message = super().invoke(*args, **kwargs)
        except OpenAIError as exc:  # pragma: no cover - network failure fallback
            print(f"LLM request failed: {exc}. Using fallback message.")

            class _Dummy:
                content = "I am unable to respond right now."
                additional_kwargs = {}

            return _Dummy()

        usage = None
        try:
            usage = message.additional_kwargs.get("token_usage")
        except Exception:
            usage = None
        if usage:
            self.token_tracker.add_usage(
                model=self.model_name,
                prompt_tokens=usage.get("prompt_tokens", 0),
                completion_tokens=usage.get("completion_tokens", 0),
            )
        return message
