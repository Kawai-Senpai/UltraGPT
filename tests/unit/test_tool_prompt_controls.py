from langchain_core.messages import SystemMessage
from pydantic import BaseModel

from ultragpt import UltraGPT


class _Params(BaseModel):
    value: int


TOOL = {
    "name": "sample_tool",
    "description": "Perform the sample action.",
    "parameters_schema": _Params,
    "usage_guide": "Provide an integer value.",
    "when_to_use": "When a sample action is required.",
}


class _ProviderManager:
    @staticmethod
    def does_support_thinking(_model):
        return False


class _ChatFlow:
    def __init__(self):
        self.messages = None

    def chat_with_model_tools(self, messages, _tools, **_kwargs):
        self.messages = messages
        return {"content": "ok"}, 1, {}


def _client():
    client = UltraGPT(openrouter_api_key="test")
    client.provider_manager = _ProviderManager()
    client.chat_flow = _ChatFlow()
    return client


def test_tool_call_can_omit_generated_tool_system_prompt():
    client = _client()
    response, _, details = client.tool_call(
        messages=[
            {"role": "system", "content": "STABLE COMMON PROMPT"},
            {"role": "user", "content": "Use the sample tool"},
        ],
        user_tools=[TOOL],
        include_tool_prompt=False,
        tools=[],
    )

    system_messages = [
        message.content for message in client.chat_flow.messages
        if isinstance(message, SystemMessage)
    ]
    assert response == {"content": "ok"}
    assert details["tool_prompt_included"] is False
    assert system_messages == ["STABLE COMMON PROMPT"]


def test_tool_call_keeps_existing_rich_prompt_by_default():
    client = _client()
    _, _, details = client.tool_call(
        messages=[{"role": "system", "content": "BASE"}],
        user_tools=[TOOL],
        tools=[],
    )

    system_messages = [
        message.content for message in client.chat_flow.messages
        if isinstance(message, SystemMessage)
    ]
    assert details["tool_prompt_included"] is True
    assert system_messages[0].startswith("Available Tools:")
    assert "Usage Guide: Provide an integer value." in system_messages[0]
    assert system_messages[1] == "BASE"
