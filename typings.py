from typing import Dict, List, Literal, TypedDict, Union
from openai.types.chat import (
    ChatCompletionAssistantMessageParam,
    ChatCompletionUserMessageParam,
)


class Action(TypedDict):
    action: Literal["CLICK", "TYPE", "REFRESH"]
    parameter: str


class LabelMapItem(TypedDict):
    text: str
    position: tuple[int, int]


LabelMap = Dict[str, LabelMapItem]


class Context(TypedDict):
    task: str
    high_level_plan: str
    history: List[
        Union[ChatCompletionAssistantMessageParam, ChatCompletionUserMessageParam]
    ]
    actions_history: List[List[Action]]
