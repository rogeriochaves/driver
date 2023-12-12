from typing import Dict, List, Literal, Optional, TypedDict, Union
from openai.types.chat import (
    ChatCompletionAssistantMessageParam,
    ChatCompletionUserMessageParam,
)


class Click(TypedDict):
    action: Literal["CLICK"]
    label: str


class Type(TypedDict):
    action: Literal["TYPE"]
    text: str
    label: Optional[str]


class Press(TypedDict):
    action: Literal["PRESS"]
    modifier: Optional[str]
    second_modifier: Optional[str]
    key: str


class Refresh(TypedDict):
    action: Literal["REFRESH"]


Action = Union[Click, Type, Press, Refresh]


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
