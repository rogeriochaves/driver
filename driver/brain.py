import json
import re
from typing import List, cast

from openai import OpenAI
from openai.types.chat import (
    ChatCompletionMessageParam,
)
from driver.cost import log_cost
from driver.logger import print_action

from driver.types import Action, Click, Context, Press, Refresh, Type
from driver.utils import image_to_base64

client = OpenAI()


def plan_next_step_actions(context: Context, image_path: str):
    print_action("Looking at the screen to plan next steps")
    print("Analyzing...")

    initial_user_prompt = f"""\
                    Task: {context['task']}

                    Here is a screenshot of the screen, tagged with labels like A1, A2, A3 on each interactive item, please do two things:

                    A. High level list of steps to follow, using a numbered list in english text

                    B. A list of actions to execute, being one of [CLICK A1] to click the A1 button for example, \
                    [TYPE "message"] to type "message", [PRESS ENTER] to \
                    press keys ENTER or shortcuts like CMD+F if needed, and [REFRESH] to end the list and get a new screenshot of the screen. \
                    Those are the ONLY options you have, work with that. If you need to switch apps,
                    use [PRESS CMD+SPACE] to open the spotlight and then [REFRESH]. \
                    If you want to click or type on an element that is not on the screen, issue a [REFRESH] first. \
                    """

    next_step_user_prompt = f"""\
    Alright, I have executed the previous actions, let me share you the updated screenshot, so you can plan the next actions.
    Describe what you are seeing, and describe where it might have gone wrong, because usually the screen changes and we have to course correct.
    As a reminder my goal is: {context['task']}.

    Please create a list with the next actions to take if any (options are [CLICK <LABEL>], [TYPE "<TEXT>"], [SHORTCUT <shortcut>] or [REFRESH])
    """

    user_prompt = (
        initial_user_prompt if len(context["history"]) == 0 else next_step_user_prompt
    )

    user_message: List[ChatCompletionMessageParam] = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": user_prompt,
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": image_to_base64(image_path),
                        "detail": "high",
                    },
                },
            ],
        }
    ]

    history = context["history"]

    system_message: List[ChatCompletionMessageParam] = [
        {
            "role": "system",
            "content": """\
                You are an AI agent with capacity to see the user's screen, click buttons and type text.
                The user will ask you to do things, you will see the user screen, then you will first think
                in high level what are the steps you will need to follow to carry the task. Then, you will
                be given annotated screenshots with codes mapping buttons and texts on the screen, which you
                can choose to click and proceed, type in an input field, and get a refreshed update of the
                screen to continue until the task is completed. You are always very short and concise in your writing.""",
        },
    ]

    model = "gpt-4-vision-preview"
    messages = system_message + history + user_message

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        stream=True,
        max_tokens=600,
    )

    content = ""
    for chunk in response:
        if delta := chunk.choices[0].delta.content:
            print(delta, end="", flush=True)
            content += delta

    context["history"].append(
        {
            "role": "user",
            "content": user_prompt,
        }
    )
    context["history"].append(
        {
            "role": "assistant",
            "content": content,
        }
    )

    log_cost(
        model=model,
        messages=system_message + history,
        completion=content,
        image={
            "text": user_prompt,
            "path": image_path,
            "detail": "high",
        },
    )

    return content


def extract_high_level_plan_and_actions(input: str):
    pattern = r"^A\. High.?level([\s\S]*?)^B\.\s*([\s\S]*)"

    match = re.search(pattern, input, re.DOTALL | re.MULTILINE | re.IGNORECASE)

    if match:
        between_tokens = match.group(1).strip()
        after_token_b = match.group(2).strip()

        return str(between_tokens), str(after_token_b)
    else:
        return None, None


def extract_structured_actions(input: str):
    actions = heuristics_extract_structured_actions(input)
    if not actions:
        actions = llm_structured_actions(input)
    return actions


def heuristics_extract_structured_actions(input: str):
    actions_list = []

    # Split the input string by lines and iterate through them
    for line in input.split("\n"):
        # Match the pattern [ACTION "arguments" additional_arguments] ignoring anything outside brackets
        match = re.search(r"\[(\w+)(?:\s+\"?([^\"]+?)\"?)?(?:\s+(\w+))?\]", line)
        if match:
            action_type = match.group(1)
            arguments = match.group(2)
            additional_argument = match.group(3)

            if action_type == "CLICK":
                actions_list.append(Click(action="CLICK", label=arguments))
            elif action_type == "TYPE":
                actions_list.append(
                    Type(action="TYPE", text=arguments, label=additional_argument)
                )
            elif action_type == "PRESS":
                # Handle the modifiers and keys separately
                parts = arguments.split("+") if arguments else []
                modifier = parts[0] if len(parts) > 1 else None
                second_modifier = parts[1] if len(parts) > 2 else None
                key = parts[-1] if parts else None
                actions_list.append(
                    Press(
                        action="PRESS",
                        modifier=modifier,
                        second_modifier=second_modifier,
                        key=key or "",
                    )
                )
            elif action_type == "REFRESH":
                actions_list.append(Refresh(action="REFRESH"))
            else:
                # Unknown action type
                return None

    return actions_list if actions_list else None


def llm_structured_actions(input: str):
    print_action("Extracting actions")

    model = "gpt-4-1106-preview"
    messages = [
        {
            "role": "system",
            "content": """\
                You are helping extracing a structured text from another's bot unstructured output.
                That bot is responsible for taking a user task, then seeing the user screen,
                comming up with a list of discrete actions to execute.
                You are responsible for extracting the list of actions to a json we can use by using a sequence of tool calls""",
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": f"""\
                    Here is the output of the previous bot, please extract the list of actions:

                    {input}
                    """,
                },
            ],
        },
    ]

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "CLICK",
                    "description": "Clicks on an item on the screen",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "label": {
                                "type": "string",
                                "description": "The label of the item to click, in the format like A1, A2, A3, etc",
                            },
                        },
                        "required": ["label"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "TYPE",
                    "description": "Type a text on an input field on the screen",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "text": {
                                "type": "string",
                                "description": "The label of the item to click",
                            },
                        },
                        "required": ["text"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "PRESS",
                    "description": "Press a key or executes a shortcut on the screen",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "modifier": {
                                "type": "string",
                                "description": "The modifier key to press, options are CMD, CTRL, ALT or SHIFT",
                                "enum": ["CMD", "CTRL", "ALT", "SHIFT"],
                                "optional": True,
                            },
                            "second_modifier": {
                                "type": "string",
                                "description": "Optional second modifier key to press, options are CMD, CTRL, ALT or SHIFT",
                                "enum": ["CMD", "CTRL", "ALT", "SHIFT"],
                                "optional": True,
                            },
                            "key": {
                                "type": "string",
                                "description": "The key to press, in uppercase, such as any letter, number, symbols, F1-F12, ENTER, SPACE or ESC",
                            },
                        },
                        "required": ["key"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "REFRESH",
                    "description": "Type a text on an input field on the screen",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                    },
                },
            },
        ],
    )

    actions: List[Action] = []
    try:
        content = response.choices[0].message.content or ""
        log_cost(model=model, messages=messages, completion=content)
        tool_uses = json.loads(content.replace("```json", "").replace("```", ""))
        for tool_use in tool_uses:
            action = cast(Action, {"action": tool_use["recipient_name"].split(".")[-1]})
            action.update(tool_use["parameters"])
            actions.append(action)
        return actions
    except Exception as e:
        return None
