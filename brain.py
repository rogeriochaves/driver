import json
import re
from typing import List

from openai import OpenAI
from openai.types.chat import (
    ChatCompletionMessageParam,
)

from typings import Action, Context

client = OpenAI()


def initial_task_outline(input: str, image: str):
    print("\n\n> Looking at the screen to plan task\n")

    response = client.chat.completions.create(
        model="gpt-4-vision-preview",
        messages=[
            {
                "role": "system",
                "content": """\
                You are an AI agent with capacity to see the user's screen, click buttons and type text.
                The user will ask you to do things, you will see the user screen, then you will first think
                in high level what are the steps you will need to follow to carry the task. Then, you will
                be given annotated screenshots with codes mapping buttons and texts on the screen, which you
                can choose to click and proceed, type in an input field, and get a refreshed update of the
                screen to continue until the task is completed.""",
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"""\
                    {input}

                    Here is a screenshot of the screen, please do two things:

                    A. High level list of steps to follow, using a numbered list in english text
                    B. A list of actions to execute, being one of [CLICK A1] to click A1 button for example, \
                    [TYPE "I love you"] to type "I love you" in the input field, and [REFRESH] to get a new screenshot of the screen. \
                    DO NOT try to click on a button that is not visible in the screen yet, ask for a refresh first. \
                    If you need any [REFRESH], stop the action list there, and wait for the new screenshot to continue.
                    """,
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image,
                            "detail": "high",
                        },
                    },
                ],
            },
        ],
        stream=True,
        max_tokens=300,
    )

    content = ""
    for chunk in response:
        if delta := chunk.choices[0].delta.content:
            print(delta, end="", flush=True)
            content += delta

    return content


def plan_next_step_actions(context: Context, image: str):
    print("\n\n> Looking at the screen to plan next steps\n\nAnalyzing...")

    initial_user_prompt = f"""\
                    {context['task']}

                    Here is a screenshot of the screen, please do two things:

                    A. High level list of steps to follow, using a numbered list in english text
                    B. A list of actions to execute, being one of [CLICK A1] to click A1 button for example, \
                    [TYPE "I love you"] to type "I love you" in the input field, and [REFRESH] to get a new screenshot of the screen. \
                    DO NOT try to click on a button that is not visible in the screen yet, ask for a refresh first. \
                    If you need any [REFRESH], stop the action list there, and wait for the new screenshot to continue.
                    """

    next_step_user_prompt = f"""\
    Alright, I have executed the previous actions, let me share you the updated screenshot, so you can plan the next actions.
    As a reminder my goal is: {context['task']}.

    Please create a list with the next actions to take (options are [CLICK <CODE>], [TYPE <TEXT>] or [REFRESH])
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
                        "url": image,
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
                screen to continue until the task is completed.""",
        },
    ]

    response = client.chat.completions.create(
        model="gpt-4-vision-preview",
        messages=system_message + history + user_message,
        stream=True,
        max_tokens=300,
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
    print("\n\n> Extracting actions\n")

    response = client.chat.completions.create(
        model="gpt-4-1106-preview",
        messages=[
            {
                "role": "system",
                "content": """\
                You are helping extracing a structured text from another's bot unstructured output.
                That bot is responsible for taking a user task, then seeing the user screen,
                comming up with a list of discrete actions to execute.
                You are responsible for extracting the list of actions to a json we can use""",
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
        ],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "actions_extraction",
                    "description": "Extracts the actions",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "actions": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "action": {
                                            "type": "string",
                                            "enum": ["CLICK", "TYPE", "REFRESH"],
                                            "description": "The action to execute, being one of [CLICK A1] to click A1 button for example, [TYPE 'I love you'] to type 'I love you' in the input field, and [REFRESH] to get a new screenshot of the screen.",
                                        },
                                        "parameter": {
                                            "type": "string",
                                            "description": "The parameter of the action, being the button name, the text to type, or null for [REFRESH]",
                                        },
                                    },
                                    "required": ["action"],
                                },
                            },
                        },
                        "required": ["actions"],
                    },
                },
            }
        ],
        tool_choice={
            "type": "function",
            "function": {"name": "actions_extraction"},
        },
    )

    if tool_calls := response.choices[0].message.tool_calls:
        result: List[Action] = json.loads(tool_calls[0].function.arguments)["actions"]
        print(result)
        return result
    else:
        return None
