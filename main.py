# import os
# from ocr_draw import annotate_image_with_ocr

# input_image_path = "./screenshot.png"
# output_image_path = annotate_image_with_ocr(input_image_path)

# # Display the image (this will vary depending on the OS)
# os.system(
#     f"open {output_image_path}" if os.name == "posix" else f"start {output_image_path}"
# )

import base64
import json
import mimetypes
import re
from openai import OpenAI

from ocr_draw import annotate_image_with_ocr

client = OpenAI()


def image_to_base64(file_path):
    mime_type, _ = mimetypes.guess_type(file_path)
    with open(file_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
    return f"data:{mime_type};base64,{encoded_string}"


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


def extract_high_level_plan_and_actions(input: str):
    pattern = r"^A\. High.?level([\s\S]*?)^B\.\s*([\s\S]*)"

    match = re.search(pattern, input, re.DOTALL | re.MULTILINE | re.IGNORECASE)

    if match:
        between_tokens = match.group(1).strip()
        after_token_b = match.group(2).strip()

        return between_tokens, after_token_b
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
        result = json.loads(tool_calls[0].function.arguments)
        print(result)
        return result
    else:
        return None


task = "Hey, can you write an email to lauraestevesps@gmail.com saying that I love her?"
input_image_path = "./screenshot.png"

output_image_path = annotate_image_with_ocr(input_image_path)

plan_and_actions = initial_task_outline(
    input=task,
    image=image_to_base64(output_image_path),
)

plan_and_actions = """\
A. High level list of steps to follow:

1. Click the "Compose" button to open a new email composition window.
2. Enter the recipient's email address in the "To" field.
3. Enter a subject for the email.
4. Type the body of the email expressing your love.
5. Click the "Send" button to send the email.

B. List of actions to execute:

1. [CLICK A9] - Click on "Compose" to start a new email.
2. [REFRESH] - Wait for a new screenshot after clicking "Compose" to see the email composition window.
"""

high_level_plan, actions = extract_high_level_plan_and_actions(plan_and_actions)

if actions:
    actions = extract_structured_actions(input=actions)

    print("\n\nactions\n\n", actions)
else:
    print("\n\n> No actions extracted, debug plan_and_actions:\n", plan_and_actions)
