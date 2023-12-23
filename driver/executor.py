import sys
import time
from typing import List

import pyautogui
import pyperclip
from driver.brain import (
    extract_high_level_plan_and_actions,
    extract_structured_actions,
    plan_next_step_actions,
)
from driver.logger import print_action
from driver.ocr_draw import annotate_image_with_ocr
from driver.typings import Action, LabelMap, Context, LabelMapItem
from colorama import Fore, Style

from driver.utils import is_retina_display


def take_screenshot():
    screenshot = pyautogui.screenshot()
    screenshot.save("screenshot.png")
    return "./screenshot.png"


def start(task: str):
    screenshot = take_screenshot()
    label_map, output_image_path = annotate_image_with_ocr(screenshot)

    context: Context = {
        "task": task,
        "history": [],
        "high_level_plan": "",
        "actions_history": [],
    }
    plan_and_actions = plan_next_step_actions(
        context=context,
        image_path=output_image_path,
    )
    if plan_and_actions:
        high_level_plan, str_actions = extract_high_level_plan_and_actions(
            plan_and_actions
        )
        context["high_level_plan"] = high_level_plan or ""
    else:
        raise Exception(f"No plan and actions were written: {plan_and_actions}")

    parse_actions_and_execute(
        label_map=label_map, context=context, str_actions=str_actions
    )


def next_step(context: Context):
    screenshot = take_screenshot()
    label_map, output_image_path = annotate_image_with_ocr(screenshot)
    str_actions = plan_next_step_actions(
        context=context,
        image_path=output_image_path,
    )
    parse_actions_and_execute(
        label_map=label_map, context=context, str_actions=str_actions
    )


def parse_actions_and_execute(
    context: Context, label_map: LabelMap, str_actions: str | None
):
    if str_actions:
        actions = extract_structured_actions(input=str_actions) or []
        context["actions_history"].append(actions)
    else:
        raise Exception(f"No actions found on the plan: {str_actions}")

    if len(actions) > 0:
        execute(context, label_map=label_map, actions=actions)
    else:
        print(
            Fore.GREEN
            + "\nNo actions found, assuming our job here is done! Exiting"
            + Style.RESET_ALL
        )
        sys.exit(0)


def execute(context: Context, label_map: LabelMap, actions: List[Action]):
    print_action("Executing actions")

    for action in actions:
        if action["action"] == "CLICK":
            if action["label"] not in label_map:
                print(
                    f"WARN: Label {action['label']} not present in the screenshot, skipping CLICK action"
                )
                continue
            item = label_map[action["label"]]
            print(f"Clicking {item}")
            click(item)
        elif action["action"] == "TYPE":
            if "label" in action and action["label"] in label_map:
                item = label_map[action["label"]]
                print(f"Clicking {item}")
                click(item)
            type(action["text"])
        elif action["action"] == "PRESS":
            modifier_map = {
                "CMD": "command",
                "CTRL": "ctrl",
                "ALT": "alt",
                "SHIFT": "shift",
            }

            if (
                "modifier" in action
                and action["modifier"]
                and "second_modifier" in action
                and action["second_modifier"]
            ):
                print(
                    f"Executing shortcut {action['modifier']}+{action['second_modifier']}+{action['key']}"
                )
                pyautogui.hotkey(
                    modifier_map[action["modifier"]],
                    modifier_map[action["second_modifier"]],
                    action["key"].lower(),
                    interval=0.1,
                )
            elif "modifier" in action and action["modifier"]:
                print(f"Executing shortcut {action['modifier']}+{action['key']}")
                pyautogui.hotkey(
                    modifier_map[action["modifier"]],
                    action["key"].lower(),
                    interval=0.1,
                )
            else:
                print(f"Pressing {action['key']}")
                pyautogui.press(action["key"].lower(), interval=0.1)
        elif action["action"] == "REFRESH":
            time.sleep(1)
            print("Refreshing screenshot")
            next_step(context)
            return
        else:
            print("Unknown action")
        time.sleep(0.2)  # little bit of sleep in between actions

    # Refresh by default if refresh was not issued
    time.sleep(2)
    print("Refreshing screenshot")
    next_step(context)


def click(item: LabelMapItem):
    division = 2 if is_retina_display() else 1
    pyautogui.moveTo(
        round(item["position"][0] + 12) / division,
        round(item["position"][1] + 12) / division,
        duration=1,
    )
    pyautogui.click()


def type(text):
    text = text.replace("\\n", "\n")
    print(f"Typing {text}")
    if contains_non_typeable_characters(text):
        pyperclip.copy(text)
        if sys.platform == "darwin":
            pyautogui.hotkey("command", "v", interval=0.1)
        else:
            pyautogui.hotkey("ctrl", "v", interval=0.1)
    else:
        pyautogui.write(text, interval=0.05)


def contains_non_typeable_characters(text):
    typeable_characters = set(
        "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890 ,.?!@#$%^&*()-_=+[]{}|;:'\"<>/\\`~"
    )

    for char in text:
        if char not in typeable_characters:
            return True
    return False
