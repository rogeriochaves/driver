import atexit
from typing import Any, List, Literal, TypedDict, Union
import litellm
from colorama import Fore, Style
from openai.types.chat import (
    ChatCompletionMessageParam,
)
from PIL import Image

total_cost: float = 0.0


class ImageMessage(TypedDict):
    text: str
    path: str
    detail: Literal["high", "low"]


def log_cost(
    model: str,
    messages: List[ChatCompletionMessageParam],
    completion: str,
    image: Union[ImageMessage, None] = None,
):
    global total_cost

    if image:
        messages.append({"role": "user", "content": image["text"]})

    cost = litellm.completion_cost(
        model=model, messages=messages, completion=completion
    )

    if image:
        img = Image.open(image["path"])
        width, height = img.size
        img.close()

        tokens = calculate_token_cost(width, height, image["detail"])
        (image_cost, _) = litellm.cost_per_token(model=model, prompt_tokens=tokens)
        cost += image_cost

    print(Fore.CYAN + f"\n\nCost: ${cost:.5f}" + Style.RESET_ALL)
    total_cost += cost


def print_total_cost():
    print(Fore.CYAN + f"\n\nTotal session cost: ${total_cost:.5f}" + Style.RESET_ALL)


# trap python to print total cost before exiting
atexit.register(print_total_cost)


# Token calculation from: https://platform.openai.com/docs/guides/vision/calculating-costs
def calculate_token_cost(width, height, detail):
    if detail == "low":
        return 85
    else:
        # Scale the image to fit within 2048 x 2048 while maintaining aspect ratio
        max_dim = max(width, height)
        if max_dim > 2048:
            scale_factor = 2048 / max_dim
            width, height = int(width * scale_factor), int(height * scale_factor)

        # Scale such that the shortest side is 768px long
        min_dim = min(width, height)
        scale_factor = 768 / min_dim
        width, height = int(width * scale_factor), int(height * scale_factor)

        # Calculate the number of 512px squares needed
        num_tiles_wide = -(-width // 512)  # Ceiling division
        num_tiles_high = -(-height // 512)  # Ceiling division
        num_tiles = num_tiles_wide * num_tiles_high

        # Each tile costs 170 tokens, plus an additional 85 tokens
        return 170 * num_tiles + 85
