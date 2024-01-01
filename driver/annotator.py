import math
import os
from PIL import Image, ImageDraw, ImageFont
import cv2
from driver.UIED.run_single import detect_components
from driver.UIED.utils import show_image
from driver.ocr_call import ocr_text_detection


from driver.types import DebugConfig, ImgMultiplierFactor, LabelMap
from driver.utils import is_retina_display


def annotate_image(input_image_path, debug: DebugConfig):
    ocr_result = ocr_text_detection(input_image_path, debug)

    components = detect_components(
        input_image_path,
        ocr_result,
        showOCR=debug["ocr"],
        showUIED=debug["uied"],
    )

    original_image = Image.open(input_image_path)
    size = {"width": original_image.width, "height": original_image.height}
    img_multiplier_factor: ImgMultiplierFactor = {
        "height": components["img_shape"][0] / size["height"],
        "width": components["img_shape"][1] / size["width"],
    }

    label_counter = 1
    label_prefix = "A"
    drawn_positions = []
    label_map: LabelMap = {}

    label_width = 48 if is_retina_display() else 24
    label_height = 24 if is_retina_display() else 12

    # Most likely the biggest components are the most important ones to be clicked on the screen,
    # and seems like GPT-V will be biased anyway towards choosing A1, A2, etc the early labels,
    # so we try to play into that and label from the biggest to the smallest components
    sorted_components = sorted(
        components["compos"],
        key=lambda x: (x.row_max - x.row_min) * (x.col_max - x.col_min),
        reverse=True
    )

    for component in sorted_components:
        if component.text_content and len(component.text_content) < 2:
            continue

        component_position = {
            "x": round(component.col_min / img_multiplier_factor["width"]),
            "y": round(component.row_min / img_multiplier_factor["height"]),
            "x2": round(component.col_max / img_multiplier_factor["width"]),
            "y2": round(component.row_max / img_multiplier_factor["height"]),
        }
        component_width = component_position["x2"] - component_position["x"]
        component_height = component_position["y2"] - component_position["y"]

        if component_height < label_height:
            continue

        label_position = (
            round(component_position["x"] - label_width / 2),
            round(component_position["y"] - label_height / 2),
        )

        # Draw label in the center of the component for big components
        big_component = 200 if is_retina_display() else 100
        if component_width > big_component and component_height > big_component:
            label_position = (
                round(component_position["x"] + component_width / 2 - label_width),
                round(component_position["y"] + component_height / 2 - label_height),
            )

        too_close = any(
            abs(label_position[0] - pos[0]) < label_width
            and abs(label_position[1] - pos[1]) < label_height * 2
            for pos in drawn_positions
        )
        if too_close:
            continue

        if label_counter > 9:
            label_counter = 1
            next_char = chr(ord(label_prefix[-1]) + 1)
            if next_char == "I":
                next_char = "J"  # Skip 'I' to avoid confusion with 'l'
            if label_prefix[-1] == "Z":
                label_prefix += "A"
            else:
                label_prefix = label_prefix[:-1] + next_char
        label = f"{label_prefix}{label_counter}"
        draw_square(
            original_image,
            label_position,
            label,
            width=label_width,
            height=label_height,
        )
        drawn_positions.append(label_position)
        label_map[label] = {
            "text": component.text_content or "",
            "position": (
                component_position["x"],
                component_position["y"],
            ),
            "size": (
                component_width,
                component_height,
            ),
        }
        label_counter += 1

    os.makedirs("./output/annotated", exist_ok=True)
    output_image_path = f"./output/annotated/{os.path.basename(input_image_path)}"
    original_image.save(output_image_path)

    print(f"{len(label_map.keys())} elements found on the screen", end="")
    if debug["annotations"]:
        show_image("Annotated", cv2.imread(output_image_path))

    return label_map, output_image_path, img_multiplier_factor


def draw_square(
    image,
    position,
    code,
    width,
    height,
    fill_color_start="#EFDD88",
    fill_color_end="#EBD872",
    outline_color="#EBD872",
):
    draw = ImageDraw.Draw(image)
    font_size = round(height - 1)
    try:
        font = ImageFont.truetype("arialbd.ttf", font_size)
    except Exception:
        font = ImageFont.truetype("/Library/Fonts/Arial Bold.ttf", font_size)
    x, y = position
    x = max(0, x)
    y = max(0, y)
    x1, y1 = x + width, y + height

    # Create a gradient
    gradient = Image.new("RGB", (1, height), color=fill_color_start)
    for i in range(height):
        gradient.putpixel(
            (0, i),
            tuple(
                int(fill_color_start[j : j + 2], 16)
                + int(
                    (
                        int(fill_color_end[j : j + 2], 16)
                        - int(fill_color_start[j : j + 2], 16)
                    )
                    * (i / height)
                )
                for j in (1, 3, 5)
            ),  # type: ignore
        )

    # Apply gradient
    for i in range(x, x1):
        image.paste(gradient, (i, y, i + 1, y1))

    draw.rounded_rectangle((x, y, x1, y1), radius=5, outline=outline_color, width=2)
    _, _, w, h = draw.textbbox(xy=(0, 0), text=code, font=font)  # type: ignore
    draw.text(
        (x + (width - w) / 2, y - 1 + (height - h) / 2),
        code,
        fill="black",
        font=font,
    )
