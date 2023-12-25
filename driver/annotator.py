import math
import os
from PIL import Image, ImageDraw, ImageFont
import cv2
from driver.UIED.run_single import detect_components
from driver.logger import print_action
from driver.ocr_call import ocr_text_detection


from driver.types import ImgMultiplierFactor, LabelMap
from driver.utils import is_retina_display


def annotate_image(input_image_path, show=False):
    ocr_result = ocr_text_detection(input_image_path)

    max_height = 982
    components = detect_components(
        input_image_path, ocr_result, max_height=max_height, show=False
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

    sorted_components = sorted(
        components["compos"],
        key=lambda x: (math.ceil(x.row_min / (label_height * 2)), x.col_min),
    )
    for component in sorted_components:
        if component.text_content and len(component.text_content) < 2:
            continue
        if (
            component.row_max
            and (
                component.row_max / img_multiplier_factor["height"]
                - component.row_min / img_multiplier_factor["height"]
            )
            < 24
        ):
            continue
        position = (
            round(component.col_min / img_multiplier_factor["width"] - label_width / 2),
            round(
                component.row_min / img_multiplier_factor["height"] - label_height / 2
            ),
        )

        too_close = any(
            abs(position[0] - pos[0]) < label_width
            and abs(position[1] - pos[1]) < label_height * 2
            for pos in drawn_positions
        )

        if not too_close:
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
                position,
                label,
                width=label_width,
                height=label_height,
            )
            drawn_positions.append(position)
            label_map[label] = {
                "text": component.text_content or "",
                "position": position,
            }
            label_counter += 1

    os.makedirs("./output/annotated", exist_ok=True)
    output_image_path = f"./output/annotated/{os.path.basename(input_image_path)}"
    original_image.save(output_image_path)

    print(f"{len(label_map.keys())} elements found on the screen", end="")
    if show:
        cv2.imshow("Annotated Image", cv2.imread(output_image_path))
        cv2.waitKey(0)
        cv2.destroyWindow("Annotated Image")

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
