import os
from PIL import Image, ImageDraw, ImageFont
from google.cloud import vision


def annotate_image_with_ocr(input_image_path):
    client_options = {
        "api_endpoint": "eu-vision.googleapis.com",
        "api_key": os.environ.get("GCLOUD_API_KEY"),
    }

    client = vision.ImageAnnotatorClient(client_options=client_options)

    with open(input_image_path, "rb") as image_file:
        content = image_file.read()

    image = vision.Image(content=content)
    response = client.text_detection(image=image)  # type: ignore
    texts = response.text_annotations

    original_image = Image.open(input_image_path)
    label_counter = 1
    label_prefix = "A"
    drawn_positions = []

    for text in texts:
        if len(text.description) < 2:
            continue
        vertices = [(vertex.x, vertex.y) for vertex in text.bounding_poly.vertices]

        too_close = any(
            abs(vertices[0][0] - pos[0]) < 48 * 2
            and abs(vertices[0][1] - pos[1]) < 24 * 2
            for pos in drawn_positions
        )

        if not too_close:
            if label_counter > 9:
                label_counter = 1
                next_char = chr(ord(label_prefix[-1]) + 1)
                if next_char == 'I':
                    next_char = 'J'  # Skip 'I' to avoid confusion with 'l'
                if label_prefix[-1] == 'Z':
                    label_prefix += 'A'
                else:
                    label_prefix = label_prefix[:-1] + next_char
            label = f"{label_prefix}{label_counter}"
            draw_square(original_image, vertices[0], label)
            drawn_positions.append(vertices[0])
            label_counter += 1

    output_filename = f"./annotated_{os.path.basename(input_image_path)}"
    original_image.save(output_filename)
    return output_filename


def draw_square(
    image,
    position,
    code,
    width=48,
    height=24,
    fill_color_start="#EFDD88",
    fill_color_end="#EBD872",
    outline_color="#EBD872",
):
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("arialbd.ttf", 22)
    except Exception:
        font = ImageFont.truetype("/Library/Fonts/Arial Bold.ttf", 22)
    x, y = position
    x, y = x - width, y - height
    x1, y1 = x + width, y + height

    # Create a gradient
    gradient = Image.new("RGB", (1, 24), color=fill_color_start)
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
        # Different way of doing bold
        # stroke_width=1,
        # stroke_fill="black",
    )
