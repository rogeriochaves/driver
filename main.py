import os
from PIL import Image, ImageDraw, ImageFont
from google.cloud import vision

client_options = {
    "api_endpoint": "eu-vision.googleapis.com",
    "api_key": os.environ.get("GCLOUD_API_KEY"),
}

client = vision.ImageAnnotatorClient(client_options=client_options)

with open("./screenshot.png", "rb") as image_file:
    content = image_file.read()

image = vision.Image(content=content)

response = client.text_detection(image=image)  # type: ignore

texts = response.text_annotations


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
        font = ImageFont.truetype("arial.ttf", 22)
    except Exception:
        font = ImageFont.truetype("/Library/Fonts/Arial.ttf", 22)
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
        stroke_width=1,
        stroke_fill="black",
    )


# Load the original image
original_image = Image.open("./screenshot.png")

# Initialize variables for labeling
label_counter = 1
label_prefix = "A"
print("Texts:")

for text in texts:
    if len(text.description) < 2:
        continue
    vertices = [(vertex.x, vertex.y) for vertex in text.bounding_poly.vertices]
    print(f'\n"{text.description}"')
    print("bounds: {}".format(",".join([f"({v[0]},{v[1]})" for v in vertices])))

    # Draw a square on the first vertex
    if label_counter > 9:
        label_counter = 1
        if label_prefix[-1] == "Z":
            label_prefix += "A"
        else:
            label_prefix = label_prefix[:-1] + chr(ord(label_prefix[-1]) + 1)
    label = f"{label_prefix}{label_counter}"
    draw_square(original_image, vertices[0], label)
    label_counter += 1

# Save the modified image
output_filename = "./annotated_screenshot.png"
original_image.save(output_filename)

# Display the image (this will vary depending on the OS)
os.system(
    f"open {output_filename}" if os.name == "posix" else f"start {output_filename}"
)
