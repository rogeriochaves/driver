import os
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
print("Texts:")

for text in texts:
    print(f'\n"{text.description}" {text}')

    # vertices = [f"({vertex.x},{vertex.y})" for vertex in text.bounding_poly.vertices]

    # print("bounds: {}".format(",".join(vertices)))
