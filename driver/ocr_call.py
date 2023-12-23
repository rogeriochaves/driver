from dataclasses import dataclass
import os
from typing import List, cast

from driver.logger import print_action
from google.cloud import vision
from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from azure.cognitiveservices.vision.computervision.models import (
    OcrResult,
)
from msrest.authentication import CognitiveServicesCredentials


@dataclass
class Vertex:
    x: int
    y: int


@dataclass
class BoundingPoly:
    vertices: List[Vertex]


@dataclass
class TextAnnotation:
    description: str
    bounding_poly: BoundingPoly


@dataclass
class AnnotatedImage:
    text_annotations: List[TextAnnotation]


def ocr_text_detection(input_image_path) -> AnnotatedImage:
    if os.environ.get("GCLOUD_VISION_API_KEY"):
        print_action("Annotating screenshot with Google Cloud Vision")
        return google_ocr_text_detect(input_image_path)
    elif os.environ.get("AZURE_VISION_API_KEY"):
        print_action("Annotating screenshot with Azure Vision")
        return azure_ocr_text_detect(input_image_path)
    raise Exception(
        "No OCR API env variable set, please set either GCLOUD_VISION_API_KEY or AZURE_VISION_API_KEY"
    )


def google_ocr_text_detect(input_image_path) -> AnnotatedImage:
    client_options = {
        "api_endpoint": "eu-vision.googleapis.com",
        "api_key": os.environ.get("GCLOUD_VISION_API_KEY"),
    }

    client = vision.ImageAnnotatorClient(client_options=client_options)
    with open(input_image_path, "rb") as image_file:
        content = image_file.read()

    image = vision.Image(content=content)
    response = client.text_detection(image=image)  # type: ignore
    return response


def azure_ocr_text_detect(input_image_path) -> AnnotatedImage:
    subscription_key = os.environ["AZURE_VISION_API_KEY"]
    endpoint = os.environ["AZURE_VISION_ENDPOINT"]

    client = ComputerVisionClient(
        endpoint, CognitiveServicesCredentials(subscription_key)
    )

    with open(input_image_path, "rb") as image_file:
        image_analysis = cast(
            OcrResult,
            client.recognize_printed_text_in_stream(
                image=image_file,
            ),
        )

    annotations: List[TextAnnotation] = []
    for region in image_analysis.regions or []:
        for line in region.lines:
            description = ""
            for word in line.words:
                description += word.text + " "

            bounding_box = line.bounding_box.split(",")
            vertexes = [
                Vertex(x=int(bounding_box[0]), y=int(bounding_box[1])),
                Vertex(x=int(bounding_box[2]), y=int(bounding_box[3])),
            ]
            annotations.append(
                TextAnnotation(
                    description=description,
                    bounding_poly=BoundingPoly(vertices=vertexes),
                )
            )
    result = AnnotatedImage(text_annotations=annotations)

    return result
