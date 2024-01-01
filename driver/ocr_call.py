import base64
from dataclasses import dataclass
import json
import os
from typing import List, cast

import requests
from urllib.parse import urlencode, quote_plus

from driver.logger import print_action
from google.cloud import vision
from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from azure.cognitiveservices.vision.computervision.models import (
    OcrResult,
)
from msrest.authentication import CognitiveServicesCredentials

from driver.types import (
    AnnotatedImage,
    BoundingPoly,
    DebugConfig,
    TextAnnotation,
    Vertex,
)
from driver.utils import image_to_base64


def ocr_text_detection(input_image_path, config: DebugConfig) -> AnnotatedImage:
    ocr_provider = config["ocr_provider"]
    if not ocr_provider:
        if os.environ.get("AZURE_VISION_API_KEY"):
            ocr_provider = "azure"
        elif os.environ.get("GCLOUD_VISION_API_KEY"):
            ocr_provider = "google"
        elif os.environ.get("BAIDU_OCR_API_KEY"):
            ocr_provider = "baidu"

    if ocr_provider == "azure":
        print_action("Annotating screenshot with Azure Vision")
        return azure_ocr_text_detect(input_image_path)
    elif ocr_provider == "google":
        print_action("Annotating screenshot with Google Cloud Vision")
        return google_ocr_text_detect(input_image_path)
    elif ocr_provider == "baidu":
        print_action("Annotating screenshot with Baidu Vision")
        return baidu_ocr_text_detect(input_image_path)
    else:
        raise Exception(
            "No OCR API env variable set, please set either AZURE_VISION_API_KEY or GCLOUD_VISION_API_KEY"
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
                Vertex(
                    x=int(bounding_box[0]) + int(bounding_box[2]),
                    y=int(bounding_box[1]) + int(bounding_box[3]),
                ),
            ]
            annotations.append(
                TextAnnotation(
                    description=description,
                    bounding_poly=BoundingPoly(vertices=vertexes),
                )
            )
    result = AnnotatedImage(text_annotations=annotations)

    return result


def baidu_ocr_text_detect(input_image_path) -> AnnotatedImage:
    api_key = os.environ["BAIDU_OCR_API_KEY"]
    secret_key = os.environ["BAIDU_OCR_SECRET_KEY"]

    def get_access_token():
        url = "https://aip.baidubce.com/oauth/2.0/token"
        params = {
            "grant_type": "client_credentials",
            "client_id": api_key,
            "client_secret": secret_key,
        }
        return str(requests.post(url, params=params).json().get("access_token"))

    url = (
        "https://aip.baidubce.com/rest/2.0/ocr/v1/accurate?access_token="
        + get_access_token()
    )
    payload = urlencode(
        {
            "detect_direction": "false",
            "vertexes_location": "true",
            "paragraph": "false",
            "probability": "false",
            "image": image_to_base64(input_image_path),
        },
        quote_via=quote_plus,
    )
    headers = {
        "Content-Type": "application/x-www-form-urlencoded",
        "Accept": "application/json",
    }
    response = requests.request("POST", url, headers=headers, data=payload)
    if response.status_code != 200 or "error_code" in json.loads(response.text):
        raise Exception("Baidu OCR failed to annotate screenshot")
    words = json.loads(response.text)["words_result"]

    annotations: List[TextAnnotation] = []
    for word in words:
        vertices = [
            Vertex(x=vertex["x"], y=vertex["y"]) for vertex in word["vertexes_location"]
        ]
        annotations.append(
            TextAnnotation(
                description=word["words"],
                bounding_poly=BoundingPoly(vertices=vertices),
            )
        )
    result = AnnotatedImage(text_annotations=annotations)

    return result
