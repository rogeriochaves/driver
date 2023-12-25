import sys
import os
from typing import List, Tuple, TypedDict

sys.path.append(os.path.dirname(__file__))
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
)

from os.path import join as pjoin
import cv2
import numpy as np

from detect_text.ocr import ocr_detection_google
from google.cloud import vision

from driver.types import AnnotatedImage
from detect_merge.merge import DetectElementsResponse


def resize_height_by_longest_edge(img_path, resize_length=800):
    org = cv2.imread(img_path)
    height, width = org.shape[:2]
    if height > width:
        return resize_length
    else:
        return int(resize_length * (height / width))


def color_tips():
    color_map = {
        "Text": (0, 0, 255),
        "Compo": (0, 255, 0),
        "Block": (0, 255, 255),
        "Text Content": (255, 0, 255),
    }
    board = np.zeros((200, 200, 3), dtype=np.uint8)

    board[:50, :, :] = (0, 0, 255)
    board[50:100, :, :] = (0, 255, 0)
    board[100:150, :, :] = (255, 0, 255)
    board[150:200, :, :] = (0, 255, 255)
    cv2.putText(board, "Text", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    cv2.putText(
        board, "Non-text Compo", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2
    )
    cv2.putText(
        board,
        "Compo's Text Content",
        (10, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 0, 0),
        2,
    )
    cv2.putText(board, "Block", (10, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    cv2.imshow("colors", board)


"""
ele:min-grad: gradient threshold to produce binary map
ele:ffl-block: fill-flood threshold
ele:min-ele-area: minimum area for selected elements
ele:merge-contained-ele: if True, merge elements contained in others
text:max-word-inline-gap: words with smaller distance than the gap are counted as a line
text:max-line-gap: lines with smaller distance than the gap are counted as a paragraph

Tips:
1. Larger *min-grad* produces fine-grained binary-map while prone to over-segment element to small pieces
2. Smaller *min-ele-area* leaves tiny elements while prone to produce noises
3. If not *merge-contained-ele*, the elements inside others will be recognized, while prone to produce noises
4. The *max-word-inline-gap* and *max-line-gap* should be dependent on the input image size and resolution

mobile: {'min-grad':4, 'ffl-block':5, 'min-ele-area':50, 'max-word-inline-gap':6, 'max-line-gap':1}
web   : {'min-grad':3, 'ffl-block':5, 'min-ele-area':25, 'max-word-inline-gap':4, 'max-line-gap':4}
"""
key_params = {
    "min-grad": 10,
    "ffl-block": 5,
    "min-ele-area": 50,
    "merge-contained-ele": False,
    "merge-line-to-paragraph": True,
    "remove-bar": False,
}


def detect_components(
    input_path_img, ocr_result: AnnotatedImage, showOCR=False, showUIED=False
) -> DetectElementsResponse:
    output_root = "output"

    # Resizes the image to be smaller because this process is heavy, and lower resolution
    # does not lose much quality when detecting components
    max_width_or_height = 982
    resized_height = resize_height_by_longest_edge(
        input_path_img, resize_length=max_width_or_height
    )
    # color_tips()

    is_clf = False

    import detect_text.text_detection as text

    os.makedirs(pjoin(output_root, "ocr"), exist_ok=True)
    text_json = text.text_detection(
        ocr_result, input_path_img, output_root, show=showOCR
    )

    import detect_compo.ip_region_proposal as ip

    os.makedirs(pjoin(output_root, "ip"), exist_ok=True)
    # switch of the classification func
    classifier = None
    if is_clf:
        classifier = {}
        from cnn.CNN import CNN

        # classifier['Image'] = CNN('Image')
        classifier["Elements"] = CNN("Elements")
        # classifier['Noise'] = CNN('Noise')
    compo_json = ip.compo_detection(
        input_path_img,
        output_root,
        key_params,
        classifier=classifier,
        resize_by_height=resized_height,
        show=False,
    )

    import detect_merge.merge as merge

    os.makedirs(pjoin(output_root, "merge"), exist_ok=True)
    name = input_path_img.split("/")[-1][:-4]
    compo_path = pjoin(output_root, "ip", str(name) + ".json")
    ocr_path = pjoin(output_root, "ocr", str(name) + ".json")
    board, components = merge.merge(
        input_path_img,
        compo_json,
        text_json,
        pjoin(output_root, "merge"),
        is_remove_bar=key_params["remove-bar"],
        is_paragraph=key_params["merge-line-to-paragraph"],
        show=showUIED,
    )

    return components


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv("../../.env")

    input_image_path = "./twitter.png"

    client_options = {
        "api_endpoint": "eu-vision.googleapis.com",
        "api_key": os.environ.get("GCLOUD_VISION_API_KEY"),
    }
    client = vision.ImageAnnotatorClient(client_options=client_options)
    with open(input_image_path, "rb") as image_file:
        content = image_file.read()

    image = vision.Image(content=content)
    ocr_result = client.text_detection(image=image)  # type: ignore

    # ocr_result2 = ocr_detection_google("./twitter.png")

    components = detect_components(
        input_image_path, ocr_result, showOCR=True, showUIED=True
    )

    print("\n\ncomponents\n\n", components)
