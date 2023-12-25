import sys
from dotenv import load_dotenv
import argparse

from driver.types import DebugConfig

load_dotenv()

from driver.executor import start


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("task", type=str, help="The task to execute")
    parser.add_argument(
        "--debug-ocr",
        action="store_true",
        help="Display bounding boxes of OCR results for debugging their position before executing the action",
    )
    parser.add_argument(
        "--debug-uied",
        action="store_true",
        help="Display bounding boxes of UIED detected elements for debugging their position before executing the action",
    )
    parser.add_argument(
        "--debug-annotations",
        action="store_true",
        help="Display annotations for debugging their position before executing the action",
    )
    args = parser.parse_args()

    debug: DebugConfig = {
        "annotations": args.debug_annotations,
        "ocr": args.debug_ocr,
        "uied": args.debug_uied,
    }

    start(args.task, debug=debug)


if __name__ == "__main__":
    main()
