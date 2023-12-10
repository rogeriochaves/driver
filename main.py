# import os
# from ocr_draw import annotate_image_with_ocr

# input_image_path = "./screenshot.png"
# output_image_path = annotate_image_with_ocr(input_image_path)

# # Display the image (this will vary depending on the OS)
# os.system(
#     f"open {output_image_path}" if os.name == "posix" else f"start {output_image_path}"
# )


import sys
from brain import (
    extract_high_level_plan_and_actions,
    extract_structured_actions,
    initial_task_outline,
)
from executor import execute, start
from ocr_draw import annotate_image_with_ocr
from utils import image_to_base64


task = "Hey, can you write an email to lauraestevesps@gmail.com saying that I love her?"

start(task)