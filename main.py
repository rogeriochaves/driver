import os
from ocr_draw import annotate_image_with_ocr

input_image_path = "./screenshot.png"
output_image_path = annotate_image_with_ocr(input_image_path)

# Display the image (this will vary depending on the OS)
os.system(
    f"open {output_image_path}" if os.name == "posix" else f"start {output_image_path}"
)
