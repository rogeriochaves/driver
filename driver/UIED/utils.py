import os
import sys
import cv2
import cv2.typing


def show_image(window_name: str, image: cv2.typing.MatLike):
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    if sys.platform == "darwin":
        os.system(
            """/usr/bin/osascript -e 'tell app "Finder" to set frontmost of process "python" to true' """
        )

    cv2.imshow(window_name, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # Hack to fix closing bug on macos: https://stackoverflow.com/questions/6116564/destroywindow-does-not-close-window-on-mac-using-python-and-opencv
    for _ in range(1, 5):
        cv2.waitKey(1)
