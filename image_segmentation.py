from sam2 import SAM2Image, draw_masks
import cv2
import os
import platform
import numpy as np
from imread import imread

if platform.system().lower().startswith("linux") or os.path.abspath(__file__).startswith("/nix"):
    if os.environ.get('DISPLAY'):
        print("Warning: Force QT_QPA_PLATFORM=xcb for better user experience")
        os.environ['QT_QPA_PLATFORM'] = "xcb"

encoder_model_path = "models/sam2_hiera_base_plus_encoder.onnx"
decoder_model_path = "models/sam2_hiera_base_plus_decoder.onnx"

img = imread("Racing_Terriers.jpg")

sam2 = SAM2Image(encoder_model_path, decoder_model_path)

sam2.set_image(img)

masks = None
label_id = 0

def mouse_callback(event, x, y, flags, param):
    global label_id, masks
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"Left mouse button clicked at ({x}, {y})")
        sam2.add_point((x, y), 1, label_id)
        label_id += 1
        masks = sam2.update_mask(select_best=True)
    elif event == cv2.EVENT_RBUTTONDOWN:
        print(f"Right mouse button clicked at ({x}, {y})")

window_name = "image"
cv2.namedWindow(window_name)
cv2.setMouseCallback(window_name, mouse_callback)

while True:
    masked_img = draw_masks(img, masks) if masks is not None else img
    cv2.imshow(window_name, masked_img)
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

print("exit")
