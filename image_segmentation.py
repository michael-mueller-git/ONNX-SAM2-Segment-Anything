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

point_coords = [np.array([[420, 440]]), np.array([[360, 275], [370, 210]]), np.array([[810, 440]]),
                np.array([[920, 314]])]
point_labels = [np.array([1]), np.array([1, 1]), np.array([1]), np.array([1])]

for label_id, (point_coord, point_label) in enumerate(zip(point_coords, point_labels)):
    for i in range(point_label.shape[0]):
        sam2.add_point((point_coord[i][0], point_coord[i][1]), point_label[i], label_id)

    masks = sam2.update_mask(select_best=True)

    masked_img = draw_masks(img, masks)

    cv2.imshow("masked_img", masked_img)
    if cv2.waitKey(1000) & 0xFF == ord('q'):
        break
