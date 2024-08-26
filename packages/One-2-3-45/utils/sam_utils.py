import os
import time

import numpy as np
import torch
from PIL import Image
from segment_anything import SamPredictor, sam_model_registry


def sam_init(device_id=0):
    sam_checkpoint = os.path.join(os.path.dirname(__file__), "../sam_vit_h_4b8939.pth")
    model_type = "vit_h"

    device = "cuda:{}".format(device_id) if torch.cuda.is_available() else "cpu"

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint).to(device=device)
    predictor = SamPredictor(sam)
    return predictor


def sam_out_nosave(predictor, input_image, *bbox_sliders):
    bbox = np.array(bbox_sliders)
    image = np.asarray(input_image)

    start_time = time.time()
    predictor.set_image(image)

    masks_bbox, scores_bbox, logits_bbox = predictor.predict(
        box=bbox, multimask_output=True
    )

    out_image = np.zeros((image.shape[0], image.shape[1], 4), dtype=np.uint8)
    out_image[:, :, :3] = image
    out_image_bbox = out_image.copy()
    out_image_bbox[:, :, 3] = (
        masks_bbox[-1].astype(np.uint8) * 255
    )  # np.argmax(scores_bbox)
    torch.cuda.empty_cache()
    return Image.fromarray(out_image_bbox, mode="RGBA")
