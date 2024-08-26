#!/usr/bin/env python
# -*- coding:utf-8 -*-
import glob
import os

import numpy as np
from PIL import Image


def visualize_image(img):
    # o3d.visualization.draw_geometries([img])
    img = Image.fromarray(img)
    img.show()


def o3d_image_to_numpy(img):
    img = np.asarray(img)
    img *= 255
    return img.astype(np.uint8)


def search_folder_for_meshes(folder_path):
    folder_path = os.path.join(folder_path, "**", "*.obj")
    print(folder_path)
    return glob.glob(folder_path, recursive=False)
