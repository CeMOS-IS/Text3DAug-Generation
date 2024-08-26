#!/usr/bin/env python
# -*- coding:utf-8 -*-
import argparse
import copy
import math

import numpy as np
import open3d as o3d


def visualize_mesh(mesh_path):
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    o3d.visualization.draw_geometries([mesh])


def o3d_image_to_numpy(img):
    img = np.asarray(img)
    img *= 255
    return img.astype(np.uint8)


def rotate_mesh(mesh, deg, axis):
    mesh = copy.deepcopy(mesh)

    rot = [0.0, 0.0, 0.0]
    for a in axis:
        rot[a] = math.radians(deg)

    # Euler Angles
    rot = mesh.get_rotation_matrix_from_xyz(np.asarray(rot))
    return mesh.rotate(rot)


def render_mesh_to_image(mesh, vis):
    # Update a visualizer instance
    vis.add_geometry(mesh)
    vis.poll_events()
    vis.update_renderer()

    # Render image
    mesh_img = vis.capture_screen_float_buffer(do_render=True)
    vis.remove_geometry(mesh)

    return o3d_image_to_numpy(mesh_img)
