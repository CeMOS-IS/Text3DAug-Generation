#!/usr/bin/env python
# -*- coding:utf-8 -*-
import argparse
import os
import time
from collections import Counter

import open3d as o3d
from PIL import Image
from tqdm import tqdm

from packages.clip_eval.file_functions import search_folder_for_meshes, visualize_image
from packages.clip_eval.metric import ClipScore
from packages.clip_eval.open3d_functions import (
    render_mesh_to_image,
    rotate_mesh,
    visualize_mesh,
)

CLASSES = [
    "bicycle",
    "bicyclist",
    "bus",
    "car",
    "construction_vehicle",
    "motorcycle",
    "motorcyclist",
    "other_vehicle",
    "pedestrian",
    "truck",
    "unrecognized",
    "other",
    "random",
]


def cli_parsing():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--folder",
        type=str,
        required=True,
        help="Folder to Mesh files.",
    )

    parser.add_argument(
        "--out_path",
        type=str,
        required=True,
        help="Filename with .txt to save results to.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Flag. Set to enable visualizations for debugging.",
    )

    return parser.parse_args()


def init_o3d_visualizer(visible):
    """Intitialize and Open3D visualizer instance.

    Args:
        visible (bool): Make visible or not.

    Returns:
        Open3D: Visualizer instance.
    """
    # Render images
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=visible)
    return vis


def render_to_images(mesh_path, vis, use_color, debug):
    """Uses Open3D to rotate a mesh and render four images from
    each side.

    Args:
        mesh_path (str): Path to mesh.
        vis (Open3d): Open3d visualizer instance.
        use_color (bool): Keep color in mesh.
        debug (bool): Visualize each mesh view.

    Returns:
        [4 x np.ndarray]: Images rendered from each side.
    """
    # Open mesh
    mesh = o3d.io.read_triangle_mesh(mesh_path, enable_post_processing=True)

    # Rotate mesh into camera views #################################
    # NOTE: THIS MIGHT HAVE TO BE CHANGED DEPENDING ON HOW
    # OPEN3D RENDERS INITIAL IMAGES
    #
    # SOMETIMES VIEW IS FROM FRONT, OTHER TIMES TOP DOWN
    # WITHOUT CHANGING AXES

    front_mesh = rotate_mesh(mesh, -90, [0])
    back_mesh = rotate_mesh(front_mesh, 180, [1])
    side_mesh = rotate_mesh(front_mesh, -90, [1])
    other_side_mesh = rotate_mesh(side_mesh, 180, [1])

    #################################################################

    front_view = mesh_to_gray_image(front_mesh, use_color, vis, debug)
    side_view = mesh_to_gray_image(side_mesh, use_color, vis, debug)
    back_view = mesh_to_gray_image(back_mesh, use_color, vis, debug)
    other_side_view = mesh_to_gray_image(other_side_mesh, use_color, vis, debug)

    return front_view, side_view, back_view, other_side_view


def mesh_to_gray_image(mesh, use_color, vis, debug):
    """Renders a single image from a mesh.
    Shading is done according to estimated vertex normals.

    Args:
        mesh (str): Path to mesh.
        use_color (bool): Render with color.
            If False, surface normal shading is used.
        vis (Open3d): Instance to Open3D visualizer.
        debug (bool): Visualize the image.

    Returns:
        np.ndarray: Rendered image.
    """
    mesh.compute_vertex_normals()
    mesh.paint_uniform_color([0.5, 0.5, 0.5])
    img = render_mesh_to_image(mesh, vis)

    # Optional debug visualization
    if debug:
        visualize_image(img)

    # To PIL Image
    img = Image.fromarray(img)

    if use_color:
        return img
    else:
        return img.convert("L")  # <- grayscale image


if __name__ == "__main__":
    # Get arguments from CLI
    args = cli_parsing()

    # Initalize CLIP and Open3D Rendering
    clip_score = ClipScore(model="ViT-L/14")
    vis = init_o3d_visualizer(visible=False)

    avg_score = 0
    value_list = []

    # Find and iterate over all meshes
    mesh_files = search_folder_for_meshes(args.folder)

    with open(args.out_path, mode="w") as txt_file:
        for n, mesh in enumerate(tqdm(mesh_files)):
            # Get filename
            mesh_fn = os.path.basename(mesh)
            class_folder = os.path.basename(os.path.dirname(mesh))

            class_idx = CLASSES.index(class_folder)

            # Get CLIP score for different views of the mesh
            views = render_to_images(mesh, vis, False, args.debug)
            view_scores = (clip_score(v, CLASSES)[0][class_idx] for v in views)

            # Choose the best score from all views
            max_score = max(*view_scores)
            avg_score += max_score
            value_list.append(max_score)

            # Write into .txt log
            txt_file.write(
                f"{class_folder} {CLASSES[class_idx]} {mesh_fn} {max_score}\n"
            )

            # Wait on debug
            if args.debug:
                time.sleep(5)

    # Result
    avg_score /= n + 1
    print(
        f"Class index {class_idx} with class {CLASSES[class_idx]} and {n+1} Samples has average CLIP score of {avg_score}"
    )

    # Show distribution of values
    value_list = [int(v * 10) for v in value_list]
    value_list = [v / 10 for v in value_list]
    print(Counter(value_list))
