#!/usr/bin/env python
# -*- coding:utf-8 -*-
import argparse
import glob
import os

import numpy as np
import open3d as o3d


def visualize_mesh(mesh_path):
    """Visualize mesh in Open3D.

    Args:
        mesh_path (str): Path to .obj mesh.
    """

    mesh = o3d.io.read_triangle_mesh(mesh_path, enable_post_processing=True)

    # mesh = rotate_mesh(mesh, 90, [0])
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=0.5, origin=np.array([0.0, 0.0, 0.0])
    )
    print(f"Viewing:\t{mesh_path}")
    o3d.visualization.draw_geometries([mesh, frame])


def search_folder_for_meshes(folder_path):
    """Searches a folder and its subdirectories, returning all
    paths to .obj mesh files.

    Args:
        folder_path (str): Path to folder.

    Returns:
        List[str]: Paths to .obj files.
    """
    folder_path = os.path.join(folder_path, "**", "*.obj")
    print(folder_path)
    return glob.glob(folder_path, recursive=True)


def cli_parsing():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--folder",
        type=str,
        required=True,
        help="Folder to Mesh files.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    # Get arguments from CLI
    args = cli_parsing()

    mesh_files = search_folder_for_meshes(args.folder)
    for mesh in mesh_files:
        visualize_mesh(mesh)
