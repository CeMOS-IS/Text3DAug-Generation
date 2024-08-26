#!/usr/bin/env python
# -*- coding:utf-8 -*-
import argparse
import glob
import os
import shutil

import numpy as np
import open3d as o3d
from tqdm import tqdm

from models.mesh_saver import save_mesh
from packages.clip_eval.open3d_functions import rotate_mesh


def remove_non_obj(f):
    """Removes a file or folder that does not end in .obj

    Args:
        f (str): File / folder path
    """
    if not os.path.isfile(f):
        shutil.rmtree(f)
    elif not f.endswith(".obj"):
        os.remove(f)


def post_process_mesh(mesh_path):
    """Corrects the mesh of One-2-3-45 into the coordinate system uniform
    to all models used in Text3DAug.

    Mesh is scaled into a unit cube, with the z-axis facing up.

    Args:
        mesh_path (str): File path to mesh .obj file.
    """
    mesh = o3d.io.read_triangle_mesh(mesh_path, enable_post_processing=True)

    # Correct axis rotation so z is up
    mesh = rotate_mesh(mesh, 90, [0])

    # Change scale into unit cube
    vertices = np.copy(mesh.vertices)[:, :3]
    vertices[:, 2] -= np.min(vertices[:, 2])
    max_z = np.max(vertices[:, 2])

    correction = 1 / max_z
    vertices *= correction
    mesh.vertices = o3d.utility.Vector3dVector(vertices)

    save_mesh(mesh, mesh_path)


def search_folder_for_meshes(folder_path):
    """Recursively searches a folder and subfolders for .obj mesh files.

    Args:
        folder_path (str): Path to folder.

    Returns:
        List: filenames of all found .obj files
    """
    folder_path = os.path.join(folder_path, "**", "*.obj")
    print(folder_path)
    return glob.glob(folder_path, recursive=True)


def remove_useless_files(folder_path):
    """During mesh generation, One-2-3-45 creates folders that are
    not used for further steps. This function removes unnecessary folders
    and files left over from the generation process.

    Args:
        folder_path (str): Path to mesh output folder of One-2-3-45, usually ./exp.
    """
    # Remove useless files
    class_dirs = glob.glob(os.path.join(folder_path, "*"))
    for c_dir in class_dirs:
        files = glob.glob(os.path.join(c_dir, "**", "*"))

        for f in files:
            remove_non_obj(f)

    # Move and rename obj files
    for c_dir in class_dirs:
        files = glob.glob(os.path.join(c_dir, "**", "*"))
        for f in files:
            folder = os.path.dirname(f)
            number = os.path.basename(folder)

            new_name = str(number) + ".obj"
            new_path = os.path.join(c_dir, new_name)

            shutil.move(f, new_path)

    # Remove useless folders
    class_dirs = glob.glob(os.path.join(folder_path, "*"))
    for c_dir in class_dirs:
        empty_folder = glob.glob(os.path.join(c_dir, "**"))

        for f in empty_folder:
            remove_non_obj(f)


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

    remove_useless_files(args.folder)

    mesh_files = search_folder_for_meshes(args.folder)
    for mesh in tqdm(mesh_files):
        post_process_mesh(mesh)
