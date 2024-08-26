#!/usr/bin/env python
# -*- coding:utf-8 -*-
import argparse
import glob
import os
import pathlib
import random
from contextlib import redirect_stdout
from io import StringIO

import numpy as np
import open3d as o3d
from tqdm import tqdm

HEIGHT_SCALING_BY_LABEL = {
    "bicycle": 1.3,
    "bicyclist": 1.8,
    "bus": 3.2,
    "car": 1.5,
    "construction_vehicle": 3.2,
    "motorcycle": 1.3,
    "motorcyclist": 1.8,
    "other_vehicle": 2.8,
    "pedestrian": 2.1,
    "truck": 3.0,
}
HEIGHT_MIN_SCALE = 0.6
RANDOM_SEED = 55555


class NullIO(StringIO):
    def write(self, txt):
        pass


def silent(fn):
    """Decorator to silence functions printing."""

    def silent_fn(*args, **kwargs):
        with redirect_stdout(NullIO()):
            return fn(*args, **kwargs)

    return silent_fn


def save_mesh(mesh: o3d.geometry.TriangleMesh, filename: str, ext: str = "obj") -> None:
    """Saves an Open3D mesh in Open3D supported formats.
    Ref.: http://www.open3d.org/docs/release/tutorial/geometry/file_io.html

    Args:
        mesh (o3d.geometry.TriangleMesh): Open3D mesh object
        filename (str): Pathlib Path or string for filename
        ext (str, optional): File extension of mesh. Defaults to "obj".
    """
    filename = pathlib.Path(filename)
    if not filename.suffix != str(ext):
        filename = filename / ext  # <- pathlib style joining of strings
    o3d.io.write_triangle_mesh(str(filename), mesh)


def check_mesh_height(xyz: np.ndarray, check_height: float):
    if np.max(xyz[:, 2]) != check_height:
        print(np.max(xyz[:, 2]))
        # raise ValueError(f"Mesh height maximum != 1.0")
    return None


def open_scale_save_mesh(mesh_path: str, height: float):
    """
    Opens an .obj mesh file and scales it according to the input height.
    The rescaled mesh ist then saved inplace.
    """

    # Open mesh
    open_mesh = silent(o3d.io.read_triangle_mesh)  # Supress annoying Open3D printing
    mesh = open_mesh(mesh_path, enable_post_processing=True, print_progress=False)

    # Get vertices
    vertices = np.copy(mesh.vertices)
    xyz = vertices[:, :3]
    check_mesh_height(xyz, check_height=1.0)

    # Scale vertices proportional to height
    xyz *= height
    vertices[:, :3] = xyz
    # check_mesh_height(xyz, check_height=height)

    # Overwrite mesh vertices and save
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    if np.isnan(xyz).any():
        print("DELETING", mesh_path)
        os.remove(mesh_path)
    else:
        save_mesh(mesh, mesh_path)


def get_class_folders(base_mesh_folder: str) -> dict:
    """
    Get child directories of a folder and map these into a dictionary.
    """
    class_folders = {}
    for f in os.scandir(base_mesh_folder):
        if f.is_dir():
            class_folders[f.name] = f.path
    return class_folders


def get_obj_mesh_files(class_folder: str) -> list:
    """
    Get obj mesh files from within a folder.
    """
    glob_fn = os.path.join(class_folder, "*.obj")
    return glob.glob(glob_fn, recursive=False)


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
    random.seed(RANDOM_SEED)
    args = cli_parsing()

    # Get class folder
    class_folders = get_class_folders(args.folder)

    # Check if the folder is a supported class
    for c in class_folders.keys():
        if c not in HEIGHT_SCALING_BY_LABEL.keys():
            raise NotImplementedError(
                f"Folder {c} not supported by {HEIGHT_SCALING_BY_LABEL.keys()}"
            )

    # Go through each folder and select the height
    for class_name, folder_path in class_folders.items():
        max_height = HEIGHT_SCALING_BY_LABEL[class_name]
        min_height = max_height * HEIGHT_MIN_SCALE

        # Go through each file and randomly choose a height within the range
        class_mesh_files = get_obj_mesh_files(folder_path)
        print(f"Working on class {class_name} with {len(class_mesh_files)} meshes.")

        for mesh_file in tqdm(class_mesh_files):

            # Scale a mesh by the random height and save it
            random_h = random.uniform(min_height, max_height)
            open_scale_save_mesh(mesh_file, random_h)
