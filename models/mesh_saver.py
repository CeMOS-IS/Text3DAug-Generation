#!/usr/bin/env python
# -*- coding:utf-8 -*-

import pathlib

import open3d as o3d


def visualize(mesh):
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(mesh)
    vis.run()
    vis.destroy_window()


def open_obj(filename):
    textured_mesh = o3d.io.read_triangle_mesh(filename)
    visualize(textured_mesh)


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
