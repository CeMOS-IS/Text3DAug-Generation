#!/usr/bin/env python
# -*- coding:utf-8 -*-

import open3d as o3d


def visualize(mesh):
    """Visualizes a pointcloud or mesh in Open3D.

    Args:
        mesh (o3d): Pointcloud / mesh
    """
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(mesh)
    vis.run()
    vis.destroy_window()


def save_mesh(mesh, filename, ext=".obj"):
    """Saves an Open3D mesh.

    Args:
        mesh (o3d): Open3d mesh geometry.
        filename (str): Output filename.
        ext (str, optional): Extension to use for filename. Defaults to ".obj".
    """
    if not filename.endswith(ext):
        filename += ext
    print(filename)
    o3d.io.write_triangle_mesh(filename, mesh)

    textured_mesh = o3d.io.read_triangle_mesh(filename)

    visualize(textured_mesh)  # The mesh looks broken here
