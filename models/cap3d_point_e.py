#!/usr/bin/env python
# -*- coding:utf-8 -*-

import argparse
import glob
import os
import random
import urllib

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import torch
from cap3d.point_e.diffusion.configs import DIFFUSION_CONFIGS, diffusion_from_config
from cap3d.point_e.diffusion.sampler import PointCloudSampler
from cap3d.point_e.models.configs import MODEL_CONFIGS, model_from_config
from cap3d.point_e.models.download import load_checkpoint
from cap3d.point_e.util.pc_to_mesh import marching_cubes_mesh
from cap3d.point_e.util.plotting import plot_point_cloud

try:
    from .mesh_saver import save_mesh
except ImportError:
    from mesh_saver import save_mesh

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
THIS_DIR = os.path.dirname(os.path.abspath(__file__))


def get_and_load_weights():
    # Make folder at this file
    weight_dir = os.path.join(THIS_DIR, ".cap3d_weights")
    os.makedirs(weight_dir, exist_ok=True)

    # Actually download the weights
    weight_fn = os.path.join(weight_dir, "cap3d_point_e.pt")
    if not os.path.isfile(weight_fn):
        dl_url = "https://huggingface.co/datasets/tiange/Cap3D/resolve/main/misc/our_finetuned_models/pointE_finetuned_with_330kdata.pth?download=true"
        urllib.request.urlretrieve(dl_url, weight_fn)
        print("Cap3D Point-E weights complete")

    # Load and return weights
    return torch.load(weight_fn, map_location=DEVICE)


class Cap3DPointE:
    def __init__(self) -> None:
        # Meshing model
        name = "sdf"
        model = model_from_config(MODEL_CONFIGS[name], DEVICE)
        model.eval()
        model.load_state_dict(load_checkpoint(name, DEVICE))
        self.model = model

        # Text -> pointcloud model
        base_name = "base40M-textvec"
        base_model = model_from_config(MODEL_CONFIGS[base_name], DEVICE)
        base_model.eval()
        base_model_weights = get_and_load_weights()["model_state_dict"]
        base_model.load_state_dict(base_model_weights)

        base_diffusion = diffusion_from_config(DIFFUSION_CONFIGS[base_name])

        # Upsampling model
        upsampler_model = model_from_config(MODEL_CONFIGS["upsample"], DEVICE)
        upsampler_model.eval()
        upsampler_diffusion = diffusion_from_config(DIFFUSION_CONFIGS["upsample"])
        upsampler_model.load_state_dict(load_checkpoint("upsample", DEVICE))

        # Assemble models for sampling
        self.sampler = PointCloudSampler(
            device=DEVICE,
            models=[base_model, upsampler_model],
            diffusions=[base_diffusion, upsampler_diffusion],
            num_points=[1024, 4096 - 1024],
            aux_channels=["R", "G", "B"],
            guidance_scale=[3.0, 0.0],
            model_kwargs_key_filter=(
                "texts",
                "",
            ),
        )

    def __call__(self, prompt: str, debug: bool = False) -> o3d.geometry.TriangleMesh:
        """Generates a mesh from a text prompt using Point-E and marching cubes.

        Args:
            prompt (str): A single prompt
            debug (bool, optional): Visualize for debugging. Defaults to False.

        Returns:
            o3d.geometry.TriangleMesh: Output mesh as open3d object.
        """
        prompt = self.preprocess(prompt)

        # Iterativly overwrite result of during diffusion steps
        pointcloud = None
        for x in self.generate(prompt):
            pointcloud = x

        pointcloud = self.sampler.output_to_point_clouds(pointcloud)[0]

        # Optionally visualize
        if debug:
            self._visualize(sample=pointcloud, sample_type="pc")

        # Mesh the pointcloud with marching cubes
        mesh = self.mesh(pointcloud)
        mesh = self.unify_coordinates(mesh)
        o3d_mesh = self.post_process(mesh)
        if debug:
            self._visualize(sample=o3d_mesh, sample_type="mesh")

        return o3d_mesh

    def preprocess(self, prompt):
        """
        Put the prompt in the proper format.
        """
        return dict(texts=[prompt])

    def generate(self, prompt):
        """
        Use model to generate a pointcloud from text.
        """
        for x in self.sampler.sample_batch_progressive(
            batch_size=1, model_kwargs=prompt
        ):
            yield x

    def generate_exception(self, rider: str, bike: str):
        rider_mesh = o3d.io.read_triangle_mesh(rider, enable_post_processing=True)
        bike_mesh = o3d.io.read_triangle_mesh(bike, enable_post_processing=True)

        # Move rider up, so they are sitting on the bipedal
        vertices = np.copy(rider_mesh.vertices)
        vertices[:, 2] += 0.25
        rider_mesh.vertices = o3d.utility.Vector3dVector(vertices)

        bike_mesh += rider_mesh  # Add person to bike
        bike_mesh = self.unify_coordinates(bike_mesh, o3d_mesh=True)
        # o3d.visualization.draw_geometries([bike_mesh]) # NOTE: For debugging
        return bike_mesh

    def transform(self, *args):
        # Currently unused
        return None

    def mesh(self, pointcloud):
        """
        Applies marching cubes to generate mesh from pointcloud.
        """
        # Increased grid size for higher resolution on small objects
        return marching_cubes_mesh(
            pc=pointcloud, model=self.model, batch_size=4096, grid_size=128
        )

    def unify_coordinates(self, mesh, o3d_mesh: bool = False):
        """
        Scales height of instance into 0..1 while maintaining aspect ratio.
        """
        if o3d_mesh:
            vertices = np.copy(mesh.vertices)[:, :3]
        else:
            vertices = np.copy(mesh.verts[:, :3])

        vertices[:, 2] -= np.min(vertices[:, 2])
        max_z = np.max(vertices[:, 2])

        correction = 1 / max_z
        vertices *= correction

        if o3d_mesh:
            mesh.vertices = o3d.utility.Vector3dVector(vertices)
        else:
            mesh.verts[:, :3] = vertices
        return mesh

    def post_process(self, mesh):
        """
        Converts the marching cubes mesh into an Open3D mesh.
        """

        triangles = mesh.faces[:, :3]
        vertices = mesh.verts[:, :3]
        colors = mesh.vertex_channels
        rgb = np.stack(
            [
                colors["R"],
                colors["G"],
                colors["B"],
            ],
            axis=-1,
            dtype=np.float64,
        )

        mesh = o3d.geometry.TriangleMesh()
        mesh.triangles = o3d.utility.Vector3iVector(triangles)
        mesh.vertices = o3d.utility.Vector3dVector(vertices)
        mesh.vertex_colors = o3d.utility.Vector3dVector(rgb)
        return mesh

    def _visualize(self, sample, sample_type):
        if sample_type == "pc":
            fig = plot_point_cloud(
                sample,
                grid_size=3,
                fixed_bounds=((-0.75, -0.75, -0.75), (0.75, 0.75, 0.75)),
            )

            plt.show(block=True)
            # plt.pause(100)
            plt.close(fig)
            plt.pause(1)

        elif sample_type == "mesh":
            o3d.visualization.draw_geometries([sample])

        else:
            raise NotImplementedError(
                f"Visualization for sample type {sample_type} not implemented."
            )

        return None


def cli_parsing():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--folder",
        type=str,
        required=True,
        help="Path to generated mesh class folders.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    GENERATE_EXCEPTIONS = False
    args = cli_parsing()

    # For testing
    point_e = Cap3DPointE()

    if not GENERATE_EXCEPTIONS:
        x = point_e("motorcycle", debug=True)
        exit()

    # Generate exception classes
    folder = args.folder
    pedestrians = os.path.join(folder, "pedestrian")
    motocycles = os.path.join(folder, "motorcycle")
    bicycle = os.path.join(folder, "bicycle")

    # Search folder for all .obj files
    all_pedestrians = glob.glob(os.path.join(pedestrians, "*.obj"))
    all_motocycles = glob.glob(os.path.join(motocycles, "*.obj"))
    all_bicycles = glob.glob(os.path.join(bicycle, "*.obj"))

    # Shuffle files of pedestrians to mix up data
    random.shuffle(all_pedestrians)

    # Create motorcyclists
    for i, (motorcycle, rider) in enumerate(zip(all_motocycles, all_pedestrians)):
        bike_rider_mesh = point_e.generate_exception(rider, motorcycle)
        out_filename = os.path.join(folder, "motorcyclist", f"{i}.obj")
        save_mesh(bike_rider_mesh, out_filename)
        print(i)

    # Create bicyclists
    random.shuffle(all_pedestrians)
    for i, (bike, rider) in enumerate(zip(all_bicycles, all_pedestrians)):
        bike_rider_mesh = point_e.generate_exception(rider, bike)
        out_filename = os.path.join(folder, "bicyclist", f"{i}.obj")
        save_mesh(bike_rider_mesh, out_filename)
        print(i)
