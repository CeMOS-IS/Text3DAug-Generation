#!/usr/bin/env python
# -*- coding:utf-8 -*-

import argparse
import glob
import os
import random
import urllib

import numpy as np
import open3d as o3d
import torch
from cap3d.shap_e.diffusion.gaussian_diffusion import diffusion_from_config
from cap3d.shap_e.diffusion.sample import sample_latents
from cap3d.shap_e.models.configs import model_from_config
from cap3d.shap_e.util.notebooks import decode_latent_mesh
from shap_e.models.download import load_config, load_model

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
    weight_fn = os.path.join(weight_dir, "cap3d_shap_e.pt")
    if not os.path.isfile(weight_fn):
        dl_url = "https://huggingface.co/datasets/tiange/Cap3D/resolve/main/misc/our_finetuned_models/shapE_finetuned_with_330kdata.pth?download=true"
        urllib.request.urlretrieve(dl_url, weight_fn)
        print("Cap3D Shap-E weights complete")

    # Load and return weights
    return torch.load(weight_fn, map_location=DEVICE)


class Cap3DShapE:
    def __init__(self) -> None:
        self.xm = load_model("transmitter", device=DEVICE)
        self.model = model_from_config(load_config("text300M"), device=DEVICE)
        # self.model = load_model("text300M", device=DEVICE)
        self.diffusion = diffusion_from_config(load_config("diffusion"))
        self.model.load_state_dict(get_and_load_weights()["model_state_dict"])

    def __call__(self, prompt: str, debug: bool = False) -> o3d.geometry.TriangleMesh:
        """Generates a mesh from a text prompt using model.

        Args:
            prompt (str): A single prompt
            debug (bool, optional): Visualize for debugging. Defaults to False.

        Returns:
            o3d.geometry.TriangleMesh: Output mesh as open3d object.
        """
        prompt = self.preprocess(prompt)

        latents = sample_latents(
            batch_size=1,
            model=self.model,
            diffusion=self.diffusion,
            guidance_scale=15.0,
            model_kwargs=prompt,
            progress=True,
            clip_denoised=True,
            use_fp16=True,
            use_karras=True,
            karras_steps=64,
            sigma_min=1e-3,
            sigma_max=160,
            s_churn=0,
        )

        for mesh in latents:
            mesh = decode_latent_mesh(self.xm, mesh).tri_mesh()

        # Mesh the pointcloud with marching cubes
        mesh = self.unify_coordinates(mesh)
        o3d_mesh = self.post_process(mesh)

        if debug:
            self._visualize(sample=o3d_mesh)
        return o3d_mesh

    def preprocess(self, prompt):
        """
        Put the prompt in the proper format for model.
        """
        return dict(texts=[prompt])

    def generate_exception(self, rider: str, bike: str):
        rider_mesh = o3d.io.read_triangle_mesh(rider, enable_post_processing=True)
        bike_mesh = o3d.io.read_triangle_mesh(bike, enable_post_processing=True)

        # Move rider up, so they are sitting on the bipedal
        vertices = np.copy(rider_mesh.vertices)
        vertices[:, 2] += 0.25

        rider_mesh.vertices = o3d.utility.Vector3dVector(vertices)

        bike_mesh += rider_mesh  # Add person to bike
        bike_mesh = self.unify_coordinates(bike_mesh, o3d_mesh=True)
        # o3d.visualization.draw_geometries([bike_mesh])  # NOTE: For debugging

        # Check for NANs
        vcheck = np.copy(bike_mesh.vertices)[:, :3]
        if np.isnan(vcheck).any():
            return None
        return bike_mesh

    def transform(self, *args):
        # Currently unused
        return None

    def unify_coordinates(self, mesh, o3d_mesh: bool = False):
        """
        Scales height of instance into 0..1 while maintaining aspect ratio.
        Center of object is at (0, 0, 0.5)
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
        Converts the TriMesh into an Open3D mesh.
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

    def _visualize(self, sample, _=None):
        o3d.visualization.draw_geometries([sample])


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
    GENERATE_EXCEPTIONS = True
    args = cli_parsing()

    # For testing
    shap_e = Cap3DShapE()

    if not GENERATE_EXCEPTIONS:
        x = shap_e("motorcycle", debug=True)
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

        out_filename = os.path.join(folder, "motorcyclist", f"{i}.obj")
        bike_rider_mesh = shap_e.generate_exception(rider, motorcycle)

        while not bike_rider_mesh:
            # Deal with NAN
            rider = random.choice(all_pedestrians)
            motorcycle = random.choice(all_motocycles)
            bike_rider_mesh = shap_e.generate_exception(rider, motorcycle)
            print(i, "NAN")
        save_mesh(bike_rider_mesh, out_filename)

    # Create bicyclists
    random.shuffle(all_pedestrians)
    for i, (bike, rider) in enumerate(zip(all_bicycles, all_pedestrians)):

        out_filename = os.path.join(folder, "bicyclist", f"{i}.obj")
        bike_rider_mesh = shap_e.generate_exception(rider, bike)

        while not bike_rider_mesh:
            # Deal with NAN
            rider = random.choice(all_pedestrians)
            bike = random.choice(all_bicycles)
            bike_rider_mesh = shap_e.generate_exception(rider, bike)
            print(i, "NAN")
        save_mesh(bike_rider_mesh, out_filename)
