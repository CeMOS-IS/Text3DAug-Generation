#!/usr/bin/env python
# -*- coding:utf-8 -*-
import argparse
import pathlib

import yaml
from models import Cap3DPointE, Cap3DShapE, PointE, PyramidXLPointE, ShapE
from models.mesh_saver import save_mesh
from tqdm import tqdm


def get_model(name):
    name = name.lower()  # all to lowercase

    if name in ["pointe", "point-e", "point_e"]:
        model = PointE()
    elif name in ["shape", "shap-e", "shap_e"]:
        model = ShapE()
    elif name in ["cap3dpointe", "cap3dpoint-e", "cap3dpe"]:
        model = Cap3DPointE()
    elif name in ["cap3dshape", "cap3dshap-e", "cap3dse"]:
        model = Cap3DShapE()
    elif name in ["gpt4point", "gpt", "pyramid", "pyramidxl"]:
        model = PyramidXLPointE()

    return model


def read_prompt_dict(yaml_file):
    yaml_file = pathlib.Path(yaml_file)
    if not yaml_file.is_file():
        raise FileNotFoundError(f"Yaml file {yaml_file} not found.")

    with open(yaml_file, "r") as f:
        prompt_dict = yaml.safe_load(f)

    return prompt_dict


def make_child_folder(parent_folder, new_child_folder):
    parent_folder = pathlib.Path(parent_folder)
    if not parent_folder.is_dir():
        raise NotADirectoryError(f"Directory {parent_folder} not found.")

    # Make the folder if it does not exist
    new_folder = parent_folder / new_child_folder
    new_folder.mkdir(parents=True, exist_ok=True)

    return new_folder


def cli_parsing():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--prompts",
        type=str,
        required=True,
        help="Path to .yaml prompt file.",
    )
    parser.add_argument(
        "--out",
        type=str,
        required=True,
        help="Path to output folders",
    )
    parser.add_argument(
        "--gen",
        type=str,
        required=True,
        help="Name of text-to-3d network, e.g. pointe, shape, cap3dpointe, cap3dshape, gpt4point",
    )
    return parser.parse_args()


if __name__ == "__main__":
    # Get arguments from CLI
    args = cli_parsing()

    # Read prompts and get the generation model
    prompt_dict = read_prompt_dict(args.prompts)
    generator = get_model(args.gen)

    # Make output folder specific to the model
    output_folder = pathlib.Path(args.out).absolute()
    output_folder = make_child_folder(output_folder, args.gen)

    for class_name, prompts in prompt_dict.items():
        # Make output folder specific to the class
        class_folder = make_child_folder(output_folder, class_name)
        print(f"Prompting class {class_name}")

        # Go trough each prompt
        for idx, p in enumerate(tqdm(prompts)):
            out_filepath = class_folder / f"{idx}.obj"

            mesh = generator(prompt=p)
            save_mesh(mesh, out_filepath)
