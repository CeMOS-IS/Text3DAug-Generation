#!/usr/bin/env python
# -*- coding:utf-8 -*-
import argparse
import pathlib

import yaml
from img_models import SDXL
from tqdm import tqdm


def get_model(name):
    name = name.lower()  # all to lowercase
    if name in ["sdxl"]:
        model = SDXL()
    else:
        raise NotImplementedError(name)

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

    # Make the folder if it does not exists
    new_folder = parent_folder / new_child_folder
    new_folder.mkdir(parents=True, exist_ok=True)

    return new_folder


def sdxl_prompt(prompt: str, class_name: str) -> str:
    """
    Changes the prompt to get better results from SDXL.
    Otherwise objects are only shown partially.
    """
    if class_name.lower() == "pedestrian":
        prompt = f"A single person, {prompt}, in a full-body shot against a white background without any cutoffs. Their shoes and head have to be visible."
    else:
        prompt += ". The entire subject should be visible."
    return prompt


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
        required=False,
        default="sdxl",
        help="Name of text-to-image network. SDXL.",  # SDXL used in paper
    )
    return parser.parse_args()


if __name__ == "__main__":
    #
    # Text -> Image generation with Stable Diffusion XL Turbo.
    # Required for Image -> 3D model One-2-3-45
    #

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

        # Go through each prompt
        for idx, p in enumerate(tqdm(prompts)):
            p = sdxl_prompt(p, class_name)
            image = generator(prompt=p)

            out_filepath = class_folder / f"{idx}.png"
            image.save(out_filepath)
