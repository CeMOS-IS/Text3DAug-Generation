#!/usr/bin/env python
# -*- coding:utf-8 -*-

import argparse
import pathlib
import random

import torch
import yaml

from prompt_list import CLASS_PROMPTS, HUMANS


class PromptGeneration:
    """Build a prompt based on the recipe described in the paper.

    The prompt recipe consists of:
    a {context / attribute size} {context / attribute color} {class / class synonym}

    Class: Class names and their synonyms are defined in prompt_list.py.
    Attributes: Attributes are defined in this class, differentiating between humans and objects.

    The list of human classes are defined in prompt_list.py. Skin color is not used for humans.
    """

    def __init__(self) -> None:
        self.human_attributes()
        self.object_attributes()
        self.reset()

    def reset(self):
        self.prompt_collection = []

    def human_attributes(self):
        """Defines the attribute size for humans."""
        self.human_sizes = [
            "tall",
            "average-sized",
            "small",
            "fat",
            "skinny",
        ]

    def object_attributes(self):
        """Defines the attribute color for objects"""
        self.colors = [
            "red",
            "green",
            "blue",
            "yellow",
            "orange",
            "purple",
            "brown",
            "pink",
            "black",
            "gray",
            "white",
            "violet",
        ]
        self.shades = [
            "dark",
            "light",
            "pastel",
        ]
        self.sizes = [
            "small",
            "medium-sized",
            "large",
        ]

    def object_template(self, synonyms):
        color = random.choice(self.colors)
        shade = random.choice(self.shades)
        size = random.choice(self.sizes)
        obj = random.choice(synonyms)

        prompt = f"A {size} {shade} {color} {obj}"
        return prompt.lower()

    def human_template(self, synonyms):
        size = random.choice(self.human_sizes)
        human = random.choice(synonyms)

        prompt = f"A {size} {human}"
        return prompt.lower()

    @staticmethod
    def save_prompts(prompt_dict, save_filename):
        """Takes a dictionary of prompts and saves them as a .yaml file.

        Args:
            prompt_dict (dict): Key = class, value = list[prompts]
            save_filename (str): .yaml filename to save to

        Raises:
            NotADirectoryError: Folder of filename does not exist.
        """
        # Check directory of save path
        save_filename = pathlib.Path(save_filename).absolute()
        if not save_filename.parent.is_dir():
            raise NotADirectoryError()

        # Dump dictionary to yaml
        with open(save_filename, "w") as prompt_yaml:
            yaml.dump(prompt_dict, prompt_yaml, sort_keys=False)

        print(
            f"Prompt dictionary with keys {prompt_dict.keys()} saved to {save_filename}."
        )

    def generate(self, synonyms, iterations, type="object"):
        """_summary_

        Args:
            synonyms (list): List of class synonyms
            iterations (int): Number of prompts to generate per class.
            type (str, optional): Create prompt for "human" or "object". Defaults to "object".

        Raises:
            NotImplementedError: Prompt type not supported. type != object or human

        Returns:
            list: list of prompts
        """

        # Get prompt function
        if type == "object":
            prompt_fx = self.object_template
        elif type == "human":
            prompt_fx = self.human_template
        else:
            raise NotImplementedError(f"Object of type {type} not implemented.")

        # Generate prompts and add to list
        self.reset()
        prompt_list = [prompt_fx(synonyms) for _ in range(iterations)]
        self.prompt_collection.extend(prompt_list)

        return self.prompt_collection


def cli_parsing():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--out",
        type=str,
        required=True,
        help="Path to output .yaml file",
    )
    parser.add_argument(
        "--nr",
        type=int,
        required=False,
        help="Number of prompts to generate per class",
        default=1000,
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = cli_parsing()

    prompt_gen = PromptGeneration()
    prompt_dict = {}

    for key, synonyms in CLASS_PROMPTS.items():
        if key in HUMANS:
            prompt_type = "human"
        else:
            prompt_type = "object"

        prompt_list = prompt_gen.generate(
            synonyms,
            iterations=args.nr,
            type=prompt_type,
        )

        prompt_dict[key] = prompt_list  # {key = class: value = list[prompts]}

    # Save all prompts into a single .yaml file.
    prompt_gen.save_prompts(prompt_dict, args.out)
