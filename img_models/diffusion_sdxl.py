#!/usr/bin/env python
# -*- coding:utf-8 -*-

import torch
from diffusers import AutoPipelineForText2Image

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SDXL:
    def __init__(self, *args) -> None:
        self.pipe = AutoPipelineForText2Image.from_pretrained(
            "stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16"
        )
        self.pipe.to(DEVICE)

    def __call__(self, prompt: str, debug: bool = False):
        image = self.pipe(
            prompt=prompt, num_inference_steps=4, guidance_scale=0.0
        ).images[0]

        if debug:
            image.show()
        return image

    def preprocess(self, *args):
        return None

    def generate(self, prompt):
        return None

    def generate_exception(self, prompt):
        return None

    def transform(self, *args):
        return None

    def mesh(self, *args):
        return None

    def unify_coordinates(self, *args):
        return None

    def post_process(self, *args):
        return None

    def _visualize(self, *args):
        return None
