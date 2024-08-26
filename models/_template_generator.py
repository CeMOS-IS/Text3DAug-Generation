#!/usr/bin/env python
# -*- coding:utf-8 -*-

import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Generator:
    """
    All generators are implemented based on this template.
    This class is unused, but remains here for documentation purposes.
    """

    def __init__(self, *args) -> None:
        return None

    def __call__(self, *args):
        return None

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
