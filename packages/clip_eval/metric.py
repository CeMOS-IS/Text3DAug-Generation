#!/usr/bin/env python
# -*- coding:utf-8 -*-
import clip
import torch


class ClipScore:
    def __init__(self, model="ViT-B/32") -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.clip_model, self.preprocess = clip.load(model, device=self.device)

    @torch.no_grad()
    def __call__(self, image, text):
        # Preprocess Inputs
        image = self.preprocess(image).unsqueeze(0).to(self.device)
        text = clip.tokenize(text).to(self.device)

        # Get similarity scores
        logits_per_image, _ = self.clip_model(image, text)
        score = logits_per_image.softmax(dim=-1).cpu().numpy()

        return score
