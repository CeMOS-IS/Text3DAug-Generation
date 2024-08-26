# Text3DAug – Prompted Instance Augmentation for LiDAR Perception
:fire: Accepted at IROS 2024 (oral) :fire:

[Laurenz Reichardt](https://scholar.google.com/citations?user=cBhzz5kAAAAJ&hl=en), Luca Uhr, [Oliver Wasenmüller](https://scholar.google.de/citations?user=GkHxKY8AAAAJ&hl=de) \
**[CeMOS - Research and Transfer Center](https://www.cemos.hs-mannheim.de/ "CeMOS - Research and Transfer Center"), [University of Applied Sciences Mannheim](https://www.english.hs-mannheim.de/the-university.html "University of Applied Sciences Mannheim")**

[![arxiv.org](https://img.shields.io/badge/cs.CV-arXiv%3A0000.0000-B31B1B.svg)](https://arxiv.org/)
[![cite-bibtex](https://img.shields.io/badge/Cite-BibTeX-1f425f.svg)](#citing)
[![download meshes](https://img.shields.io/badge/Download-Meshes-b3a017.svg)](https://shields.io/)

## About - Mesh Generation
This repository prompts text to 3D models in order to create meshes for the Text3DAugmentation.
For the augmentation itself, please refer to the following [repository](https://github.com/CeMOS-IS/Text3DAug-Augmentation).

## Installation
The code was tested with CUDA 11.8 and Python 3.8.10. The code was not optimized for multi-GPU setups and various models only support 1 GPU.

**Anaconda**

To set up an Anaconda environment, run the following commands from */mesh_generation*:
```
conda create -n text3daug_meshgen python=3.8.10
```
```
export PYTHONNOUSERSITE=True && conda activate text3daug_meshgen
```
```
conda install pip --yes && python -m pip install --upgrade pip
```
```
bash ./setup/local_setup.sh
```

**Docker**

To build a Docker image, run the following from */mesh_generation*:
```
docker build -t text3daug_generation .
```

Adjust visible GPUs, shared memory size and mounted local directory according to your setup, then run the container.:
```
docker run -it --rm --gpus '"device=0"' --shm-size 100g -v {LOCAL PATH}:/home/mesh_output text3daug_generation bash
```
Some packages are installed after running the container for the first time, which will take some time.

**Local**

For a local installation, without Anaconda or Docker, just run:
```
bash ./setup/local_setup.sh
```

## Creating Prompts
Prompts are created based on the prompt recipe described in the paper. To create a dictionary of prompts (and as such comparable prompts across models), run:
```
python3 ./prompting/prompt_generation.py --out ./prompt_dict.yaml --nr {NUMBER OF PROMPTS}
```
Adjust flags *- -nr *and *- -out *according to the required number of prompts and desired output *.yaml* path respectively.

Classes and their synonyms are defined in *./prompting/prompt_list.py*. This file also defines human classes, which changes the prompt recipe slightly. Prompt context / attributes are defined in *./prompting/prompt_generation.py*.
Classes and Context match those in the paper, overlapping classes used in the SemanticKITTI and NuScenes datasets.

## Generating Meshes
Meshes are defined in a uniform coordinate system, with the maximum mesh height scaled to 1. Mesh width and length are scaled proportionately, meaning that the center of a mesh is at (0, 0, 0.5).


**Text &rarr; Image &rarr; Mesh** ([One-2-3-45](https://one-2-3-45.github.io/))

To generate meshes with this model, take the following steps.
Generate Text->Image using [SDXL-Turbo](https://huggingface.co/stabilityai/sdxl-turbo) using the following command, adjusting the flags *- -prompts* and *- -out* accordingly. Each generated *.png* image will be stored in *- - out*, with the class name as a subdirectory.
```
mkdir ./prompted_images
```
```
python3 generate_img.py --prompts ./prompt_dict.yaml --out ./prompted_images --gen sdxl
```

Navigate into the One-2-3-45 folder and download the necessary weights:
```
cd ./packages/One-2-3-45/ && python3 download_ckpt.py
```

From this directory, run the model on each generated image subdirectory, matching the class name. The first argument points to the generated image folder, the second to the class subdirectory. E.g. for the class "car", output meshes are saved in */packages/One-2-3-45/exp*.
```
bash run_on_folder.sh ../../prompted_images/sdxl car
```

Finally, post process the generated meshes into the uniform coordinate system. For this, return to the main directory *mesh_generation* and run:
```
cd ../../ && python3 post_process_one2345.py --folder ./packages/One-2-3-45/exp/
```


**Text &rarr; Mesh** ([Point-E](https://github.com/openai/point-e), [Shap-E](https://github.com/openai/shap-e), [Cap3D](https://github.com/crockwell/Cap3D), [GPT4Point](https://github.com/Pointcept/GPT4Point))

Generate meshes with the following command, using *Point-E* as an example. Adjust the flags *- -prompts*, *- - out*, and *- -gen* as needed:
```
python3 generate_mesh.py --prompts ./prompting/prompt_dict.yaml --out ./mesh_output/ --gen pointe
```

These Text &rarr; Mesh models do not natively support the classes *cyclist* and *motorcyclist*. This stems from the limited amount of classes in their training data. These exception classes are handled by merging the meshes of *person* with *bicycle* meshes or *person* with *motorcycle*. After generating meshes, stitch together the exception classes with the following command:
```
cd models && python3 openai_point_e.py --folder ../mesh_output/pointe/
```


**Scaling Prior To Training**

The height of a mesh is randomly scaled from 1. to a realistic height. Scale the meshes prior to augmenting the training pipeline with the following command, adjusting the mesh folder accordingly:
```
python3 scale_instances --folder ./mesh_output/pointe/
```

## Evaluating Meshes
**Viewing Meshes**

We include a simple Open3D-based visualizer that can be run with the following command. Adjust *- -folder* as needed:
```
python3 viewer.py --folder ./mesh_output/pointe/
```


**Clip Scoring**

Our proposed CLIP scoring of mesh quality compares the class names and void-classes to rendered grayscale images of the surface-normal shaded mesh. The classes are defined in *evaluate.py*. To evaluate generated meshes, run:
```
python3 evaluate.py --folder ./mesh_output/pointe/ --out_path ./mesh_output/clipeval.txt
```
Adjust the flags *- -folder* and *- -out_path* accordingly. Set the flag *- -debug* to show the images used for CLIP scoring. The mesh path and corresponding CLIP score are saved as a *.txt* file. This *.txt* file was used in the paper to filer the top-*n* meshes for quality vs. quantity experiments.

The resulting generated meshes and evaluation file should now be in a data format similar to this:
```
/mesh_output/
        ├── pointe/
               ├── car/
                    ├── 0.obj
                    ├── 1.obj
                    ├── ...
               ├── motorcycle/
                    ├── ...
               ├── .../
               .
               .
               .
               ├── clip.txt
```

## Licenses and Acknowledgements
We sincerely appreciate the great contribution of the following works and acknowledge their use in this repository. As such, the licenses of the original work / repositories have to be respected. The original licenses are contained where applicable in the respective folders of this repository.
- [Point-E](https://github.com/openai/point-e): MIT
- [Shap-E](https://github.com/openai/shap-e):MIT
- [Cap3D](https://github.com/crockwell/Cap3D): MIT
- [GPT4Point](https://github.com/Pointcept/GPT4Point): MIT
- [SDXL-Turbo](https://huggingface.co/stabilityai/sdxl-turbo): StabilityAI non-commercial
- [One-2-3-45](https://one-2-3-45.github.io/): Apache 2.0
- [CLIP](https://github.com/openai/CLIP): MIT

## Citing
If you have used Text3DAug in your research, please cite our work. :mortar_board: 
```bibtex
@inproceedings{reichardt2024text3daug,
    title = {Text3DAug – Prompted Instance Augmentation for LiDAR Perception},
    author = {Reichardt, Laurenz and Uhr, Luca and Wasenm{\"u}ller, Oliver},
    booktitle = {International Conference on Intelligent Robots and Systems (IROS)},
    year = {2024},
}
```
