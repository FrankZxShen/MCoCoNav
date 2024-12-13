# MCoCoNav
## 📝 TODO List

 - [ ] Release demo video
 - [ ] Improve repository information
## Installation

The code has been tested with Python 3.10.8, CUDA 12.1.

### 1. Installing Dependencies
- We use adjusted versions of [habitat-sim](https://github.com/facebookresearch/habitat-sim) and [habitat-lab](https://github.com/facebookresearch/habitat-lab) as specified below.

- Install habitat-sim:
```
git clone https://github.com/facebookresearch/habitat-sim.git
cd habitat-sim; git checkout tags/challenge-2022; 
pip install -r requirements.txt; 
python setup.py install --headless
```

- Install habitat-lab:
```
git clone https://github.com/facebookresearch/habitat-lab.git
cd habitat-lab; git checkout tags/challenge-2022; 
pip install -e .
```

Back to the current repo, and replace the habitat folder in habitat-lab repo for the multi-robot setting: 

```
mv -r multi-robot-setting/habitat enter-your-path/habitat-lab
```

- Install [pytorch](https://pytorch.org/) according to your system configuration. The code is tested on torch v2.0.1, torchvision 0.15.2. 

- Install [detectron2](https://github.com/facebookresearch/detectron2/) according to your system configuration.

### 2. Download HM3D_v0.2 and MP3D datasets

#### Habitat Matterport
Download [HM3D_v0.2](https://aihabitat.org/datasets/hm3d/) and [MP3D](https://niessner.github.io/Matterport/) datasets using the download utility and [instructions](https://github.com/facebookresearch/habitat-sim/blob/main/DATASETS.md).

### 3. Download segmentation model

Download the [segmentation model](https://drive.google.com/file/d/1U0dS44DIPZ22nTjw0RfO431zV-lMPcvv/view?usp=share_link) in RedNet/model path.

### 4. Install YOLOv10

Follow the [README](detect/README.md) to install YOLOv10.

### 5. Install VLM

We recommend recreating an environment to install VLM.

- Install CogVLM2:
```
git clone https://github.com/THUDM/CogVLM2.git
cd basic_demo
pip install -r requirements.txt
cd enter-your-path-of-MCoCoNav
mv VLM/glm4_openai_api_demo_1gpu.py CogVLM2/basic_demo/
```
- Download GLM-4V-9B:
[🤗 Huggingface](https://huggingface.co/THUDM/glm-4v-9b)
[🤖 ModelScope](https://modelscope.cn/models/ZhipuAI/glm-4-9b-chat-1m)
[💫 Wise Model](https://wisemodel.cn/models/ZhipuAI/GLM-4V-9B)

## Setup
Install other requirements:
```
cd MCoCoNav/
pip install -r requirements.txt
```

### Setting up datasets
The code requires the datasets in a `data` folder in the following format (same as habitat-lab):
```
MCoCoNav/
  data/
    scene_datasets/
        hm3d_v0.2/
            val/
            hm3d_annotated_basis.scene_dataset_config.json
            hm3d_annotated_val_basis.scene_dataset_config.json
        mp3d/
    matterport_category_mappings.tsv
    object_norm_inv_perplexity.npy
    versioned_data
    objectgoal_hm3d_v2/
        train/
        val/
        val_mini/
```

## Evaluation
### Start the VLM server:
```
python glm4_openai_api_demo_1gpu.py
```

### Eval 2-robot on HM3D_v0.2: 
```
python main.py -d ./VLM_EXP/multi_hm3d_2-robot/  --num_agents 2 --task_config tasks/multi_objectnav_hm3d.yaml
```
