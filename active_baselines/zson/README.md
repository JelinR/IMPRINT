# ZSON: Zero-Shot Object-Goal Navigation using Multimodal Goal Embeddings

**Paper Reference:** [Link](https://arxiv.org/abs/2206.12403)

**Author:** Arjun Majumdar*, Gunjan Aggarwal*, Bhavika Devnani, Judy Hoffman and Dhruv Batra

**Source Repo:** [Link](https://github.com/gunagg/zson)

## Set up Habitat-Lab

ZSON uses the Habitat challenge-2022 software stack. Set it up from the
IMPRINT repository with the matching Habitat-Lab checkout:

```bash
# Enter the ZSON baseline directory
cd IMPRINT/active_baselines/zson

# Clone Habitat-Lab challenge-2022 in a sibling directory.
mkdir -p habitat-labs/challenge-2022
cd habitat-labs/challenge-2022
git clone --branch challenge-2022 https://github.com/facebookresearch/habitat-lab.git
cd habitat-lab
pip install -e .
cd ../../..
ln -s habitat-labs/challenge-2022/habitat-lab habitat-lab
```

## Installation

```bash

# Create conda env
conda create -n imprint_zson python=3.9 cmake=3.14.0 -y
conda activate imprint_zson

# CUDA and Torch
conda install pytorch==1.10.2 torchvision==0.11.3 cudatoolkit=11.3 -c pytorch -c conda-forge

# Habitat-Sim
git clone https://github.com/facebookresearch/habitat-sim.git
cd habitat-sim; git checkout tags/challenge-2022; 
pip install -r requirements.txt 
python setup.py install --headless
cd ..

# Habitat-Lab
cd habitat-lab
pip install -e .
cd ..

# ZSON requirements
pip install -r requirements.txt
python setup.py develop
```

## Download weights

All the required data can be downloaded from [here](https://huggingface.co/gunjan050/ZSON/tree/main).

The following trained checkpoints are to be downloaded into the directory `data/checkpoints`:

  - [`zson_conf_B.pth`](https://huggingface.co/gunjan050/ZSON/resolve/main/zson_conf_B.pth)


Download the models weights into `data/models/`:

   - [omnidata_DINO_02.pth](https://huggingface.co/gunjan050/ZSON/resolve/main/omnidata_DINO_02.pth)


## Evaluation

ZSON is a policy baseline for ObjectNav. Configure the dataset and scene paths
in an experiment file under `configs/experiments/`, then run from the ZSON
directory:

```bash
python run.py \
  --run-type eval \
  --exp-config configs/experiments/objectnav_mp3d.yaml \
  EVAL_CKPT_PATH_DIR=data/checkpoints \
  NUM_ENVIRONMENTS=1
```

For IMPRINT experiments, create or adapt an ObjectNav experiment and task
configuration with:

- The HSSD-rare or OVON episode `DATA_PATH`.
- The corresponding Habitat scene dataset and scene directory.
- `TASK_CONFIG.TASK.SENSORS` containing `OBJECTGOAL_PROMPT_SENSOR`.
- The ZSON checkpoint in `data/checkpoints/`.
- The pretrained visual encoder in `data/models/`.

ZSON consumes the target category prompt directly. It does not use IMPRINT's
web-image query enrichment or semantic-map aggregation; it is retained as the
direct-policy baseline for comparison with the map-based IMPRINT systems.

Evaluation logs and videos are written to the directories configured by
`LOG_DIR`, `VIDEO_DIR`, and `TENSORBOARD_DIR`.

## Source Citation

```
@inproceedings{majumdar2022zson,
  title={ZSON: Zero-Shot Object-Goal Navigation using Multimodal Goal Embeddings},
  author={Majumdar, Arjun and Aggarwal, Gunjan and Devnani, Bhavika and Hoffman, Judy and Batra, Dhruv},
  booktitle={Neural Information Processing Systems (NeurIPS)},
  year={2022}
}
```
