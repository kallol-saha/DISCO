<div align="center">

# 🪩 DISCO: Distilled Tree Search over Continuous Actions
### CoRL 2026 (Oral Presentation - top 1%)

[Kallol Saha](https://kallol-saha.github.io/)<sup>1</sup>, [Yining Hong](https://evelinehong.github.io/)<sup>2</sup>, [Amber Li](https://amburger66.github.io/)<sup>1</sup>, [Yejin Choi](https://profiles.stanford.edu/yejin-choi)<sup>2</sup>, [Maxim Likhachev](https://www.cs.cmu.edu/~maxim/)<sup>1</sup>, [David Held](https://davheld.github.io/)<sup>1</sup><br>
<sup>1</sup>Robotics Institute, Carnegie Mellon University &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <sup>2</sup>Stanford University

</div>

<p align="center">
    <!-- <a href='ARXIV_URL'>
      <img src='https://img.shields.io/badge/Paper-arXiv-green?style=plastic&logo=arXiv&logoColor=green' alt='Paper arXiv'>
    </a> -->
    <a href='https://distilled-search.github.io/'>
      <img src='https://img.shields.io/badge/Project-Page-blue?style=plastic&logo=Google%20chrome&logoColor=blue' alt='Project Page'>
    </a>
    <a href='https://huggingface.co/kallol-saha/disco'>
      <img src='https://img.shields.io/badge/Models-Hugging%20Face-yellow?style=plastic&logo=huggingface&logoColor=yellow' alt='Models'>
    </a>
    <a href='https://huggingface.co/datasets/kallol-saha/disco-data'>
      <img src='https://img.shields.io/badge/Data-Hugging%20Face-yellow?style=plastic&logo=huggingface&logoColor=yellow' alt='Data'>
    </a>
</p>

https://github.com/user-attachments/assets/02609cb7-f2dc-456d-911f-4d8de2a05f4a

We present **DISCO** (**Di**stilled Tree **S**earch over **Co**ntinuous Actions), which transfers
long-horizon reasoning from search into policies over continuous actions. DISCO runs progressive-widening
MCTS in physics simulation: a learned policy proposes continuous actions, and a learned value function
scores them by their expected progress toward task completion. Using GPU-parallelized simulation, DISCO
periodically distills the plans found by search into the policy-value model, improving the proposals and
value estimates of the next search. The learned policy-value model can then be deployed directly, without
simulator search at test time.

---
<br>

This is the official repository of [DISCO](https://distilled-search.github.io/). If you find our work useful, please
consider citing our paper:
```bibtex
@inproceedings{saha2026disco,
  title     = {Distilled Tree Search over Continuous Actions},
  author    = {Saha, Kallol and Hong, Yining and Li, Amber and Choi, Yejin and Likhachev, Maxim and Held, David},
  booktitle = {Conference on Robot Learning (CoRL)},
  year      = {2026},
}
```
If you find any bugs in the code, or have any questions, feel free to raise an issue.

# Table of Contents

- [Installation](#installation)
- [Download Models and Data](#download-models-and-data)
- [Quick Demo](#quick-demo)
- [Running DISCO](#running-disco)
   * [Pre-training the initial policy](#pre-training-the-initial-policy)
   * [Search and distillation](#search-and-distillation)
- [Repository Layout](#repository-layout)
- [Acknowledgements](#acknowledgements)

# Installation

Tested on Linux with Python 3.10, CUDA 12.6 and 2× RTX 4090. cuRobo is compiled locally, so the CUDA
toolkit (`nvcc` 12.x) must be on your `PATH`.

### Clone the repository and cuRobo:
```bash
git clone --recursive https://github.com/kallol-saha/disco.git
cd disco
```
`curobo/` is a submodule: our fork, pinned to upstream cuRobo commit `ebb7170` and otherwise
unmodified. If you cloned without `--recursive`, run `git submodule update --init`.

### Install the dependencies:
```bash
conda create -n disco python=3.10 -y && conda activate disco
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
pip install -e curobo --no-build-isolation     # builds cuRobo's CUDA kernels
```

# Download Models and Data

The train/test environments, the rollouts used by the demo, and the object meshes and grasps are
included in the repository. The trained networks and the larger data are on Hugging Face; run these
from the repository root:

```bash
# DISCO policies and value functions (38 MB) -> models/
hf download kallol-saha/disco --include "models/*" --local-dir .

# Pre-training demonstrations (280 MB) -> assets/processed_data/
hf download kallol-saha/disco-data --repo-type dataset --include "assets/*" --local-dir .

# Optional: MCTS search trees of the train environments (850 MB) -> assets/trees/
hf download kallol-saha/disco-data --repo-type dataset --include "archives/*" --local-dir .
for f in archives/*.tar.gz; do tar -xzf "$f"; done && rm -r archives
```

# Quick Demo

Execute the DISCO policy-value rollouts on the test environments and print the success rate:

```bash
python demo.py --task shelf --split test
python demo.py --task stacking --split test
```

Add `--env <i>` to run a single environment, or use `--split train`.

# Running DISCO

### Pre-training the initial policy:
Behavior cloning of the grasp and place policies on single-step demonstrations:
```bash
python pretrain.py                             # -> models/pretrain/P0_{grasp,place}/last.pth
```

### Search and distillation:
Each round searches 25 MCTS iterations per train environment from scratch, then distills the policy
and value function on those trees, until the simulation budget is spent:
```bash
python run_disco.py                            # shelf packing
python run_disco.py task=stacking              # object stacking
```

By default the search starts from the released DISCO policies in `models/<task>/`. Pass
`grasp=... place=...` to start from another policy (for example the output of `pretrain.py`), and
`value=models/<task>/value.pth` to start with a value function.

Settings live in `config/run_disco.yaml` and can be overridden on the command line, e.g.
`python run_disco.py envs=[0,1] budget=5000 wandb=true`. MCTS parameters are in `config/mcts.yaml`.
Outputs go to `results/disco_<task>/`, and re-running the same command resumes. The simulator always
runs 32 parallel environments, and `run_disco.py` runs at most 4 search jobs per GPU.

# Repository Layout

```
disco/
├── demo.py                        # execute DISCO policy-value rollouts, report the success rate
├── pretrain.py                    # pre-train the initial grasp and place policies
├── run_disco.py                   # rounds of MCTS search + distillation
├── config/
│   ├── run_disco.yaml             # run_disco settings (task, checkpoints, budget, rounds)
│   ├── mcts.yaml                  # MCTS parameters
│   └── curobo/franka.yml          # cuRobo robot config
├── disco/
│   ├── search/
│   │   ├── run.py                 # search one environment
│   │   ├── mcts.py                # tree, progressive widening, UCB descent, backup
│   │   ├── expansion.py           # expand nodes: propose, execute, reward
│   │   ├── candidates.py          # plan and execute grasp-place candidates in 32 parallel sims
│   │   └── budget.py              # simulation budget shared by all search processes
│   ├── data.py                    # search trees -> policy and value datasets
│   ├── tasks.py                   # what differs between shelf packing and object stacking
│   ├── policy/                    # load and sample the policies and value function
│   ├── flowmatch/                 # 3D FlowMatch Actor networks and trainers
│   ├── envs/                      # ManiSkill shelf packing and object stacking environments
│   ├── motion/                    # cuRobo motion planner, pick and place primitives
│   └── utils/                     # point clouds, poses, visualization
├── assets/
│   ├── environments/              # train/test scenes per task
│   ├── rollouts/                  # DISCO policy-value rollouts used by the demo
│   ├── acronym_objects/           # object meshes
│   ├── final_grasps/              # object grasps
│   ├── processed_data/            # pre-training data (downloaded)
│   └── trees/                     # MCTS search trees (downloaded, optional)
├── models/                        # DISCO policies and value functions (downloaded)
├── curobo/                        # cuRobo submodule
└── tests/
```

# Acknowledgements

The policy and value networks build on 3D FlowMatch Actor (Gkanatsios et al., 2025; `disco/flowmatch`,
MIT license, see `disco/flowmatch/LICENSE`). We also use [cuRobo](https://github.com/NVlabs/curobo),
[ManiSkill](https://github.com/haosulab/ManiSkill) and objects from
[ACRONYM](https://sites.google.com/nvidia.com/graspdataset).
