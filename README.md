# CRAFTS

CRAFTS is a recommendation framework that learns transferable user-profile and functional skills from historical interactions and uses context-relevant skills to support candidate ranking. It further employs skill-guided trajectory supervision to enable adaptive skill utilization, while failed recommendation trajectories are used to refine the Skill Bank.

## Project Structure

```text
CRAFTS/
├── conf/                        # Command-line configurations
├── data/clean/                  # Processed datasets
├── recommender/
│   ├── agent/                   # Recommendation agent and skill modules
│   └── sequential/              # Sequential recommender
├── tool_conf/                   # Tool-model configurations
├── util/                        # Data processing and evaluation utilities
├── scripts/                     # Main experiment scripts
├── main.py                      # Main entry point
├── train_skill_distill_lora.py  # Trajectory-supervised SFT
└── requirements.txt             # Python dependencies
```

## Environment Setup

```bash
# Create conda environment
conda create -n crafts python=3.10
conda activate crafts

# Install dependencies
pip install -r requirements.txt
```

## Dataset

Download the datasets from [Dataset Download](https://drive.google.com/drive/folders/1ahiLmzU7cGRPXf5qGMqtAChte2eYp9gI).

After downloading, extract the datasets into the `data/clean/` directory.

```text
data/
└── clean/
    └── Instruments/
```

## Run CRAFTS

The complete main-experiment pipeline is provided in:

```text
scripts/run_crafts.sh
```

This script contains the commands for Skill Bank construction and refinement, trajectory-supervised training, inference, and evaluation.
