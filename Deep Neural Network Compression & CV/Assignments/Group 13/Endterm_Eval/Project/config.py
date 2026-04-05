import torch
import os
from pathlib import Path

def config_device():
    device = None
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    #print(f"Using device: {device}")
    return device

# Automatically configure and store the device when config is imported
DEVICE = config_device()

# ==========================================
# Data Settings
# ==========================================
# Use pathlib to create a robust, absolute path for the dataset
PROJECT_ROOT = Path(__file__).parent.absolute()
DATA_DIR = PROJECT_ROOT / 'data' / 'data_files'

BATCH_SIZE = 64           # Keeping this at 64 is safe for memory
NUM_WORKERS = 2           # Helps load data faster

# ==========================================
# Training Hyperparameters
# ==========================================
LEARNING_RATE = 0.001
FINETUNE_LR = 0.0001      # NEW: 10x smaller for delicate fine-tuning
CENTROID_LR = 0.00005     # NEW: Extremely small for Stage 2.5 shifts

EPOCHS = 10             # INCREASED: Better baseline = better compressed model

# ==========================================
# Compression Targets (from project specs)
# ==========================================
TARGET_SPARSITY = 0.90    # 90% weights removed
K_MEANS_CLUSTERS = 32     # Representing weights with 5 bits (2^5 = 32)