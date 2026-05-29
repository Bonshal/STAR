import nbformat as nbf
import os

nb = nbf.v4.new_notebook()

# Read contents of files
def get_file_contents(filename):
    with open(filename, 'r') as f:
        return f.read()

# Cells
cells = []

cells.append(nbf.v4.new_markdown_cell("# Multi-GNN Standalone Training Notebook\n\nThis notebook contains the complete training code for the Multi-GNN architecture. All python files have been merged into this single notebook for easier execution on Google Colab."))

cells.append(nbf.v4.new_code_cell('''# 1. Install Dependencies
!pip install datatable munch wandb tabulate
import torch
TORCH_VERSION = torch.__version__.split('+')[0]
CUDA_VERSION = torch.version.cuda.replace('.', '')
!pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-{TORCH_VERSION}+cu{CUDA_VERSION}.html'''))

cells.append(nbf.v4.new_code_cell('''# 2. Imports
import argparse
import numpy as np
import torch
import random
import logging
import os
import sys
import json
import itertools
import pandas as pd
import tqdm
from sklearn.metrics import f1_score
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, HeteroData
from torch_geometric.typing import OptTensor
from torch_geometric.nn import GINEConv, BatchNorm, Linear, GATConv, PNAConv, RGCNConv
from torch_geometric.transforms import BaseTransform
from typing import Union
from torch_geometric.loader import LinkNeighborLoader
from torch_geometric.nn import to_hetero, summary
from torch_geometric.utils import degree
import wandb'''))

# Data Utils
cells.append(nbf.v4.new_markdown_cell("## Data Utilities"))
cells.append(nbf.v4.new_code_cell(get_file_contents('data_util.py').replace('from torch_geometric.data import Data, HeteroData\nfrom torch_geometric.typing import OptTensor\nimport numpy as np\n', '').replace('import torch\n', '')))

# Models
cells.append(nbf.v4.new_markdown_cell("## Model Architectures"))
cells.append(nbf.v4.new_code_cell(get_file_contents('models.py').replace('import torch.nn as nn\nfrom torch_geometric.nn import GINEConv, BatchNorm, Linear, GATConv, PNAConv, RGCNConv\nimport torch.nn.functional as F\nimport torch\nimport logging\n', '')))

# Train Utils
cells.append(nbf.v4.new_markdown_cell("## Training Utilities"))
cells.append(nbf.v4.new_code_cell(get_file_contents('train_util.py').replace('import torch\nimport tqdm\nfrom torch_geometric.transforms import BaseTransform\nfrom typing import Union\nfrom torch_geometric.data import Data, HeteroData\nfrom torch_geometric.loader import LinkNeighborLoader\nfrom sklearn.metrics import f1_score\nimport json\n', '')))

# Data Loading
cells.append(nbf.v4.new_markdown_cell("## Data Loading"))
cells.append(nbf.v4.new_code_cell(get_file_contents('data_loading.py').replace('import pandas as pd\nimport numpy as np\nimport torch\nimport logging\nimport itertools\nfrom data_util import GraphData, HeteroData, z_norm, create_hetero_obj\n', '')))

# Training
cells.append(nbf.v4.new_markdown_cell("## Training Loops"))
cells.append(nbf.v4.new_code_cell(get_file_contents('training.py').replace('import torch\nimport tqdm\nfrom sklearn.metrics import f1_score\nfrom train_util import AddEgoIds, extract_param, add_arange_ids, get_loaders, evaluate_homo, evaluate_hetero, save_model, load_model\nfrom models import GINe, PNA, GATe, RGCN\nfrom torch_geometric.data import Data, HeteroData\nfrom torch_geometric.nn import to_hetero, summary\nfrom torch_geometric.utils import degree\nimport wandb\nimport logging\n', '')))

# Setup and main
main_code = '''
## Main Setup and Execution

# Configuration
class Args:
    pass

args = Args()
args.data = "Small_LI"
args.model = "gin"
args.batch_size = 8192
args.n_epochs = 100
args.num_neighs = [100, 100]
args.seed = 1
args.tqdm = True
args.emlps = True
args.reverse_mp = True
args.ports = True
args.ego = True
args.tds = False
args.save_model = False
args.unique_name = "test"
args.finetune = False
args.testing = True # Disables wandb logging

data_config = {
    "paths": {
        "aml_data": "./data",
        "model_to_save": "./models",
        "model_to_load": "./models"
    }
}

# Write model_settings.json
model_settings = %s

os.makedirs('models', exist_ok=True)
os.makedirs('data', exist_ok=True)
with open('model_settings.json', 'w') as f:
    json.dump(model_settings, f)

# Make sure you upload your formatted Kaggle CSV to `./data/Small_HI/formatted_transactions.csv`!
''' % get_file_contents('model_settings.json')

cells.append(nbf.v4.new_markdown_cell("## Main Script"))
cells.append(nbf.v4.new_code_cell(main_code))

cells.append(nbf.v4.new_code_cell('''
# Execute Training Pipeline
def set_seed(seed: int = 0) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    logging.info(f"Random seed set as {seed}")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)-5.5s] %(message)s", stream=sys.stdout)

set_seed(args.seed)

print("Retrieving Data...")
# This will fail if the dataset does not exist! Please upload it!
try:
    tr_data, val_data, te_data, tr_inds, val_inds, te_inds = get_data(args, data_config)
    print("Data retrieved! Starting Training...")
    train_gnn(tr_data, val_data, te_data, tr_inds, val_inds, te_inds, args, data_config)
except FileNotFoundError as e:
    print(f"Error: {e}")
    print("Please make sure you have the 'formatted_transactions.csv' dataset located at ./data/Small_HI/formatted_transactions.csv in the Colab instance.")
'''))

nb.cells = cells
with open('Standalone_Multi_GNN_Colab.ipynb', 'w') as f:
    nbf.write(nb, f)
print("Notebook Created!")
