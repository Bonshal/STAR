#!/bin/bash
set -e

echo "Installing python3-venv..."
sudo apt-get update && sudo apt-get install -y python3.12-venv

echo "Setting up Virtual Environment..."
python3 -m venv ~/.venv
source ~/.venv/bin/activate

echo "Setting up Kaggle credentials..."
mkdir -p ~/.kaggle
echo '{"username":"bonshallangthasa","key":"2ad5574a4312e6aa43400378f16af386"}' > ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json

echo "Installing requirements..."
pip install kaggle datatable scikit-learn pandas wandb
pip install torch torchvision torchaudio
pip install torch_geometric

echo "Downloading HI-Small dataset..."
mkdir -p AML_dataset
cd AML_dataset
kaggle datasets download -d ealtman2019/ibm-transactions-for-anti-money-laundering-aml -f HI-Small_Trans.csv
unzip -o HI-Small_Trans.csv.zip
rm HI-Small_Trans.csv.zip
cd ..

echo "Formatting CSV..."
python format_kaggle_files.py AML_dataset/HI-Small_Trans.csv

echo "Moving formatted dataset to PyG directory..."
mkdir -p data/Small_HI
mv AML_dataset/formatted_transactions.csv data/Small_HI/

echo "Starting TGNN training on HI-Small..."
# We enable both temporal dynamics (--tds) and structural features (--ports, --ego, --emlps)
# to run the fully optimized Spatial-Temporal Hybrid GNN for the production AML pipeline.
python main.py --data Small_HI --model tgnn --tds --ports --ego --emlps --testing --batch_size 131072 --save_model --unique_name small_hi_tgnn
