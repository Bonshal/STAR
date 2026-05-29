import torch
import pandas as pd
import numpy as np
from sklearn.metrics import precision_recall_curve, f1_score, fbeta_score

from train_util import AddEgoIds, extract_param, add_arange_ids, get_loaders, evaluate_homo
from training import get_model
from data_loading import get_data
from util import create_parser, set_seed
import json

def find_best_threshold():
    print("=> Loading data and configurations...")
    parser = create_parser()
    # Mocking arguments for local testing
    args = parser.parse_args(['--data', 'Small_LI', '--model', 'gin', '--testing', '--batch_size', '8192'])
    
    with open('data_config.json', 'r') as config_file:
        data_config = json.load(config_file)
        
    set_seed(1)
    
    tr_data, val_data, te_data, tr_inds, val_inds, te_inds = get_data(args, data_config)
    add_arange_ids([tr_data, val_data, te_data])
    
    tr_loader, val_loader, te_loader = get_loaders(tr_data, val_data, te_data, tr_inds, val_inds, te_inds, None, args)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    print("=> Loading optimized H100 weights...")
    sample_batch = next(iter(tr_loader))
    
    # Create a dummy config to match wandb
    class DummyConfig:
        n_gnn_layers = 2
        n_hidden = 66
        dropout = 0.01
        final_dropout = 0.10
        w_ce1 = 1.0
        w_ce2 = 6.27
    
    model = get_model(sample_batch, DummyConfig(), args)
    
    # Load the tar file downloaded from H100
    checkpoint = torch.load('models/checkpoint_True.tar', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print("=> Running inference on Test set to extract probabilities (this may take a minute)...")
    
    pred_probs = []
    ground_truths = []
    
    # Manually run the test loader to extract probabilities
    for batch in te_loader:
        inds = te_inds.detach().cpu()
        batch_edge_inds = inds[batch.input_id.detach().cpu()]
        batch_edge_ids = te_loader.data.edge_attr.detach().cpu()[batch_edge_inds, 0]
        mask = torch.isin(batch.edge_attr[:, 0].detach().cpu(), batch_edge_ids)
        batch.edge_attr = batch.edge_attr[:, 1:]
        
        with torch.no_grad():
            batch.to(device)
            out = model(batch.x, batch.edge_index, batch.edge_attr)
            out = out[mask]
            probs = torch.nn.functional.softmax(out, dim=-1)[:, 1]
            pred_probs.append(probs)
            ground_truths.append(batch.y[mask])
            
    pred_probs = torch.cat(pred_probs, dim=0).cpu().numpy()
    ground_truths = torch.cat(ground_truths, dim=0).cpu().numpy()
    
    print("=> Calculating optimal threshold via Precision-Recall Curve...")
    precisions, recalls, thresholds = precision_recall_curve(ground_truths, pred_probs)
    
    # Find threshold for 80% Recall
    target_recall_80 = 0.80
    valid_indices_80 = np.where(recalls >= target_recall_80)[0]
    best_idx_80 = valid_indices_80[np.argmax(precisions[valid_indices_80])] if len(valid_indices_80) > 0 else 0
    
    # Find threshold for 95% Recall
    target_recall_95 = 0.95
    valid_indices_95 = np.where(recalls >= target_recall_95)[0]
    best_idx_95 = valid_indices_95[np.argmax(precisions[valid_indices_95])] if len(valid_indices_95) > 0 else 0
    
    print(f"\n--- BUSINESS TARGET: 80% RECALL ---")
    print(f"REQUIRED THRESHOLD: {thresholds[best_idx_80]:.5f}")
    print(f"Resulting Precision: {precisions[best_idx_80]:.4f}")
    print(f"Actual Recall Achieved: {recalls[best_idx_80]:.4f}")
    
    print(f"\n--- BUSINESS TARGET: 95% RECALL ---")
    print(f"REQUIRED THRESHOLD: {thresholds[best_idx_95]:.5f}")
    print(f"Resulting Precision: {precisions[best_idx_95]:.4f}")
    print(f"Actual Recall Achieved: {recalls[best_idx_95]:.4f}")
    
if __name__ == "__main__":
    find_best_threshold()
