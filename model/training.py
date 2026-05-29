import torch
import tqdm
from sklearn.metrics import f1_score, precision_score, recall_score
from train_util import AddEgoIds, extract_param, add_arange_ids, get_loaders, evaluate_homo, evaluate_hetero, save_model, load_model
from models import GINe, PNA, GATe, RGCN, TGNN
from torch_geometric.data import Data, HeteroData
from torch_geometric.nn import to_hetero, summary
from torch_geometric.utils import degree
import wandb
import logging

class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, weight=None):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight

    def forward(self, inputs, targets):
        ce_loss = torch.nn.functional.cross_entropy(inputs, targets, weight=self.weight, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt)**self.gamma * ce_loss
        return focal_loss.mean()

def train_homo(tr_loader, val_loader, te_loader, tr_inds, val_inds, te_inds, model, optimizer, loss_fn, args, config, device, val_data, te_data, data_config):
    #training
    best_val_roc = 0
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == 'cuda'))
    for epoch in range(config.epochs):
        total_loss = total_examples = 0
        preds = []
        ground_truths = []
        for batch in tqdm.tqdm(tr_loader, disable=not args.tqdm):
            optimizer.zero_grad(set_to_none=True)
            #select the seed edges from which the batch was created (done on CPU to avoid GPU syncs)
            batch_edge_inds = tr_inds[batch.input_id]
            batch_edge_ids = tr_loader.data.edge_attr[batch_edge_inds, 0]
            mask = torch.isin(batch.edge_attr[:, 0], batch_edge_ids)

            #remove the unique edge id from the edge features, as it's no longer needed
            batch.edge_attr = batch.edge_attr[:, 1:]

            batch.to(device, non_blocking=True)
            
            with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
                out = model(batch.x, batch.edge_index, batch.edge_attr)
                pred = out[mask]
                ground_truth = batch.y[mask]
                loss = loss_fn(pred, ground_truth)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            preds.append(pred.argmax(dim=-1))
            ground_truths.append(ground_truth)
            total_loss += float(loss) * pred.numel()
            total_examples += pred.numel()

        pred = torch.cat(preds, dim=0).detach().cpu().numpy()
        ground_truth = torch.cat(ground_truths, dim=0).detach().cpu().numpy()
        f1 = f1_score(ground_truth, pred, zero_division=0)
        prec = precision_score(ground_truth, pred, zero_division=0)
        rec = recall_score(ground_truth, pred, zero_division=0)
        
        wandb.log({"f1/train": f1, "prec/train": prec, "rec/train": rec}, step=epoch)
        logging.info(f'Train F1: {f1:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f}')

        #evaluate
        val_f1, val_prec, val_rec, val_roc, val_pr = evaluate_homo(val_loader, val_inds, model, val_data, device, args)
        te_f1, te_prec, te_rec, te_roc, te_pr = evaluate_homo(te_loader, te_inds, model, te_data, device, args)

        wandb.log({"f1/validation": val_f1}, step=epoch)
        wandb.log({"f1/test": te_f1}, step=epoch)
        logging.info(f'Validation F1: {val_f1:.4f} | Prec: {val_prec:.4f} | Rec: {val_rec:.4f} | ROC-AUC: {val_roc:.4f} | PR-AUC: {val_pr:.4f}')
        logging.info(f'Test F1: {te_f1:.4f} | Prec: {te_prec:.4f} | Rec: {te_rec:.4f} | ROC-AUC: {te_roc:.4f} | PR-AUC: {te_pr:.4f}')

        if epoch == 0:
            wandb.log({"best_test_roc": te_roc}, step=epoch)
        elif val_roc > best_val_roc:
            best_val_roc = val_roc
            wandb.log({"best_test_roc": te_roc}, step=epoch)
            if args.save_model:
                save_model(model, optimizer, epoch, args, data_config)
    
    return model

def train_hetero(tr_loader, val_loader, te_loader, tr_inds, val_inds, te_inds, model, optimizer, loss_fn, args, config, device, val_data, te_data, data_config):
    #training
    best_val_f1 = 0
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == 'cuda'))
    for epoch in range(config.epochs):
        total_loss = total_examples = 0
        preds = []
        ground_truths = []
        for batch in tqdm.tqdm(tr_loader, disable=not args.tqdm):
            optimizer.zero_grad(set_to_none=True)
            #select the seed edges from which the batch was created (done on CPU to avoid GPU syncs)
            batch_edge_inds = tr_inds[batch['node', 'to', 'node'].input_id]
            batch_edge_ids = tr_loader.data['node', 'to', 'node'].edge_attr[batch_edge_inds, 0]
            mask = torch.isin(batch['node', 'to', 'node'].edge_attr[:, 0], batch_edge_ids)
            
            #remove the unique edge id from the edge features, as it's no longer needed
            batch['node', 'to', 'node'].edge_attr = batch['node', 'to', 'node'].edge_attr[:, 1:]
            batch['node', 'rev_to', 'node'].edge_attr = batch['node', 'rev_to', 'node'].edge_attr[:, 1:]

            batch.to(device, non_blocking=True)
            
            with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
                out = model(batch.x_dict, batch.edge_index_dict, batch.edge_attr_dict)
                out = out[('node', 'to', 'node')]
                pred = out[mask]
                ground_truth = batch['node', 'to', 'node'].y[mask]
                loss = loss_fn(pred, ground_truth)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            preds.append(pred.argmax(dim=-1))
            ground_truths.append(batch['node', 'to', 'node'].y[mask])
            total_loss += float(loss) * pred.numel()
            total_examples += pred.numel()
            
        pred = torch.cat(preds, dim=0).detach().cpu().numpy()
        ground_truth = torch.cat(ground_truths, dim=0).detach().cpu().numpy()
        f1 = f1_score(ground_truth, pred)
        wandb.log({"f1/train": f1}, step=epoch)
        logging.info(f'Train F1: {f1:.4f}')

        #evaluate
        val_f1 = evaluate_hetero(val_loader, val_inds, model, val_data, device, args)
        te_f1 = evaluate_hetero(te_loader, te_inds, model, te_data, device, args)

        wandb.log({"f1/validation": val_f1}, step=epoch)
        wandb.log({"f1/test": te_f1}, step=epoch)
        logging.info(f'Validation F1: {val_f1:.4f}')
        logging.info(f'Test F1: {te_f1:.4f}')

        if epoch == 0:
            wandb.log({"best_test_f1": te_f1}, step=epoch)
        elif val_f1 > best_val_f1:
            best_val_f1 = val_f1
            wandb.log({"best_test_f1": te_f1}, step=epoch)
            if args.save_model:
                save_model(model, optimizer, epoch, args, data_config)
        
    return model

def get_model(sample_batch, config, args):
    n_feats = sample_batch.x.shape[1] if not isinstance(sample_batch, HeteroData) else sample_batch['node'].x.shape[1]
    e_dim = (sample_batch.edge_attr.shape[1] - 1) if not isinstance(sample_batch, HeteroData) else (sample_batch['node', 'to', 'node'].edge_attr.shape[1] - 1)

    if args.model == "gin":
        model = GINe(
                num_features=n_feats, num_gnn_layers=config.n_gnn_layers, n_classes=2,
                n_hidden=round(config.n_hidden), residual=False, edge_updates=args.emlps, edge_dim=e_dim, 
                dropout=config.dropout, final_dropout=config.final_dropout
                )
    elif args.model == "tgnn":
        model = TGNN(
                num_features=n_feats, num_gnn_layers=config.n_gnn_layers, n_classes=2,
                n_hidden=round(config.n_hidden), edge_updates=args.emlps, edge_dim=e_dim,
                dropout=config.dropout, final_dropout=config.final_dropout
                )
    elif args.model == "gat":
        model = GATe(
                num_features=n_feats, num_gnn_layers=config.n_gnn_layers, n_classes=2,
                n_hidden=round(config.n_hidden), n_heads=round(config.n_heads), 
                edge_updates=args.emlps, edge_dim=e_dim,
                dropout=config.dropout, final_dropout=config.final_dropout
                )
    elif args.model == "pna":
        if not isinstance(sample_batch, HeteroData):
            d = degree(sample_batch.edge_index[1], dtype=torch.long)
        else:
            index = torch.cat((sample_batch['node', 'to', 'node'].edge_index[1], sample_batch['node', 'rev_to', 'node'].edge_index[1]), 0)
            d = degree(index, dtype=torch.long)
        deg = torch.bincount(d, minlength=1)
        model = PNA(
            num_features=n_feats, num_gnn_layers=config.n_gnn_layers, n_classes=2,
            n_hidden=round(config.n_hidden), edge_updates=args.emlps, edge_dim=e_dim,
            dropout=config.dropout, deg=deg, final_dropout=config.final_dropout
            )
    elif config.model == "rgcn":
        model = RGCN(
            num_features=n_feats, edge_dim=e_dim, num_relations=8, num_gnn_layers=round(config.n_gnn_layers),
            n_classes=2, n_hidden=round(config.n_hidden),
            edge_update=args.emlps, dropout=config.dropout, final_dropout=config.final_dropout, n_bases=None #(maybe)
        )
    
    return model

def train_gnn(tr_data, val_data, te_data, tr_inds, val_inds, te_inds, args, data_config):
    #set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    #define a model config dictionary and wandb logging at the same time
    wandb.init(
        mode="disabled" if args.testing else "online",
        project="your_proj_name", #replace this with your wandb project name if you want to use wandb logging

        config={
            "epochs": args.n_epochs,
            "batch_size": args.batch_size,
            "model": args.model,
            "data": args.data,
            "num_neighbors": args.num_neighs,
            "lr": extract_param("lr", args),
            "n_hidden": extract_param("n_hidden", args),
            "n_gnn_layers": extract_param("n_gnn_layers", args),
            "loss": "ce",
            "w_ce1": extract_param("w_ce1", args),
            "w_ce2": extract_param("w_ce2", args),
            "dropout": extract_param("dropout", args),
            "final_dropout": extract_param("final_dropout", args),
            "n_heads": extract_param("n_heads", args) if args.model == 'gat' else None
        }
    )

    config = wandb.config

    #set the transform if ego ids should be used
    if args.ego:
        transform = AddEgoIds()
    else:
        transform = None

    #add the unique ids to later find the seed edges
    add_arange_ids([tr_data, val_data, te_data])

    tr_loader, val_loader, te_loader = get_loaders(tr_data, val_data, te_data, tr_inds, val_inds, te_inds, transform, args)

    #get the model
    sample_batch = next(iter(tr_loader))
    model = get_model(sample_batch, config, args)

    if args.reverse_mp:
        model = to_hetero(model, te_data.metadata(), aggr='mean')
    
    if args.finetune:
        model, optimizer = load_model(model, device, args, config, data_config)
    else:
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    
    sample_batch.to(device)
    sample_x = sample_batch.x if not isinstance(sample_batch, HeteroData) else sample_batch.x_dict
    sample_edge_index = sample_batch.edge_index if not isinstance(sample_batch, HeteroData) else sample_batch.edge_index_dict
    if isinstance(sample_batch, HeteroData):
        sample_batch['node', 'to', 'node'].edge_attr = sample_batch['node', 'to', 'node'].edge_attr[:, 1:]
        sample_batch['node', 'rev_to', 'node'].edge_attr = sample_batch['node', 'rev_to', 'node'].edge_attr[:, 1:]
    else:
        sample_batch.edge_attr = sample_batch.edge_attr[:, 1:]
    sample_edge_attr = sample_batch.edge_attr if not isinstance(sample_batch, HeteroData) else sample_batch.edge_attr_dict
    logging.info(summary(model, sample_x, sample_edge_index, sample_edge_attr))
    
    if hasattr(torch, 'compile') and device.type == 'cuda':
        logging.info("⚡ Compiling model using torch.compile for A100 GPU speedup...")
        try:
            model = torch.compile(model)
        except Exception as e:
            logging.warning(f"Could not compile model: {e}")
            
    # Using FocalLoss as requested to combat massive class imbalance
    loss_fn = FocalLoss(alpha=0.25, gamma=2.0, weight=torch.FloatTensor([config.w_ce1, config.w_ce2]).to(device))

    if args.reverse_mp:
        model = train_hetero(tr_loader, val_loader, te_loader, tr_inds, val_inds, te_inds, model, optimizer, loss_fn, args, config, device, val_data, te_data, data_config)
    else:
        model = train_homo(tr_loader, val_loader, te_loader, tr_inds, val_inds, te_inds, model, optimizer, loss_fn, args, config, device, val_data, te_data, data_config)
    
    wandb.finish()