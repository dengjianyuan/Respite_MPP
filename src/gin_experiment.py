import os

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim

# for graph representation
from torch_geometric.data import DataLoader

from dataset import GraphDataset
from utils import load_data, get_data_splits, metric_calc
from utils import NoamLR, build_optimizer, build_lr_scheduler
from models import GIN

from argparse import ArgumentParser

import wandb

### run experiments
def run():
    # parse args
    args = parser.parse_args()

    # get all_symbols and n_symbols for the specific mol_prop
    all_symbols, n_symbols = load_data(args.mol_prop)

    # get SMILES list and label list for training and test sets
    train_smiles, train_label, valid_smiles, valid_label, test_smiles, test_label, task, ckpts_path = get_data_splits(args.mol_prop, args.split_type, args.seed, args.task_type) 

    # modify ckpts_path by model name
    ckpts_path = ckpts_path + 'GIN/'

    # specify batch_size in dataloaders: 32 by default 
    batch_size = args.batch_size 

    # set up dataloaders for training set
    data_train = GraphDataset(train_smiles, train_label)
    dataloader_train = DataLoader(
        data_train, batch_size=batch_size, shuffle=False, num_workers=args.num_workers, drop_last=False)
    # set up dataloaders for valid set 
    data_valid = GraphDataset(valid_smiles, valid_label) 
    dataloader_valid = DataLoader(
        data_valid, batch_size=batch_size, shuffle=False, num_workers=args.num_workers, drop_last=False)
    # set up dataloaders for test set 
    data_test = GraphDataset(test_smiles, test_label) 
    dataloader_test = DataLoader(
        data_test, batch_size=batch_size, shuffle=False, num_workers=args.num_workers, drop_last=False)

    # select which gpu to use
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device('cuda:0')

    # specify loss: CE loss for classification tasks & MSE loss for regression tasks
    if args.mol_prop in ['BACE', 'BBBP', 'HIV'] or task == 'cutoff6': # currently only 3 moleculenet datasets are CLS task
        task_setting = 'CLS'
        loss_fn = nn.BCEWithLogitsLoss() 
    else:
        task_setting = 'REG' 
        loss_fn = nn.MSELoss()

    # initialize the neural network model
    model = GIN(task_setting, num_layer=args.num_layers, emb_dim=args.emb_dim, feat_dim=args.feat_dim, drop_ratio=args.drop_ratio, pool=args.pooling).to(device)

    # initialize molecular property prediction experiment in wandb
    wandb.init(project="aidd_MPP", name='GIN_{}_{}_{}'.format(args.mol_prop, args.split_type, args.seed), entity="dengjianyuan", save_code=True)
    # add all of the arguments as config variables
    wandb.config.update(args) 

    # specify model training epochs: default 100
    EPOCH = args.epochs 

    # initialize optimizer
    optimizer = build_optimizer(model, args)

    # when is set True, enable lr scheduler 
    if args.alter_lr:
        # add args.train_data_size 
        args.train_data_size = len(train_smiles)
        # add args.num_lrs
        args.num_lrs = 1
        # instantiate scheduler
        scheduler = build_lr_scheduler(optimizer, args)

    # create model checkpoints path
    os.makedirs(ckpts_path, exist_ok=True)

    # create a sigmoid function for converting logits into probs
    logits2probs = nn.Sigmoid()

    # start training 
    global_step = 0
    for epoch in range(EPOCH):
        print("Epoch", epoch+1)
        # initialize training_loss
        training_loss = 0.0
        for i, batch_data in enumerate(dataloader_train):
            # reset parameter gradients
            optimizer.zero_grad()
            # put batch_data and labels to device
            batch_data, labels = batch_data.to(device), batch_data.y.to(device)
            # get raw predictions in forward pass 
            preds = model(batch_data)
            # calculate loss
            loss = loss_fn(preds.squeeze(), labels.float().squeeze())
            # back propogation & update model parameters
            loss.backward()
            # if args.grad_clip:
            #     nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            if args.alter_lr:
                scheduler.step()
            # update training loss
            training_loss += loss.item() 

        # log train_loss after each training epoch 
        wandb.log({'training loss': training_loss, 'epoch': epoch})

        # evaluate on the valid set after each training epoch: valid_labels, valid_preds
        valid_labels, valid_preds = [], []
        # initialize valid loss and a best_valid_loss (reference: molclr)
        valid_loss, best_valid_loss = 0.0, np.inf

        with torch.no_grad():
            for j, batch_data in enumerate(dataloader_valid):
                # put batch_data and labels to device
                batch_data, labels = batch_data.to(device), batch_data.y.to(device)
                # get raw predictions in forward pass 
                batch_preds = model(batch_data)
                # extend valid_preds and valid_labels
                valid_preds.extend(batch_preds)
                valid_labels.extend(labels)
                # calculate validation loss
                loss = loss_fn(batch_preds.squeeze(), labels.float().squeeze())
                # update validation loss
                valid_loss += loss.item() 
    
        # log train_loss after each training epoch 
        wandb.log({'valid loss': valid_loss, 'epoch': epoch})

        # convert raw (logits) preds into probs in case of CLS task_setting
        if task_setting == 'CLS':
            valid_preds = [logits2probs(x) for x in valid_preds]

        # put the list of tensors to cpu
        valid_labels, valid_preds = [x.cpu().item() for x in valid_labels], [x.cpu().item() for x in valid_preds]

        # calculate performance metrics
        if task_setting == 'CLS':
            AUROC, AUPRC = metric_calc(valid_labels, valid_preds, 'AUROC'), metric_calc(valid_labels, valid_preds, 'AUPRC')
            Precision_PPV, Precision_NPV = metric_calc(valid_labels, valid_preds, 'Precision_PPV'), metric_calc(valid_labels, valid_preds, 'Precision_NPV')
            wandb.log({'AUROC': AUROC, 'AUPRC': AUPRC, 'Precision_PPV': Precision_PPV, 'Precision_NPV': Precision_NPV, 'epoch': epoch})
        elif task_setting == 'REG':
            RMSE, MAE = metric_calc(valid_labels, valid_preds, 'RMSE'), metric_calc(valid_labels, valid_preds, 'MAE')
            R2, Pearson_R = metric_calc(valid_labels, valid_preds, 'R2'), metric_calc(valid_labels, valid_preds, 'Pearson_R')
            wandb.log({'RMSE': RMSE, 'MAE': MAE, 'R2': R2, 'Pearson_R': Pearson_R, 'epoch': epoch})

        # save best model by validation loss 
        if valid_loss < best_valid_loss:
            # update best_valid_loss
            best_valid_loss = valid_loss
            # save model weights
            torch.save(model.state_dict(), os.path.join(ckpts_path, 'model.pth'))

    # initialize a new empty model for testing predictions
    test_model = GIN(task_setting, num_layer=args.num_layers, emb_dim=args.emb_dim, feat_dim=args.feat_dim, drop_ratio=args.drop_ratio, pool=args.pooling).to(device)
    # load best model weights and apply predictions on the test set
    state_dict = torch.load(os.path.join(ckpts_path, 'model.pth'), map_location=device)
    test_model.load_state_dict(state_dict)
    print("Best trained model loaded successfully!")

    # make lists to store the molecules, test_labels, raw_preds 
    smiles, raw_labels, raw_preds = [], [], []

    with torch.no_grad():
        for k, batch_data in enumerate(dataloader_test):
            # put batch_data and labels to device
            batch_data, labels = batch_data.to(device), batch_data.y.to(device)
            # get raw predictions in forward pass 
            batch_preds = model(batch_data)
            # convert raw (logits) preds into probs in case of CLS task_setting
            if task_setting == 'CLS':
                batch_preds = logits2probs(batch_preds)
            # extend raw_preds and raw_labels
            raw_preds.extend(batch_preds)
            raw_labels.extend(labels)

    # put the list of tensors to cpu
    raw_labels, raw_preds = [x.cpu().item() for x in raw_labels], [x.cpu().item() for x in raw_preds]

    # record predictions at the end of 100 training epochs in a dataframe and save to .csv file
    test_result_df = pd.DataFrame()
    test_result_df['preds'], test_result_df['labels'], test_result_df['SMILES'] = raw_preds, raw_labels, test_smiles
    test_result_df['mol_prop'], test_result_df['model_name'], test_result_df['split_type'], test_result_df['fold'] = args.mol_prop, 'GIN', args.split_type, args.seed
    test_result_df['mol_rep'] = 'Graph'

    save_path = '../results/raw_predictions/GIN/{}/{}/{}'.format(task, args.mol_prop, args.split_type)
    os.makedirs(save_path, exist_ok=True)
    test_result_df.to_csv(save_path+'/test_result_fold{}.csv'.format(args.seed), index=False)

    # finish logging in wandb
    wandb.finish()

# add arguments    
parser = ArgumentParser()
parser.add_argument("--task_type", type=str, default="reg", help='Adjust task type for opioids-related datasets; options: reg, cutoff6')
parser.add_argument("--mol_prop", type=str, default="ESOL")
parser.add_argument("--split_type", type=str, default="scaffold", help='Dataset split types; options: random, scaffold')
parser.add_argument("--seed", type=str, default="0", help='Seeds for dataset splitting; options: 0-29')

parser.add_argument("--batch_size", type=int, default=32, help="Batch size in the dataloader")

parser.add_argument("--num_layers", type=int, default=5, help="Number of GNN layers")
parser.add_argument("--emb_dim", type=int, default=300, help="Dimension in latent embedding space")
parser.add_argument("--feat_dim", type=int, default=512, help="Dimension for final learned embedding")
parser.add_argument("--drop_ratio", type=float, default=0.3, help="Dropout ratio")
parser.add_argument("--pooling", type=str, default="mean", help="Pooling type; options: mean, add, max")
parser.add_argument("--pred_n_layer", type=int, default=2, help="Number of linear layers after graph readout")
parser.add_argument("--pred_act", type=str, default='softplus', help="Activation type; options: softplus, relu")

parser.add_argument("--gpu", type=str, default='2', help="Which GPU is idle to put a model") # 2, 3, 4
parser.add_argument("--num_workers", type=int, default=16, help="How many workers to use in dataloaders")

parser.add_argument("--epochs", type=int, default=100, help="How many epochs for generator training")

parser.add_argument("--lr", type=float, default=1.5e-4, help='Constant learning rate')
parser.add_argument("--weight_decay", type=float, default=1e-6, help='Initial learning rate') # grover 1e-7, molclr 1e-6
parser.add_argument("--grad_clip", action='store_true', help='Whether to clip gradients')

parser.add_argument('--alter_lr', action='store_true', help='Whether to alter learning rate during training')
parser.add_argument('--warmup_epochs', type=float, default=2.0, help='Number of epochs during which learning rate increases linearly from init_lr to max_lr. Afterwards, learning rate decreases exponentially from max_lr to final_lr.')
parser.add_argument('--init_lr', type=float, default=1.5e-4, help='Initial learning rate') #chemprop 1e-4, grpver 1.5e-4
parser.add_argument('--max_lr', type=float, default=1e-3, help='Maximum learning rate')
parser.add_argument('--final_lr', type=float, default=1e-4, help='Final learning rate')

if __name__ == "__main__":
    run()   