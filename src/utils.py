import numpy as np
import pandas as pd

from sklearn import metrics
import scipy

from argparse import Namespace
from typing import List, Union

import torch.nn as nn
from torch.optim import Adam, Optimizer
from torch.optim.lr_scheduler import _LRScheduler

from rdkit.ML.Scoring.Scoring import CalcBEDROC, CalcEnrichment, CalcRIE

### load original datasets to get all symbols
def load_data(mol_prop: str):
    """
    Load original datasets to get all symbols
    Inputs: mol_prop, options include benchmark (moleculenet), opioids, bender, desc datasets
    Return: a list of all symbols, number of all symbols
    """
    # if mol_prop in the following category, then specify the directories accordingly
    if mol_prop in ['BACE', 'BBBP', 'HIV', 'ESOL', 'FreeSolv', 'Lipop']:
        filename = "../data/benchmark/{}_benchmark.csv".format(mol_prop)
    elif mol_prop in ['MDR1', 'CYP2D6', 'CYP3A4', 'MOR', 'DOR', 'KOR']:
        filename =  "../data/opioids/{}_reg.csv".format(mol_prop)
    elif mol_prop in ['A2a', 'ABL1', 'Acetylcholinesterase', 'Aurora-A', 'B-raf', 'COX-1', 'COX-2', \
             'Cannabinoid', 'Carbonic', 'Caspase', 'Coagulation', 'Dihydrofolate', 'Dopamine', \
             'Ephrin', 'Estrogen', 'Glucocorticoid', 'Glycogen', 'HERG', 'JAK2', 'LCK', \
             'Monoamine', 'Vanilloid', 'erbB1', 'opioid']:
        filename =  "../data/bender/{}.csv".format(mol_prop)
    else: # add the case for desc datasets
        filename =  "../data/desc/{}.csv".format(mol_prop)

    # read data and only keep the SMILES string
    data = pd.read_csv(filename)
    data = data[['SMILES']]

    # convert all SMILES strings to an numpy array
    all_SMILES = data.SMILES.to_numpy()

    # get all symbols
    all_symbols = []
    [all_symbols.extend(list(smiles)) for smiles in all_SMILES]
    all_symbols = np.unique(all_symbols)
    all_symbols = ''.join(list(all_symbols))

    # get the number of all symbols from all_SMILES
    n_symbols = len(all_symbols)  

    print('****************Loading data*********************')
    print('Number of data points:{}'.format(len(all_SMILES)))

    return all_symbols, n_symbols

### get data splits: either load previously saved data splits 
def get_data_splits(mol_prop: str, split_type: str, seed: str, task_type: str, use_saved=True):
    """
    Get data splits from previously saved data splits 
    Inputs:
       mol_prop: name for the mol_prop (options include benchmark (moleculenet), opioids, bender)
       split_type: 'scaffold' or 'random'
       seed: '0' to '29'
       task_type: 'reg' or 'cutoff6'
       use_saved: True by default 
    Return:
        train/valid/test SMILES and labels (as list), task (indicator for datasets group), ckpts_path (for saving models)
    """
    # if mol_prop in the following category, then specify the directories accordingly
    if mol_prop in ['BACE', 'BBBP', 'HIV', 'ESOL', 'FreeSolv', 'Lipop']:
        task = 'benchmark'
        filedir = "../data/benchmark/{}_split/{}/".format(split_type, task)
        ckpts_path = "../checkpoints/benchmark/{}_split/{}/fold{}/".format(split_type, task, seed)
    elif mol_prop in ['MDR1', 'CYP2D6', 'CYP3A4', 'MOR', 'DOR', 'KOR']:
        task = task_type
        filedir =  "../data/opioids/{}_split/{}/".format(split_type, task)
        ckpts_path =  "../checkpoints/opioids/{}_split/{}/fold{}/".format(split_type, task, seed)
    elif mol_prop in ['A2a', 'ABL1', 'Acetylcholinesterase', 'Aurora-A', 'B-raf', 'COX-1', 'COX-2', \
             'Cannabinoid', 'Carbonic', 'Caspase', 'Coagulation', 'Dihydrofolate', 'Dopamine', \
             'Ephrin', 'Estrogen', 'Glucocorticoid', 'Glycogen', 'HERG', 'JAK2', 'LCK', \
             'Monoamine', 'Vanilloid', 'erbB1', 'opioid']:
        task = 'bender'
        filedir =  "../data/bender/{}_split/{}/".format(split_type, task) 
        ckpts_path =  "../checkpoints/bender/{}_split/{}/fold{}/".format(split_type, task, seed) 
    else:
        task = 'desc'
        filedir =  "../data/desc/{}_split/{}/".format(split_type, task) 
        ckpts_path =  "../checkpoints/desc/{}_split/{}/fold{}/".format(split_type, task, seed) 

    train_data = pd.read_csv(filedir+"{}_{}_train_v{}.csv".format(mol_prop, split_type, seed))
    valid_data = pd.read_csv(filedir+"{}_{}_valid_v{}.csv".format(mol_prop, split_type, seed)) 
    test_data = pd.read_csv(filedir+"{}_{}_test_v{}.csv".format(mol_prop, split_type, seed))  


    return list(train_data['SMILES']), list(train_data['label']), list(valid_data['SMILES']), list(valid_data['label']), list(test_data['SMILES']), list(test_data['label']), task, ckpts_path

### alter learning rate for the optimizer: reference chemprop
class NoamLR(_LRScheduler):
    """
    Noam learning rate scheduler with piecewise linear increase and exponential decay.
    The learning rate increases linearly from init_lr to max_lr over the course of
    the first warmup_steps (where :code:`warmup_steps = warmup_epochs * steps_per_epoch`).
    Then the learning rate decreases exponentially from :code:`max_lr` to :code:`final_lr` over the
    course of the remaining :code:`total_steps - warmup_steps` (where :code:`total_steps =
    total_epochs * steps_per_epoch`). This is roughly based on the learning rate
    schedule from `Attention is All You Need <https://arxiv.org/abs/1706.03762>`_, section 5.3.
    """
    def __init__(self,
                 optimizer: Optimizer,
                 warmup_epochs: List[Union[float, int]],
                 total_epochs: List[int],
                 steps_per_epoch: int,
                 init_lr: List[float],
                 max_lr: List[float],
                 final_lr: List[float]):
        """
        :param optimizer: A PyTorch optimizer.
        :param warmup_epochs: The number of epochs during which to linearly increase the learning rate.
        :param total_epochs: The total number of epochs.
        :param steps_per_epoch: The number of steps (batches) per epoch.
        :param init_lr: The initial learning rate.
        :param max_lr: The maximum learning rate (achieved after :code:`warmup_epochs`).
        :param final_lr: The final learning rate (achieved after :code:`total_epochs`).
        """
        if not (
            len(optimizer.param_groups) == len(warmup_epochs) == len(total_epochs)
            == len(init_lr) == len(max_lr) == len(final_lr)
        ):
            raise ValueError(
                "Number of param groups must match the number of epochs and learning rates! "
                f"got: len(optimizer.param_groups)= {len(optimizer.param_groups)}, "
                f"len(warmup_epochs)= {len(warmup_epochs)}, "
                f"len(total_epochs)= {len(total_epochs)}, "
                f"len(init_lr)= {len(init_lr)}, "
                f"len(max_lr)= {len(max_lr)}, "
                f"len(final_lr)= {len(final_lr)}"
            )

        self.num_lrs = len(optimizer.param_groups)

        self.optimizer = optimizer
        self.warmup_epochs = np.array(warmup_epochs)
        self.total_epochs = np.array(total_epochs)
        self.steps_per_epoch = steps_per_epoch
        self.init_lr = np.array(init_lr)
        self.max_lr = np.array(max_lr)
        self.final_lr = np.array(final_lr)

        self.current_step = 0
        self.lr = init_lr
        self.warmup_steps = (self.warmup_epochs * self.steps_per_epoch).astype(int)
        self.total_steps = self.total_epochs * self.steps_per_epoch
        self.linear_increment = (self.max_lr - self.init_lr) / self.warmup_steps

        self.exponential_gamma = (self.final_lr / self.max_lr) ** (1 / (self.total_steps - self.warmup_steps))

        super(NoamLR, self).__init__(optimizer)

    def get_lr(self) -> List[float]:
        """
        Gets a list of the current learning rates.
        :return: A list of the current learning rates.
        """
        return list(self.lr)

    def step(self, current_step: int = None):
        """
        Updates the learning rate by taking a step.
        :param current_step: Optionally specify what step to set the learning rate to.
                             If None, :code:`current_step = self.current_step + 1`.
        """
        if current_step is not None:
            self.current_step = current_step
        else:
            self.current_step += 1

        for i in range(self.num_lrs):
            if self.current_step <= self.warmup_steps[i]:
                self.lr[i] = self.init_lr[i] + self.current_step * self.linear_increment[i]
            elif self.current_step <= self.total_steps[i]:
                self.lr[i] = self.max_lr[i] * (self.exponential_gamma[i] ** (self.current_step - self.warmup_steps[i]))
            else:  # theoretically this case should never be reached since training should stop at total_steps
                self.lr[i] = self.final_lr[i]

            self.optimizer.param_groups[i]['lr'] = self.lr[i]


def build_optimizer(model: nn.Module, args: Namespace) -> Optimizer:
    """
    Builds a PyTorch Optimizer.
    :param model: The model to optimize.
    :param args: A :class:`~chemprop.args.TrainArgs` object containing optimizer arguments.
    :return: An initialized Optimizer.
    """
    params = [{"params": model.parameters(), "lr": args.init_lr, "weight_decay": 0}]

    return Adam(params)

def build_lr_scheduler(
    optimizer: Optimizer, args: Namespace, total_epochs: List[int] = None
) -> _LRScheduler:
    """
    Builds a PyTorch learning rate scheduler.
    :param optimizer: The Optimizer whose learning rate will be scheduled.
    :param args: A :class:`~chemprop.args.TrainArgs` object containing learning rate arguments.
    :param total_epochs: The total number of epochs for which the model will be run.
    :return: An initialized learning rate scheduler.
    """
    # Learning rate scheduler
    return NoamLR(
        optimizer=optimizer,
        warmup_epochs=[args.warmup_epochs],
        total_epochs=total_epochs or [args.epochs] * args.num_lrs,
        steps_per_epoch=args.train_data_size // args.batch_size,
        init_lr=[args.init_lr],
        max_lr=[args.max_lr],
        final_lr=[args.final_lr],
    )

### calculate metrics 
def metric_calc(labels, preds, metric='RMSE', proba_threshold='optimal'):
    """A function for metrics calculation"""
    # convert the labels and preds to list
    labels, preds = list(labels), list(preds)

    # get proba_cutoff for classification metrics calculation
    if metric not in ['RMSE', 'MAE', 'R2', 'Pearson_R', 'MAPE']:
        if proba_threshold == 'default':
            proba_cutoff = 0.5
        elif proba_threshold == 'optimal':
            # get the roc curve points
            false_pos_rate, true_pos_rate, proba = metrics.roc_curve(labels, preds)
            # calculate the optimal probability cutoff using Youden's J statistic with equal weight to FP and FN
            proba_cutoff = sorted(list(zip(np.abs(true_pos_rate - false_pos_rate), proba)), key=lambda i: i[0], reverse=True)[0][1]
        # get hard_preds
            hard_preds = [1 if p > proba_cutoff else 0 for p in preds]

    ### for REG tasks
    if metric == 'RMSE':
        score = metrics.mean_squared_error(labels, preds, squared=False)
        # an alternative way
#         score = np.sqrt(np.nanmean((np.array(labels)-np.array(preds))**2))
    elif metric == 'MAE': 
        score = metrics.mean_absolute_error(labels, preds) 
    elif metric == 'R2':
        score = metrics.r2_score(labels, preds)
    elif metric == 'Pearson_R':
        score = scipy.stats.pearsonr(labels, preds)[0] 
        # an alternative way
#         score = np.corrcoef(labels, preds)[0,1]**2
    elif metric == 'MAPE':
        score = metrics.mean_absolute_error(labels, preds)
    ### for CLS tasks
    elif metric == 'AUROC':
        try:
            score = metrics.roc_auc_score(labels, preds) 
        except ValueError:
            score = -1
    elif metric == 'AUPRC':
        # auc based on precision_recall curve
        precision, recall, _ = metrics.precision_recall_curve(labels, preds)
        score = metrics.auc(recall, precision)
        # an alternative way
#         score = metrics.average_precision_score(labels, preds)
    elif metric == 'Precision_PPV':
        # calculate precision based on hard_preds 
        score = metrics.precision_score(labels, hard_preds, pos_label=1)
    elif metric == 'Precision_NPV':
        # calculate precision based on hard_preds 
        score = metrics.precision_score(labels, hard_preds, pos_label=0)
    elif metric == 'MCC':
        # calculate precision based on hard_preds 
        score = metrics.matthews_corrcoef(labels, hard_preds)
        # print('MCC', score)
    elif metric == 'Cohen_Kappa':
        # calculate precision based on hard_preds 
        score = metrics.cohen_kappa_score(labels, hard_preds)
        # print('Cohen_Kappa', score)
    elif metric == 'BEDROC':
        # reference: deepchem - https://github.com/deepchem/deepchem/blob/master/deepchem/metrics/score_function.py#L132-L183
        scores = list(zip(labels, hard_preds))
        scores = sorted(scores, key=lambda pair: pair[1], reverse=True)
        # calculate BEDROC
        score = CalcBEDROC(scores, 0, alpha=20.0) # alpha: 0-20
        # print('BEDROC', score)
    elif metric == 'RIE': 
         # reference: deepchem - https://github.com/deepchem/deepchem/blob/master/deepchem/metrics/score_function.py#L132-L183
        scores = list(zip(labels, hard_preds))
        scores = sorted(scores, key=lambda pair: pair[1], reverse=True)
        # calculate RIE
        score = CalcRIE(scores, 0, alpha=20.0) # alpha: 0-20
        # print('RIE', score)
    elif metric == 'EF': 
        # reference: deepchem - https://github.com/deepchem/deepchem/blob/master/deepchem/metrics/score_function.py#L132-L183
        scores = list(zip(labels, hard_preds))
        scores = sorted(scores, key=lambda pair: pair[1], reverse=True)
        # calculate enrichment factor
        score = CalcEnrichment(scores, 0, fractions=[0.1])[0]
        # print('EF', score)
    return score 
