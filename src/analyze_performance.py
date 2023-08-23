import os

import pandas as pd
import numpy as np

from utils import metric_calc

from argparse import ArgumentParser

# run analysis
def run():
    # parse args
    args = parser.parse_args()

    # set up mol_props (split types, task_settings, task) based on args
    if args.folder == 'benchmark':
        mol_props = ['BACE', 'BBBP', 'HIV', 'ESOL', 'FreeSolv',  'Lipop'] 
        split_types, num_folds  = ['scaffold', 'random'], 30
        task_settings = ['CLS', 'CLS', 'CLS', 'REG', 'REG', 'REG']
        task = 'benchmark'
    elif args.folder == 'opioids':
        mol_props = ['MDR1', 'CYP2D6', 'CYP3A4', 'MOR', 'DOR', 'KOR'] 
        split_types, num_folds = ['scaffold', 'random'], 30
        if args.task_type == 'reg':
            task_settings = ['REG'] * len(mol_props)
            task = 'reg'
        elif args.task_type == 'cutoff6':
            task_settings = ['CLS'] * len(mol_props)
            task = 'cutoff6'
    elif args.folder == 'bender':
        mol_props = ['A2a', 'ABL1', 'Acetylcholinesterase', 'Aurora-A', 'B-raf', 'COX-1', 'COX-2', \
             'Cannabinoid', 'Carbonic', 'Caspase', 'Coagulation', 'Dihydrofolate', 'Dopamine', \
             'Ephrin', 'Estrogen', 'Glucocorticoid', 'Glycogen', 'HERG', 'JAK2', 'LCK', \
             'Monoamine', 'Vanilloid', 'erbB1', 'opioid', ] 
        split_types, num_folds = ['scaffold', 'random'], 30
        task_settings = ['REG'] * len(mol_props)
        task = 'bender'
    elif args.folder == 'moleculeace':
        # list all names for .csv files
        file_names =  os.listdir('../data/moleculeACE/')
        mol_props = [item.split('.')[0] for item in file_names if '.csv' in item]
        split_types, num_folds = ['fixed'], 1
        task_settings = ['REG'] * len(mol_props)
        task = 'moleculeace'
    elif args.folder == 'desc':
        # only evaluated mw and natoms
        desc_names = [ 'mw', 'natoms' ]#'mw', 'logp', 'nhbd', 'natoms'
        dataset_sizes = [100, 200, 400, 600, 800, 1000, 2000, 4000, 6000, 8000, 10000, 20000, 40000, 60000, 80000, 100000]
        # generate mol_props
        mol_props = []
        for desc_name in desc_names:
            for dataset_size in dataset_sizes:
                mol_props.append("{}_size{}_desc".format(desc_name, dataset_size))
        split_types, num_folds = ['scaffold'], 30
        task_settings = ['REG'] * len(mol_props)
        task = 'desc'

    # specify model_names
    if args.folder == 'benchmark':
        model_names = ['MolBERT', 'GROVER', 'GROVER_RDKit', 'RF', 'SVM', 'XGBoost', 'RNN', 'GCN', 'GIN', ]
        mol_reps = ['morganBits', 'morganCounts', 'maccs', 'rdkit2d', 'physchem', 'atomPairs']
    elif args.folder == 'opioids':
        if args.task_type == 'reg':
            model_names = ['MolBERT', 'GROVER', 'GROVER_RDKit', 'RF', 'SVM', 'XGBoost', 'RNN', 'GCN', 'GIN', ]
        elif args.task_type == 'cutoff6':
            model_names = ['MolBERT', 'GROVER', 'GROVER_RDKit', 'RF', 'SVM', 'XGBoost', ]
        mol_reps = ['morganBits', 'morganCounts', 'maccs', 'rdkit2d', 'physchem', 'atomPairs']
    elif args.folder == 'bender':
        model_names = ['MolBERT', 'GROVER', 'GROVER_RDKit', 'RF', 'SVM', 'XGBoost', ]
        mol_reps = ['morganBits', 'morganCounts', 'maccs', 'rdkit2d', 'physchem', 'atomPairs']
    elif args.folder == 'moleculeace':
        model_names = ['RF', 'SVM', 'XGBoost', ]
        mol_reps = ['morganBits', 'morganCounts', 'maccs', 'rdkit2d', 'physchem', 'atomPairs']
    elif args.folder == 'desc':
        model_names = ['RF', 'SVM', 'XGBoost', 'RNN', 'GCN', 'GIN', 'GROVER', 'MolBERT']
        mol_reps = ['morganBits', 'maccs', 'atomPairs']

    if args.action == 'calc':
        # make an empty dataframe to attach perf_df
        perf_df = pd.DataFrame(columns=['metric_score', 'metric_name', 'mol_rep', 'mol_prop', 'model_name','fold', 'split_type', 'task'])
        # make two other perf_df to calculate performance for AC and non-AC molecules for opioids datasets: reg/cutoff6
        AC_perf_df, nonAC_perf_df = None, None
        if task in ['reg', 'cutoff6']:
            AC_perf_df = pd.DataFrame(columns=['metric_score', 'metric_name', 'mol_rep', 'mol_prop', 'model_name','fold', 'split_type', 'task'])
            nonAC_perf_df = pd.DataFrame(columns=['metric_score', 'metric_name', 'mol_rep', 'mol_prop', 'model_name','fold', 'split_type', 'task'])
        # loop through all settings
        for fold in range(num_folds):
            for split_type in split_types:
                for mol_prop in mol_props:
                    # get task_setting based on mol_prop
                    task_setting = task_settings[mol_props.index(mol_prop)]
                    if task_setting == 'CLS':
                        metric_names = ['AUROC', 'AUPRC', 'Precision_PPV', 'Precision_NPV', 'MCC', 'Cohen_Kappa', 'BEDROC', 'EF']
                    elif task_setting == 'REG':
                        metric_names = ['RMSE', 'MAE', 'R2', 'Pearson_R', 'MAPE']
                    for model_name in model_names:
                        for metric_name in metric_names:
                            # in case of RF, SVM, XGBoost, further loop through the mol_reps
                            if model_name in ['RF', 'SVM', 'XGBoost']: 
                                for mol_rep in mol_reps:
                                    try: 
                                        row_to_add = generate_row2add(model_name, task, mol_prop, split_type, fold, metric_name, mol_rep)
                                        perf_df = perf_df.append(row_to_add, ignore_index=True)
                                        if task in ['reg', 'cutoff6']:
                                            AC_row_to_add = generate_row2add(model_name, task, mol_prop, split_type, fold, metric_name, mol_rep, True)
                                            AC_perf_df = AC_perf_df.append(AC_row_to_add, ignore_index=True)
                                            nonAC_row_to_add = generate_row2add(model_name, task, mol_prop, split_type, fold, metric_name, mol_rep, False)
                                            nonAC_perf_df = nonAC_perf_df.append(nonAC_row_to_add, ignore_index=True)
                                    except FileNotFoundError:
                                        print("No test result yet:", model_name, '-', task, '-', mol_prop, '-', split_type, '-', fold, '-', mol_rep)
                                    
                            else:
                                try:
                                    row_to_add = generate_row2add(model_name, task, mol_prop, split_type, fold, metric_name, mol_rep=None)
                                    perf_df = perf_df.append(row_to_add, ignore_index=True)
                                    if task in ['reg', 'cutoff6']:
                                        AC_row_to_add = generate_row2add(model_name, task, mol_prop, split_type, fold, metric_name, mol_rep=None, AC_status=True)
                                        AC_perf_df = AC_perf_df.append(AC_row_to_add, ignore_index=True)
                                        nonAC_row_to_add = generate_row2add(model_name, task, mol_prop, split_type, fold, metric_name, mol_rep=None, AC_status=False)
                                        nonAC_perf_df = nonAC_perf_df.append(nonAC_row_to_add, ignore_index=True)
                                except FileNotFoundError:
                                        print("No test result yet:", model_name, '-', task, '-', mol_prop, '-', split_type, '-', fold)

        # save perf_df to .csv file
        perf_df.to_csv('../results/{}_grand_perf_df_{}.csv'.format(args.folder, task), index=False)
        # save AC and nonAC perf_df if they are not none
        if AC_perf_df is not None:
            AC_perf_df.to_csv('../results/{}_grand_AC_perf_df_{}.csv'.format(args.folder, task), index=False)
        if nonAC_perf_df is not None:
            nonAC_perf_df.to_csv('../results/{}_grand_nonAC_perf_df_{}.csv'.format(args.folder, task), index=False)
    
    if args.action == 'agg':

        #read the grand perf df
        grand_perf_df = pd.read_csv('../results/{}_grand_perf_df_{}.csv'.format(args.folder, task))
        # generate agg_perf_df
        agg_perf_df = aggregate_perf(args.folder, task, grand_perf_df, split_types, mol_props, task_settings, model_names, mol_reps)
        # save agg_perf_df to .csv file
        agg_perf_df.to_csv('../results/{}_agg_perf_df_{}.csv'.format(args.folder, task), index=False)

        # enable agg for grand_AC_perf_df and grand_nonAC_perf_df 
        if task in ['reg', 'cutoff6']:
            #read the grand perf df for AC
            grand_perf_df = pd.read_csv('../results/{}_grand_AC_perf_df_{}.csv'.format(args.folder, task))
            # generate agg_perf_df for AC
            agg_perf_df = aggregate_perf(args.folder, task, grand_perf_df, split_types, mol_props, task_settings, model_names, mol_reps)
            # save .csv file
            agg_perf_df.to_csv('../results/{}_agg_AC_perf_df_{}.csv'.format(args.folder, task), index=False)

            #read the grand perf df for nonAC
            grand_perf_df = pd.read_csv('../results/{}_grand_nonAC_perf_df_{}.csv'.format(args.folder, task))
            # generate agg_perf_df for nonAC
            agg_perf_df = aggregate_perf(args.folder, task, grand_perf_df, split_types, mol_props, task_settings, model_names, mol_reps)
            # save .csv file
            agg_perf_df.to_csv('../results/{}_agg_nonAC_perf_df_{}.csv'.format(args.folder, task), index=False)

    print("Model performance analysis -", args.action, "completed!")

def aggregate_perf(folder, task, grand_perf_df, split_types, mol_props, task_settings, model_names, mol_reps):
    """
    A function to generate aggregated perf_df based on grand_perf_df
    """
    # make an empty dataframe to store the agggregated results (mean and std) over all folds
    agg_perf_df = pd.DataFrame(columns=['mean_metric_score', 'std_metric_score', 'split_type', 'mol_rep', 'mol_prop', 'model_name'])

    # loop through all settings
    for split_type in split_types:
        for mol_prop in mol_props:
            # get task_setting based on mol_prop
            task_setting = task_settings[mol_props.index(mol_prop)]
            if task_setting == 'CLS':
                metric_names = ['AUROC', 'AUPRC', 'Precision_PPV', 'Precision_NPV', 'MCC', 'Cohen_Kappa', 'BEDROC', 'EF']
            elif task_setting == 'REG':
                metric_names = ['RMSE', 'MAE', 'R2', 'Pearson_R', 'MAPE']
            for model_name in model_names: 
                perf_df = grand_perf_df.loc[(grand_perf_df['split_type']==split_type) & (grand_perf_df['mol_prop']==mol_prop) & (grand_perf_df['model_name']==model_name)] 
                # note: if perf_df has 0 rows, then the test reuslt for this setting is not completed yet
                if perf_df.shape[0] == 0:
                    print("No record:", split_type, '-', mol_prop, '-', model_name)
                    continue
                # for RF, SVM, XGBoost, add one more loop over mol_reps 
                if model_name in ['RF', 'SVM', 'XGBoost', ]:
                    for mol_rep in mol_reps:
                        # add an extra step to filter out each mol_rep
                        tmp_1 = perf_df[(perf_df['mol_rep'] == mol_rep)]
                        if tmp_1.shape[0] == 0:
                            print("No record:", split_type, '-', mol_prop, '-', model_name, '-', mol_rep)
                            continue
                        for metric_name in metric_names:
                            tmp_2 = tmp_1[tmp_1['metric_name'] == metric_name]
                            values_to_add = {'metric_name': metric_name, 'mean_metric_score': tmp_2['metric_score'].mean(), 'std_metric_score': tmp_2['metric_score'].std(),\
                                    'split_type': split_type, 'mol_rep': mol_rep, 'mol_prop': mol_prop, 'model_name':model_name}
                            row_to_add = pd.Series(values_to_add)
                            agg_perf_df = agg_perf_df.append(row_to_add, ignore_index=True)
                else:
                    # get the current mol_rep value
                    mol_rep = perf_df['mol_rep'].unique()[0]
                    for metric_name in metric_names:
                        tmp = perf_df[perf_df['metric_name'] == metric_name]
                        values_to_add = {'metric_name': metric_name, 'mean_metric_score': tmp['metric_score'].mean(), 'std_metric_score': tmp['metric_score'].std(),\
                                'split_type': split_type, 'mol_rep': mol_rep, 'mol_prop': mol_prop, 'model_name':model_name}
                        row_to_add = pd.Series(values_to_add)
                        agg_perf_df = agg_perf_df.append(row_to_add, ignore_index=True)

    return agg_perf_df

def generate_row2add_molace(model_name, mol_prop, metric_name, mol_rep):
    """
     A function to generate new row(s) as a list to add to the perf_df for moleculeace datasets
    """
    test_result_df = pd.read_csv('../results/raw_predictions/{}/moleculeace/{}/fixed/{}/test_result.csv'.format(model_name, mol_prop, mol_rep)) 
    
    score = generate_score(test_result_df, metric_name)
    values_to_add = {'metric_score': score, 'metric_name': metric_name, 'mol_rep': mol_rep, 'mol_prop': mol_prop, \
                         'model_name': model_name, 'fold': 0, 'split_type': 'fixed', 'task': 'moleculeace'}
    row_to_add = pd.Series(values_to_add)

    return row_to_add 


def generate_row2add(model_name, task, mol_prop, split_type, fold, metric_name, mol_rep, AC_status=None):
    """
    A function to generate new row(s) as a list to add to the perf_df for opioids datasets
    """
    # use a separate function to directly return row for moleculeace
    if task == 'moleculeace':
        row_to_add = generate_row2add_molace(model_name, mol_prop, metric_name, mol_rep)
        return row_to_add

    # get the list of AC molecules when task is in reg/cutoff6
    if task in ['reg', 'cutoff6']:
        df =  pd.read_csv('../data/opioids/{}_reg.csv'.format(mol_prop))
        AC_molecules = list(df[df['AC_status']=='AC']['SMILES'])

    # return row for folders other than moleculeace
    if model_name == 'MolBERT':
        test_result_df = pd.read_csv('../results/raw_predictions/molbert/{}/{}/molbert/{}/test_result_fold{}.csv'.format(task, mol_prop, split_type, fold))
        if AC_status is None:
            test_result_df = test_result_df
        elif AC_status == True:
            test_result_df = test_result_df[test_result_df['SMILES'].isin(AC_molecules)]
        elif AC_status == False:
            test_result_df = test_result_df[~test_result_df['SMILES'].isin(AC_molecules)]
        try:
            score = generate_score(test_result_df, metric_name)
        except ValueError:
            score = None
        values_to_add = {'metric_score': score, 'metric_name': metric_name, 'mol_rep': 'SMILES', 'mol_prop': mol_prop, \
                         'model_name': 'MolBERT', 'fold': fold, 'split_type': split_type, 'task': task}
        row_to_add = pd.Series(values_to_add)
    elif model_name == 'GROVER':
        test_result_df = pd.read_csv('../results/raw_predictions/grover/{}/{}/grover_base/{}/test_result_fold{}.csv'.format(task, mol_prop, split_type, fold))
        if AC_status is None:
            test_result_df = test_result_df
        elif AC_status == True:
            test_result_df = test_result_df[test_result_df['SMILES'].isin(AC_molecules)]
        elif AC_status == False:
            test_result_df = test_result_df[~test_result_df['SMILES'].isin(AC_molecules)]
        try:
            score = generate_score(test_result_df, metric_name)
        except ValueError:
            score = None
        values_to_add = {'metric_score': score, 'metric_name': metric_name, 'mol_rep': 'Graph', 'mol_prop': mol_prop, \
                         'model_name': 'GROVER', 'fold': fold, 'split_type': split_type, 'task': task}
        row_to_add = pd.Series(values_to_add)
    elif model_name == 'GROVER_RDKit':
        test_result_df = pd.read_csv('../results/raw_predictions/grover/{}/{}/grover_base_rdkit/{}/test_result_fold{}.csv'.format(task, mol_prop, split_type, fold))
        if AC_status is None:
            test_result_df = test_result_df
        elif AC_status == True:
            test_result_df = test_result_df[test_result_df['SMILES'].isin(AC_molecules)]
        elif AC_status == False:
            test_result_df = test_result_df[~test_result_df['SMILES'].isin(AC_molecules)]        
        try:
            score = generate_score(test_result_df, metric_name)
        except ValueError:
            score = None
        values_to_add = {'metric_score': score, 'metric_name': metric_name, 'mol_rep': 'Graph', 'mol_prop': mol_prop, \
                         'model_name': 'GROVER_RDKit', 'fold': fold, 'split_type': split_type, 'task': task}
        row_to_add = pd.Series(values_to_add)
    elif model_name in ['RF', 'SVM', 'XGBoost']: 
        test_result_df = pd.read_csv('../results/raw_predictions/{}/{}/{}/{}/{}/test_result_fold{}.csv'.format(model_name, task, mol_prop, split_type, mol_rep, fold)) 
        if AC_status is None:
            test_result_df = test_result_df
        elif AC_status == True:
            test_result_df = test_result_df[test_result_df['SMILES'].isin(AC_molecules)]
        elif AC_status == False:
            test_result_df = test_result_df[~test_result_df['SMILES'].isin(AC_molecules)]        
        try:
            score = generate_score(test_result_df, metric_name)
        except ValueError:
            score = None
        values_to_add = {'metric_score': score, 'metric_name': metric_name, 'mol_rep': mol_rep, 'mol_prop': mol_prop, \
                         'model_name': model_name, 'fold': fold, 'split_type': split_type, 'task': task}
        row_to_add = pd.Series(values_to_add)
    elif model_name == 'RNN':
        test_result_df = pd.read_csv('../results/raw_predictions/RNN/{}/{}/{}/test_result_fold{}.csv'.format(task, mol_prop, split_type, fold))
        if AC_status is None:
            test_result_df = test_result_df
        elif AC_status == True:
            test_result_df = test_result_df[test_result_df['SMILES'].isin(AC_molecules)]
        elif AC_status == False:
            test_result_df = test_result_df[~test_result_df['SMILES'].isin(AC_molecules)]        
        try:
            score = generate_score(test_result_df, metric_name)
        except ValueError:
            score = None
        values_to_add = {'metric_score': score, 'metric_name': metric_name, 'mol_rep': 'SMILES', 'mol_prop': mol_prop, \
                         'model_name': 'RNN', 'fold': fold, 'split_type': split_type, 'task': task}
        row_to_add = pd.Series(values_to_add)
    elif model_name == 'GCN':
        test_result_df = pd.read_csv('../results/raw_predictions/GCN/{}/{}/{}/test_result_fold{}.csv'.format(task, mol_prop, split_type, fold))
        if AC_status is None:
            test_result_df = test_result_df
        elif AC_status == True:
            test_result_df = test_result_df[test_result_df['SMILES'].isin(AC_molecules)]
        elif AC_status == False:
            test_result_df = test_result_df[~test_result_df['SMILES'].isin(AC_molecules)]        
        try:
            score = generate_score(test_result_df, metric_name)
        except ValueError:
            score = None
        values_to_add = {'metric_score': score, 'metric_name': metric_name, 'mol_rep': 'Graph', 'mol_prop': mol_prop, \
                         'model_name': 'GCN', 'fold': fold, 'split_type': split_type, 'task': task}
        row_to_add = pd.Series(values_to_add)
    elif model_name == 'GIN':
        test_result_df = pd.read_csv('../results/raw_predictions/GIN/{}/{}/{}/test_result_fold{}.csv'.format(task, mol_prop, split_type, fold))
        if AC_status is None:
            test_result_df = test_result_df
        elif AC_status == True:
            test_result_df = test_result_df[test_result_df['SMILES'].isin(AC_molecules)]
        elif AC_status == False:
            test_result_df = test_result_df[~test_result_df['SMILES'].isin(AC_molecules)]        
        try:
            score = generate_score(test_result_df, metric_name)
        except ValueError:
            score = None
        values_to_add = {'metric_score': score, 'metric_name': metric_name, 'mol_rep': 'Graph', 'mol_prop': mol_prop, \
                         'model_name': 'GIN', 'fold': fold, 'split_type': split_type, 'task': task}
        row_to_add = pd.Series(values_to_add)

    return row_to_add 

def generate_score(test_result_df, metric_name):
    """
    A function to generate metric score given an input result dataframe and metric_name
    """
    # ensure preds and labels are in float type 
    test_result_df['preds'] = test_result_df['preds'].astype(float)
    test_result_df['labels'] = test_result_df['labels'].astype(float)
    # drop rows with NA values
    test_result_df.dropna(inplace=True)
    score = metric_calc(test_result_df['labels'], test_result_df['preds'], metric_name)
    return score

# add arguments    
parser = ArgumentParser()
parser.add_argument("--action", type=str, default="calc", 
                    help='Which action to take;\
                    calc: calculate performance metrics based on raw predictions \
                    agg: aggregate prediction performance to get mean and variance ')

parser.add_argument("--folder", type=str, default="opioids", help='Which group of datasets to process; options:\
                    benchmark, opioids, bender, desc')

parser.add_argument("--task_type", type=str, default="reg", help='Which task type (reg, cutoff6) to set up; only needed for opioids datasets')

if __name__ == "__main__":
    run()   