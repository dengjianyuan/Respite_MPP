import os
from datetime import datetime

import numpy as np
import pandas as pd

from utils import get_data_splits
from dataset import generate_reps

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import LinearSVC, LinearSVR
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier

from argparse import ArgumentParser

### run experiments
def run():
    # parse args
    args = parser.parse_args()

    # specify task type - currently only 3 moleculenet datasets are CLS task
    if args.mol_prop in ['BACE', 'BBBP', 'HIV'] or args.task_type == 'cutoff6': 
        task_setting = 'CLS'
    else:
        task_setting = 'REG' 

    print(datetime.now().strftime("%d/%m/%Y %H:%M:%S"), "Loading data...")
    # get SMILES list and label list for training and test sets
    train_smiles, train_label, _, _, test_smiles, test_label, task, _ = get_data_splits(args.mol_prop, args.split_type, args.seed, args.task_type)  

    # convert SMILES list and label list to np arrays
    X_train, Y_train = np.array(train_smiles), np.array(train_label)
    X_test, Y_test = np.array(test_smiles), np.array(test_label)

    print(datetime.now().strftime("%d/%m/%Y %H:%M:%S"), "Generating fixed representations...")
    # get fingerprints for molecules
    X_train_fp, X_test_fp = [generate_reps(x, args.mol_rep) for x in X_train], [generate_reps(x, args.mol_rep) for x in X_test]

    # convert list of fignerprints to np arrays via stacking
    X_train_fp, X_test_fp = np.stack(X_train_fp), np.stack(X_test_fp)

    # initialize ml model
    if task_setting == 'REG':
        # make regressors
        if args.model_name == 'RF':
            model = RandomForestRegressor(n_estimators=500, random_state=42) 
        elif args.model_name == 'SVM':
            model = LinearSVR(random_state=42)
        elif args.model_name == 'XGBoost':
            model = GradientBoostingRegressor(random_state=42)
    elif task_setting == 'CLS':
        # make classifiers
        if args.model_name == 'RF':
            model = RandomForestClassifier(n_estimators=500, random_state=42)
        elif args.model_name == 'SVM':
            model = CalibratedClassifierCV(LinearSVC(random_state=42))
        elif args.model_name == 'XGBoost':
            model = GradientBoostingClassifier(random_state=42)

    print(datetime.now().strftime("%d/%m/%Y %H:%M:%S"), "Fitting model...")
    # train model
    model.fit(X_train_fp, Y_train)

    print(datetime.now().strftime("%d/%m/%Y %H:%M:%S"), "Making prediction...")
    # get predictions for the test set
    Y_pred = model.predict(X_test_fp)
    if task_setting == 'REG':
        #assemble the test_result_df by collecting prediction results for each molecule
        test_result_df = pd.DataFrame({'preds': Y_pred, 'labels': Y_test, 'SMILES': X_test}, columns=['preds', 'labels', 'SMILES'])
    elif task_setting == 'CLS':
        # get class probability
        Y_scores = model.predict_proba(X_test_fp)[:, 1] 
        #assemble the test_result_df by collecting prediction probability for each molecule
        test_result_df = pd.DataFrame({'preds': Y_scores, 'labels': Y_test, 'SMILES': X_test}, columns=['preds', 'labels', 'SMILES'])

    # add other experiment settings
    test_result_df['mol_prop'], test_result_df['model_name'], test_result_df['split_type'], test_result_df['fold'] = args.mol_prop, args.model_name, args.split_type, args.seed
    test_result_df['mol_rep'] = args.mol_rep

    # make a directory to save test result
    try:
        os.makedirs('../results/raw_predictions/{}/{}/{}/{}/{}'\
        .format(args.model_name, task, args.mol_prop, args.split_type, args.mol_rep))
    except FileExistsError: 
        print("Directory already made!")

    print(datetime.now().strftime("%d/%m/%Y %H:%M:%S"), "Writing result...")
    # save to csv file
    test_result_df.to_csv('../results/raw_predictions/{}/{}/{}/{}/{}/test_result_fold{}.csv'\
        .format(args.model_name, task, args.mol_prop, args.split_type, args.mol_rep, args.seed), index=False)
    
# add arguments    
parser = ArgumentParser()
parser.add_argument("--task_type", type=str, default="reg", help='Adjust task type for opioids-related datasets; options: reg, cutoff6')
parser.add_argument("--mol_prop", type=str, default="ESOL")
parser.add_argument("--model_name", type=str, default="RF", help='ML models; options: RF, SVM, XGBoost')
parser.add_argument("--mol_rep", type=str, default="morganBits", help='Fixed molecular representations; options: morganBits, morganCounts, maccs, physchem, rdkit2d, atomPairs')
parser.add_argument("--split_type", type=str, default="scaffold", help='Dataset split types; options: random, scaffold')
parser.add_argument("--seed", type=str, default="0", help='Seeds for dataset splitting; options: 0-29')

if __name__ == "__main__":
    run()   