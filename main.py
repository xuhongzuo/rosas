import os
import numpy as np
import pandas as pd
import torch
import glob
import argparse
import ast
import time
import utils
import yaml


parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, default='data', help='data path')
parser.add_argument('--datasets', type=str, default='pendigits',
                    help='dataset name of the path, use FULL to include all the datasets')
parser.add_argument('--algo', type=str, default="rosas")
parser.add_argument('--flag', type=str, default="")
parser.add_argument('--n_known', type=int, default=30)
parser.add_argument('--contamination', type=float, default=0.02)
parser.add_argument('--runs', type=int, default=1)
parser.add_argument('--log_avg', type=ast.literal_eval, default=False)
parser.add_argument('--use_es', type=ast.literal_eval, default=True)
parser.add_argument('--debug', action='store_true', default=False)
parser.add_argument('--res_path', type=str, default='@records/')

args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
os.makedirs('log/', exist_ok=True)
os.makedirs(args.res_path, exist_ok=True)

from RoSAS import RoSAS
params = yaml.load(open('configs.yaml', 'r'), Loader=yaml.FullLoader)['model_params']['rosas']
root_path = './'


def run_model(df, dataset_name, runs):
    model_name = args.algo

    print(f'{dataset_name}')
    print("------------------------------------ Dataset: [%s] ------------------------------------" % dataset_name)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(method='ffill', inplace=True)
    x = df.values[:, :-1]
    y = np.array(df.values[:, -1], dtype=int)

    x_train, y_train, x_test, y_test, x_val, y_val = utils.split_train_test_val(x, y,
                                                                                test_ratio=0.2,
                                                                                val_ratio=0.2,
                                                                                random_state=2021,
                                                                                del_features=True)
    semi_y = utils.semi_setting(y_train, n_known_outliers=args.n_known)

    # # this is to control contamination rate and estimate the robustness
    if args.contamination is not None:
        x_train, y_train, semi_y = utils.adjust_contamination(x_train, y_train, semi_y,
                                                              adjust_cont_r=args.contamination,
                                                              random_state=2021)

    rauc, raucpr, rtime = np.zeros(runs), np.zeros(runs), np.zeros(runs)
    for i in range(runs):
        st = time.time()

        params['use_es'] = args.use_es
        params['seed'] = 42 + i

        model = RoSAS(**params)
        param_lst = model.param_lst
        model.fit(x_train, semi_y, val_x=x_val, val_y=y_val)
        score = model.predict(x_test)

        auroc, aupr = utils.evaluate(y_test, score)
        rtime[i] = time.time() - st
        rauc[i] = auroc
        raucpr[i] = aupr

        txt = f'{dataset_name}, AUC-ROC: {auroc:.4f}, AUC-PR: {aupr:.4f}, ' \
              f'time: {rtime[i]:.1f}, runs: [{i+1}/{runs}]'

        print(txt)
        doc1 = open(args.res_path + f'@raw_{model_name}{args.flag}.csv', 'a')
        print(txt, file=doc1)
        doc1.close()

    print_text = f"{dataset_name}, AUC-ROC, {np.average(rauc):.4f}, {np.std(rauc):.4f}," \
                 f" AUC-PR, {np.average(raucpr):.4f}, {np.std(raucpr):.4f}, {np.average(rtime):.1f}," \
                 f" {runs}runs, {args.n_known}known, {args.contamination:.2f}cont," \
                 f" {model_name}, {str(param_lst)}"

    print(print_text, end='\n\n\n')

    if not args.debug:
        doc1 = open(args.res_path + f'{model_name}{args.flag}.csv', 'a')
        print(print_text, file=doc1)
        doc1.close()

    return np.average(rauc), np.average(raucpr)


if __name__ == '__main__':
    path = os.path.join(root_path, args.path)
    t1 = time.time()
    datasets_auc = []
    datasets_aupr = []

    if args.datasets == 'FULL':
        f_lst = glob.glob(os.path.join(path, '*.csv'))
        for f in sorted(f_lst):
            name = os.path.splitext(os.path.split(f)[1])[0]
            df = pd.read_csv(f)
            auroc, aupr = run_model(df, dataset_name=name, runs=args.runs)
            datasets_auc.append(auroc)
            datasets_aupr.append(aupr)
    else:
        datasets = args.datasets.split(',')
        for d in datasets:
            f = glob.glob(os.path.join(path, f'*{d}*.csv'))
            assert len(f) == 1
            f = f[0]

            name = os.path.splitext(os.path.split(f)[1])[0]
            df = pd.read_csv(f)
            auroc, aupr = run_model(df, dataset_name=name, runs=args.runs)
            datasets_auc.append(auroc)
            datasets_aupr.append(aupr)

    avg1 = np.average(datasets_auc)
    avg2 = np.average(datasets_aupr)
    avg = f"avg, AUC-ROC, {avg1:.3f}, AUC-PR, {avg2:.3f}, {time.time()-t1:.1f}s"
    print(avg)

    if args.log_avg and not args.debug:
        doc = open(args.res_path + f'{args.algo}{args.flag}.csv', 'a')
        print("", file=doc)
        print(avg, file=doc)
        doc.close()

