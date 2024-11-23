#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: MingQing Liu
# Date: 2024/7/19 17:17
import argparse
import os
import time
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from configs import get_cfg_defaults
from utils import set_seed, mkdir, graph_collate_func
from dataloader import DTIDataset
from model import Redumixdti
from trainer import Trainer

parser = argparse.ArgumentParser(description='Drug-Target Interaction Prediction')
parser.add_argument('--cfg', required=True, type=str)  # ./configs/ReduMixDTI.yaml
parser.add_argument('--outname', required=True, type=str)  # model_training
parser.add_argument('--data', required=True, type=str)  # human / biosnap / bindingdb
parser.add_argument('--num_worker', required=True, type=int)  # 0
args = parser.parse_args()


def main():
    torch.cuda.empty_cache()
    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.cfg)
    set_seed(cfg.Global.Seed)
    print(f"Config yaml: {args.cfg}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running on: {device}")

    datafolder = f'{cfg.Data.Path}/{args.data}'
    outfolder = f'./output/{args.outname}' if device == torch.device(
        'cpu') else f'{cfg.Result.Output_Dir}/{args.outname}'
    mkdir(outfolder)
    print(f'datafolder: {datafolder}\noutput_folder: {outfolder}')

    # load data
    train_data = pd.read_csv(os.path.join(datafolder, 'train.csv'))
    val_data = pd.read_csv(os.path.join(datafolder, 'val.csv'))
    test_data = pd.read_csv(os.path.join(datafolder, 'test.csv'))
    train_dataset = DTIDataset(train_data.index.values, train_data, max_drug_nodes=cfg.Drug.Max_Nodes,
                               max_protein_length=cfg.Protein.Max_Length)
    val_dataset = DTIDataset(val_data.index.values, val_data, max_drug_nodes=cfg.Drug.Max_Nodes,
                             max_protein_length=cfg.Protein.Max_Length)
    test_dataset = DTIDataset(test_data.index.values, test_data, max_drug_nodes=cfg.Drug.Max_Nodes,
                              max_protein_length=cfg.Protein.Max_Length)

    print('*' * 25, 'Begin training', '*' * 25)

    params = {'batch_size': cfg.Global.Batch_Size, 'shuffle': True, 'num_workers': args.num_worker,
              'drop_last': True, 'collate_fn': graph_collate_func}

    train_generator = DataLoader(train_dataset, **params)
    params['shuffle'] = False
    params['drop_last'] = False
    val_generator = DataLoader(val_dataset, **params)
    test_generator = DataLoader(test_dataset, **params)

    # load model
    model = Redumixdti(cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.Global.LR)
    loss = nn.CrossEntropyLoss()

    with open(os.path.join(outfolder, "model_configs.txt"), "w") as f:
        f.write(str(cfg))
    with open(os.path.join(outfolder, "model_architecture.txt"), "w") as wf:
        wf.write(str(model))

    start = time.time()
    trainer = Trainer(model, opt, loss, device, train_generator, val_generator, test_generator, cfg, outfolder)
    trainer.set_tensorboard(path=outfolder)
    trainer.train()
    end = time.time()

    print(f"End! Total running time: {round((end - start) / 60, 2)} min")


if __name__ == '__main__':
    main()
