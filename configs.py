#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: MingQing Liu
# Date: 2024/7/19 17:30
from yacs.config import CfgNode as CN

_C = CN()

_C.Data = CN()
_C.Data.Path = './datasets'
_C.Data.KFold = 5

# Drug feature extractor
_C.Drug = CN()
_C.Drug.Node_In_Feat = 75
_C.Drug.Padding = True
_C.Drug.Hidden_Layers = [128, 128, 128]
_C.Drug.Node_In_Embedding = 128
_C.Drug.Max_Nodes = 290
_C.Drug.GCN_Activation = None

# Protein feature extractor
_C.Protein = CN()
_C.Protein.Num_Filters = [128, 128, 128]
_C.Protein.Kernel_Size = [3, 6, 9]
_C.Protein.Embedding_Dim = 128
_C.Protein.Padding = True
_C.Protein.Max_Length = 1200
_C.Protein.After_CNN_Length = 1185

# MBCA setting
_C.MBCA = CN()
_C.MBCA.Hidden_Size = 128
_C.MBCA.Num_Layers = 12
_C.MBCA.Num_Heads = 12
_C.MBCA.Attn_Dropout_Rate = 0.3

# MLP decoder
_C.MLP = CN()
_C.MLP.In_Dim = 128
_C.MLP.Hidden_Dim = 512
_C.MLP.Out_Dim = 64
_C.MLP.Binary = 2

# Global
_C.Global = CN()
_C.Global.Max_Epoch = 100
_C.Global.Batch_Size = 64
_C.Global.LR = 5e-5
_C.Global.Seed = 2048

# Result
_C.Result = CN()
_C.Result.Output_Dir = "./output"
_C.Result.Save_Model = True


def get_cfg_defaults():
    return _C.clone()
