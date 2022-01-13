import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from utils.run_fold import run_fold
from utils.functions import init_logger
from utils.preprocessing import make_crop

import os
import sys
import hydra
from tqdm import tqdm
from omegaconf import DictConfig

from joblib import Parallel, delayed

@hydra.main(config_path="config", config_name="JPO.yaml")
def main(cfg):
    df_train = pd.read_csv(hydra.utils.get_original_cwd()+'/data/train.csv')
    df_test = pd.read_csv(hydra.utils.get_original_cwd()+'/data/test.csv')
    df_cite = pd.read_csv(hydra.utils.get_original_cwd()+'/data/cite_v2.csv')

    input_dir = hydra.utils.get_original_cwd()+'/data/'
    
    # 学習用出願画像の処理
    apply_img_paths = df_train['path'].values
    apply_img_dir = input_dir + 'apply_images/'
    apply_img_input_paths = [apply_img_dir+path for path in apply_img_paths]
    apply_img_output_paths = []
    inputs = []
    for apply_img_input_path in tqdm(apply_img_input_paths):
        apply_img_output_dir = input_dir + 'crop_train_apply_images/' + apply_img_input_path.split('/')[-2]
        if os.path.exists(apply_img_output_dir):
            continue
        apply_img_output_path = input_dir + 'crop_train_apply_images/'+'/'.join(apply_img_input_path.split('/')[-2:])
        apply_img_output_paths.append(apply_img_output_path)
        inputs.append([apply_img_input_path, apply_img_output_path, apply_img_output_dir])
    _ = Parallel(n_jobs=-1)(delayed(make_crop)(input) for input in tqdm(inputs))
    df_train['path'] = input_dir + 'crop_train_apply_images/'+df_train['path']
    
    # 学習用引用画像の処理
    cite_img_paths = df_train['cite_path'].values
    cite_img_dir = input_dir + 'cite_images/'
    cite_img_input_paths = [cite_img_dir+path for path in cite_img_paths]
    cite_img_output_paths = []
    inputs = []
    for cite_img_input_path in tqdm(cite_img_input_paths):
        cite_img_output_dir = input_dir + 'crop_train_cite_images/' + cite_img_input_path.split('/')[-2]
        if os.path.exists(cite_img_output_dir):
            continue
        cite_img_output_path = input_dir + 'crop_train_cite_images/'+'/'.join(cite_img_input_path.split('/')[-2:])
        cite_img_output_paths.append(cite_img_output_path)
        inputs.append([cite_img_input_path, cite_img_output_path, cite_img_output_dir])
    _ = Parallel(n_jobs=-1)(delayed(make_crop)(input) for input in tqdm(inputs))
    df_train['path'] = input_dir + 'crop_train_cite_images/'+df_train['path']
    
    # テスト用出願画像の処理
    apply_img_paths = df_test['path'].values
    apply_img_dir = input_dir + 'apply_images/'
    apply_img_input_paths = [apply_img_dir+path for path in apply_img_paths]
    apply_img_output_paths = []
    inputs = []
    for apply_img_input_path in tqdm(apply_img_input_paths):
        apply_img_output_dir = input_dir + 'crop_test_apply_images/' + apply_img_input_path.split('/')[-2]
        if os.path.exists(apply_img_output_dir):
            continue
        apply_img_output_path = input_dir + 'crop_test_apply_images/'+'/'.join(apply_img_input_path.split('/')[-2:])
        apply_img_output_paths.append(apply_img_output_path)
        inputs.append([apply_img_input_path, apply_img_output_path, apply_img_output_dir])
    _ = Parallel(n_jobs=-1)(delayed(make_crop)(input) for input in tqdm(inputs))
    df_test['path'] = input_dir + 'crop_test_apply_images/'+df_test['path']

    # 引用画像の処理
    cite_img_paths = df_cite['path'].values
    cite_img_dir = input_dir + 'cite_images/'
    cite_img_input_paths = [cite_img_dir+path for path in cite_img_paths]
    cite_img_output_paths = []
    inputs = []
    for cite_img_input_path in tqdm(cite_img_input_paths):
        cite_img_output_dir = input_dir + 'crop_cite_images/' + cite_img_input_path.split('/')[-2]
        if os.path.exists(cite_img_output_dir):
            continue
        cite_img_output_path = input_dir + 'crop_cite_images/'+'/'.join(cite_img_input_path.split('/')[-2:])
        cite_img_output_paths.append(cite_img_output_path)
        inputs.append([cite_img_input_path, cite_img_output_path, cite_img_output_dir])
    _ = Parallel(n_jobs=-1)(delayed(make_crop)(input) for input in tqdm(inputs))
    df_cite['path'] = input_dir + 'crop_cite_images/'+df_cite['path']

if __name__ == '__main__':
    main()