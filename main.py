import pandas as pd
import numpy as np
import copy
import os
import argparse
import random
from tqdm import tqdm
import torch
from torch import nn
from torch.optim import AdamW, lr_scheduler
from torch.nn import CrossEntropyLoss, MSELoss
from torch.utils.data import DataLoader, Dataset
from peft import LoraConfig, get_peft_model

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import torch.nn.functional as F
from sklearn.model_selection import train_test_split

from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score, MulticlassMatthewsCorrCoef, MulticlassAUROC
import warnings
import datetime
from transformers import BertModel, BertTokenizer



parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', '-d', default="datasets/", help='dataset directory', type=str)
parser.add_argument('--out_dir', '-o', default="results/", help='results directory', type=str)
parser.add_argument('--n_classes', '-n', default=6, help='number of classes', type=int)
parser.add_argument('--batch_size', '-b', default=4, help='batch size', type=int)
parser.add_argument('--learning_rate', '-l', default=1e-4, help='learning rate', type=float)
parser.add_argument('--epoch', '-e', default=20, help='epoch', type=int)
parser.add_argument('--fine_tune', '-ft', default=False, help='true or false for best weights', type=bool)
parser.add_argument('--mode', '-m', default="train", help="Choose mode, either train or instance")

args = parser.parse_args()
### hyper parameter
N_classes = args.n_classes
BATCH_SIZE = args.batch_size
learning_rate = args.learning_rate
EPOCH = args.epoch


print("dataset input path: ", args.data_dir)
print("results output path: ", args.out_dir)
print("number of classes: ", args.n_classes)
print("batch size: ", args.batch_size)
print("learning rate: ", args.learning_rate)
print("epoch: ", args.epoch)

warnings.filterwarnings("ignore")

main_train_csv_folder_path = "datasets/bacteria_inference/"
main_output_save_folder_path = "results/bacteria_inference/"



    
