import os
import torch
import glob
import numpy as np 
import random
import math
from os import listdir
from os.path import isfile, join



processed_dir="../human_features/processed/"
print(os.getcwd())
npy_file = "../human_features/npy_file_new(human_dataset).npy"
print(npy_file)

try:
  npy_ar = np.load('../human_features/npy_file_new(human_dataset).npy')
  print(npy_ar.shape)
except FileNotFoundError:
  print("File not found:",npy_file)

from torch.utils.data import Dataset as Dataset_n
from torch_geometric.data import DataLoader as DataLoader_n

