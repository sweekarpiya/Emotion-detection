import os
import sys
import pandas as pd
sys.path.append(os.path.join(os.path.abspath(os.curdir),'ml'))

from data.get_dataset import fetch_dataset

# download the dataset
dataset = fetch_dataset()

print(dataset.head())

