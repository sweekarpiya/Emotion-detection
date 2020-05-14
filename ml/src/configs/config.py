import os

ROOT_DIR = os.path.abspath(os.curdir)

DATA_DIR = os.path.join(ROOT_DIR, 'ml/data/external')

DATASET = 'ISEAR_dataset.csv'

DATASET_URL = 'https://docs.google.com/uc?export=download&id=11VDtm3zI012vFKHiuaJqwsWVwYz_prxF'

MODEL_PATH = os.path.join(ROOT_DIR, 'ml/model')

CHECKPOINT_PATH = os.path.join(BASE_DIR, "checkpoints")