# file to operate the training and prediction
import os
import sys

sys.path.append(os.path.join(os.path.abspath(os.curdir),'ml'))

from models.train_model import ml_model_training

ml_model_training(model='softmax_l2', save_report=True)



