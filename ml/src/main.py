# file to operate the training and prediction
import os
import sys

sys.path.append(os.path.join(os.path.abspath(os.curdir),'ml'))

from models.train_model import ml_model_training

# Train the model (Just change the params)
ml_model_training(model='naive_bayes', save_report=False)



