import pandas as pd
import numpy as np

from pickle import load
import torch

from utils import *
# model definition
from model import FCNet

# pipeline

with open('pipeline.pkl', 'rb') as f:
    pipeline = load(f)

# Neural Network

nn = FCNet(115, [120, 120, 120, 120], 0.1, 0.8)
nn.load_state_dict(torch.load('nn.pt', weights_only=True))

# xgb

with open('xgb.pkl', 'rb') as f:
    xgb = load(f)

# svm

with open('svm.pkl', 'rb') as f:
    svm = load(f)

# rf

with open('rf.pkl', 'rb') as f:
    rf = load(f)

# pca

with open('pca.pkl', 'rb') as f:
    pca = load(f)

# knn

with open('knn.pkl', 'rb') as f:
    knn = load(f)

# inference

# load test data
test_path = "test_class.csv"
test_csv = pd.read_csv(test_path)
# drop non-feature columns
test_ids = test_csv.id
_TEST = test_csv.drop(['id'], axis=1)
# transform using pipeline
_TEST = engineer_features(_TEST)
_TEST = pipeline.transform(_TEST)

# get predictions

predictions = {'id': test_ids}

with torch.no_grad():
    nn.cpu()
    logits = nn(torch.from_numpy(_TEST).to(torch.float32)).squeeze()
predictions['nn'] = get_preds(logits).numpy()
predictions['xgb'] = xgb.predict(_TEST)
predictions['svm'] = svm.predict(_TEST)
predictions['rf'] = rf.predict(_TEST)
predictions['knn'] = knn.predict(pca.transform(_TEST))

predictions = pd.DataFrame.from_dict(predictions)
predictions['winner'] = predictions[['nn', 'xgb', 'svm', 'rf', 'knn']].mode(axis=1)
predictions['winner'] = predictions['winner'].apply(num2prez)
predictions[['id', 'winner']].to_csv(
    f'submission.csv',
    index=False
)