import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    StandardScaler,
    OneHotEncoder,
)
from imblearn.over_sampling import RandomOverSampler

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from utils import *
from model import FCNet
from train import solver

SEED = 42

device = torch.device(
    "cuda" if torch.cuda.is_available() else
    'mps' if torch.backends.mps.is_available() else
    "cpu"
)

print('loading training data...')

train_path = "../../ucla-stats-101-c-2024-summer-classification/train_class.csv"
train_csv = pd.read_csv(train_path)

_X = train_csv.drop(['name', 'winner'], axis=1)
_y = convert(train_csv.winner.to_numpy())

ros = RandomOverSampler(sampling_strategy='minority')
_X, y = ros.fit_resample(_X, _y)

_X = _X.drop(['id'], axis=1)
_X = engineer_features(_X)

print('building pipeline...')

categorical_columns = [
    'x2013_code'
]

one_hot_enc = OneHotEncoder(handle_unknown='ignore')

categorical_pipeline = Pipeline(
    [
        ('encoder', one_hot_enc)
    ]
)

numerical_columns = list(_X.loc[:, _X.columns != 'x2013_code'].columns)

imputer = SimpleImputer()
scaler = StandardScaler()

numerical_pipeline = Pipeline(
    [
        ('imputer', imputer),
        ('std_scaler', scaler)
    ]
)

pipeline = ColumnTransformer(
    [
        ('numerical', numerical_pipeline, numerical_columns),
        ('categorical', categorical_pipeline, categorical_columns)
    ]
)

pipeline.fit(_X)

from pickle import dump

with open("pipeline.pkl", "wb") as f:
    dump(pipeline, f, protocol=5)

X = pipeline.transform(_X)

# Neural Network

print('training nn...')

BATCH_SIZE = 32
LR = 5e-4
EPOCHS = 10

train_dataset = CustomDataset(X, y)

train_dataloader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE
)

nn = FCNet(115, [120, 120, 120, 120], 0.1, 0.8)

_1, _2 = solver(
    nn,
    train_dataloader, 
    None,
    LR,
    10,
    False,
    100
)

_1, _2 = solver(
    nn,
    train_dataloader, 
    None,
    1e-5,
    10,
    False,
    100
)

nn.cpu()
torch.save(nn.state_dict(), 'nn.pt')

# xgboost

print('training xgb...')

import xgboost as xgb

xgb = xgb.XGBClassifier(
    n_estimators=150, 
    learning_rate=0.1, 
    max_depth=12
)
xgb.fit(X, y)

with open("xgb.pkl", "wb") as f:
    dump(xgb, f, protocol=5)

# SVM

print('training svm...')

from sklearn.svm import SVC

svm = SVC(
    C=10,
    kernel='rbf',
    probability=True
)
svm.fit(X, y)

with open("svm.pkl", "wb") as f:
    dump(svm, f, protocol=5)

# rf

print('training rf...')

from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(
    n_estimators=100,
    criterion='entropy',
    max_depth=9,
    max_features=None
)
rf.fit(X, y)

with open("rf.pkl", "wb") as f:
    dump(rf, f, protocol=5)

# pca + knn

print('training pca + knn...')

from sklearn.decomposition import PCA

pca = PCA(43)
pca.fit(X)
X_pca = pca.transform(X)

with open("pca.pkl", "wb") as f:
    dump(pca, f, protocol=5)

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(
    n_neighbors=11,
    weights='distance'
)
knn.fit(X_pca, y)

with open("knn.pkl", "wb") as f:
    dump(knn, f, protocol=5)

print('all done!')