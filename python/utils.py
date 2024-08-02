import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd

class CustomDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx) -> tuple[torch.Tensor, torch.float32]:
        return self.X[idx], self.y[idx]
    
def num2prez(num: int) -> str:
    '''
    Given 0 or 1, returns 'Trump' or 'Biden'
    '''
    assert num in {0, 1}, "num must be 0 or 1!"
    return 'Trump' if num == 0 else 'Biden'

def prez2num(prez: str) -> int:
    '''
    Given 'Trump' or 'Biden', returns 0 or 1
    '''
    assert prez in {'Trump', 'Biden'}, "prez must be 'Trump' or 'Biden'!"
    return 0 if prez == 'Trump' else 1

def convert(input: np.ndarray) -> np.ndarray:
    '''
    Given a np.ndarray of 'Trump'/'Biden' strs, return a np.ndarray of 0/1 ints, OR
    Given a np.ndarray of 0/1 ints, return a np.ndarray of 'Trump'/'Biden' strs.
    '''
    if type(input[0]) is str:
        return np.vectorize(prez2num)(input)
    else:
        return np.vectorize(num2prez)(input)
    
def get_preds(logits: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    '''
    Given a torch.Tensor of logits, return 0 for logit <= threshold and 1 otherwise
    '''
    return (logits > threshold).int()

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    '''
    Applies feature engineering
    '''
    # Age group ratios
    df['ratio_under_18'] = df['x0019e'] / df['x0001e']
    df['ratio_18_over'] = df['x0021e'] / df['x0001e']
    df['ratio_65_over'] = df['x0024e'] / df['x0001e']

    # Gender ratios
    df['ratio_male'] = df['x0002e'] / df['x0001e']
    df['ratio_female'] = df['x0003e'] / df['x0001e']

    # Race/Ethnicity ratios
    df['ratio_white'] = df['x0037e'] / df['x0001e']
    df['ratio_black'] = df['x0038e'] / df['x0001e']
    df['ratio_asian'] = df['x0044e'] / df['x0001e']
    df['ratio_hispanic'] = df['x0071e'] / df['x0001e']

    # Education level ratios
    df['ratio_less_hs_18_24'] = df['c01_002e'] / df['c01_001e']
    df['ratio_hs_grad_18_24'] = df['c01_003e'] / df['c01_001e']
    df['ratio_some_college_18_24'] = df['c01_004e'] / df['c01_001e']
    df['ratio_bach_18_24'] = df['c01_005e'] / df['c01_001e']

    # Income and GDP
    df['income_change_2017'] = df['income_per_cap_2017'] - df['income_per_cap_2016']
    df['income_change_2018'] = df['income_per_cap_2018'] - df['income_per_cap_2017']
    df['avg_income_per_cap'] = df[[
        'income_per_cap_2016', 
        'income_per_cap_2017', 
        'income_per_cap_2018', 
        'income_per_cap_2019', 
        'income_per_cap_2020'
    ]].mean(axis=1)

    df['gdp_change_2017'] = df['gdp_2017'] - df['gdp_2016']
    df['gdp_change_2018'] = df['gdp_2018'] - df['gdp_2017']
    df['avg_gdp'] = df[['gdp_2016', 'gdp_2017', 'gdp_2018', 'gdp_2019', 'gdp_2020']].mean(axis=1)

    # Voter turnout
    df['voter_turnout'] = df['total_votes'] / df['x0001e']

    # Log transformation
    df['log_income'] = np.log1p(df['avg_income_per_cap'])
    df['log_gdp'] = np.log1p(df['avg_gdp'])

    # Interaction terms
    df['income_x_urban'] = df['avg_income_per_cap'] * df['x2013_code']
    df['education_x_income'] = df['ratio_bach_18_24'] * df['avg_income_per_cap']

    # List of features to remove
    features_to_remove = [
        # Duplicates
        'x0033e', 'x0036e',
        # Population estimates
        'x0001e', 'x0002e', 'x0003e', 'x0019e', 'x0021e', 'x0024e',  
        # Age groups
        'x0005e', 'x0006e', 'x0007e', 'x0008e', 'x0009e', 'x0010e', 
        'x0011e', 'x0012e', 'x0013e', 'x0014e', 'x0015e', 'x0016e', 'x0017e',  
        # Income years
        'income_per_cap_2016', 'income_per_cap_2017', 'income_per_cap_2018', 
        'income_per_cap_2019', 'income_per_cap_2020',  
        # GDP years
        'gdp_2016', 'gdp_2017', 'gdp_2018', 'gdp_2019', 'gdp_2020',  
        # Detailed race estimates (optional)
        'x0040e', 'x0041e', 'x0042e', 'x0043e', 'x0045e', 'x0046e',  
    ]

    # Remove features from DataFrame
    df = df.drop(columns=features_to_remove)

    return df