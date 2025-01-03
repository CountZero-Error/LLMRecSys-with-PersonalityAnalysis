from sklearn.preprocessing import StandardScaler
from datasets import load_dataset
import networkx as nx
import pandas as pd
import numpy as np

class ssgb_preprocessing:
    def __init__(self, df):
        self.df = df


    def run(self):
        lebeled_data_df = self.df[self.df['labels'].notna()]
        unlebeled_data_df = self.df[self.df['labels'].isna()]