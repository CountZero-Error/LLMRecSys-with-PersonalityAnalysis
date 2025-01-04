from sentence_transformers import SentenceTransformer, util
from nltk.corpus import wordnet
from datasets import Dataset
from tqdm import tqdm
import pandas as pd
import random
import torch
import nltk
import os

class ssgb_preprocessing:
    def __init__(self, df):
        self.df = df

    def compute_embedding_consistency_loss(self, original_embeddings, augmented_embeddings):
        # calculate Euclidean distance for consistency_loss
        consistency_loss = torch.mean((original_embeddings - augmented_embeddings) ** 2)
        return consistency_loss










    def run(self):


        lebeled_data_df = self.df[self.df['labels'].notna()]
        unlebeled_data_df = self.df[self.df['labels'].isna()]