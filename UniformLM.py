import pandas as pd
import numpy as np
from pathlib import Path

class UniformLM(object):


    def __init__(self, tokens):
        self.tokens = tokens
        self.mdl = self.train(tokens)
        
    def train(self, tokens):      
        unique_words = np.unique(tokens)
        output = pd.Series(unique_words).value_counts()
        return output/len(unique_words) 
    
    def probability(self, words):
        prob = 1
        valid_words = self.mdl.index.to_list()
        for n in words:
            if n in valid_words:
                curr = self.mdl.loc[n]
                prob *= float(curr)

        if prob == 1:
            prob = 0

        return prob
        
    def sample(self, M):
        return ' '.join(np.random.choice(self.mdl.index, size=M, p=self.mdl))   