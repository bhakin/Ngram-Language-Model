import pandas as pd
import numpy as np
from pathlib import Path

class UnigramLM(object):
    
    def __init__(self, tokens):
        self.tokens = tokens
        self.mdl = self.train(tokens)
    
    def train(self, tokens):
        output = pd.Series(tokens).value_counts()
        return output/len(tokens)
    
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
    

class NGramLM(object):         
    
    def __init__(self, N, tokens):
        self.N = N

        ngrams = self.create_ngrams(tokens)

        self.ngrams = ngrams
        self.mdl = self.train(ngrams)

        if N < 2:
            raise Exception('N must be greater than 1')
        elif N == 2:
            self.prev_mdl = UnigramLM(tokens)
        else:
            self.prev_mdl = NGramLM(N-1, tokens)

    def create_ngrams(self, tokens):
        output = []
        for n in range(len(tokens) - self.N+1):
            curr = tuple(tokens[n:n+self.N])
            output.append(curr)
        return output
        
    def train(self, ngrams):
        # N-Gram counts C(w_1, ..., w_n)
        ngrams_series = pd.Series(ngrams).value_counts()
        
        # (N-1)-Gram counts C(w_1, ..., w_(n-1))
        n1grams = [n[:-1] for n in ngrams]
        n1grams_series = pd.Series(n1grams).value_counts()

        # Create the conditional probabilities
        prob = []
        for n in ngrams:
            n1gram_num = n[:-1]
            n1gram_value = n1grams_series[n1gram_num]
            ngram_value = ngrams_series[n]
            prob.append(ngram_value/n1gram_value)

        # Put it all together
        df = pd.DataFrame()
        df['ngram'] = ngrams
        df['n1gram'] = n1grams
        df['prob'] = prob

        return df.drop_duplicates()     
    
    def probability(self, words):
        prob = 1

        #This gets the recursive steps 
        length = self.N - 1
        prev = self.prev_mdl.probability(words[:length])

        #Base Case: getting the trigrams values
        for n in range(len(words) - self.N + 1):
            ngram = tuple(words[n:n+self.N])
            n1gram = ngram[:-1]         
            ngram_df = self.mdl[self.mdl['ngram'] == ngram]
            n1gram_df = ngram_df[ngram_df['n1gram'] == n1gram]
            if len(n1gram_df.index) == 0:
                return 0
            curr = n1gram_df['prob'].values[0]
            prob *= curr     

        total = prev * prob
        return total

    def sample(self, M):
        def length(curr_mdl, word):
            model = curr_mdl[curr_mdl['n1gram'] == word]
           
            if model.shape[0] == 0:
                return '\x03'
            ngrams = model['ngram'].values
            prob = model['prob'].values
            choice = np.random.choice(ngrams, size=1, p=prob)
            return choice[0][-1]

        # Transform the tokens to strings
        output = ['\x02']
        previous_word = '\x02'
        curr_gram = 2
        for n in range(M-1):
            if n == 0:
                if self.N <= 2:
                    prev = self.mdl
                    result = length(self.mdl, tuple(previous_word))
                    output.append(result)
              
                else: 
                    prev = self
                    for i in range(self.N - curr_gram):
                        prev = prev.prev_mdl
                    curr_gram += 1
                      
                    result = length(prev.mdl, tuple(previous_word))
                    output.append(result)

            else: 
                mdl = self
                if curr_gram != self.N:
                    for i in range(self.N - curr_gram):
                        mdl = mdl.prev_mdl
                    outcome = length(mdl.mdl, tuple(output[-(self.N-1):]))
                    curr_gram += 1

                else:
                    outcome = length(self.mdl, tuple(output[-(self.N-1):])) 
                output.append(outcome)
   
        output.append('\x03')       
        string = ' '.join(output) 
        
        return string