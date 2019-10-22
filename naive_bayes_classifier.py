

from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 
from collections import Counter
import numpy as np 
import string
class naive_bayes_classifier:
    def __init__(self):
        self.positive_counts = Counter()
        self.negative_counts = Counter()
        self.total_counts = Counter()
        self.pos_label_count = 0 # for calc prior p(pos)
        self.neg_label_count = 0 # for calc prior p(neg)

    def clean_review_text(self,text):
        '''
            Returns the clean text after removing stop words
            :params: uncleaned text review
            :out_params: a filtered text review
        '''
        text = text.lower()
        stop_words = set(stopwords.words('english'))
        stop_words.add(',')
        stop_words.add('.')
        stop_words.add('br')
        stop_words.add('”')

        word_tokens = word_tokenize(text)

        filtered_sentence = [w for w in word_tokens if not w in stop_words]
        filtered_sentence = []
        for w in word_tokens:
            if w not in stop_words and w.find('\'') != 0:
                filtered_sentence.append(w)
        return filtered_sentence

    def counts(self,reviews,labels):
        pass

    def find_prior(self):
        pass
    
    def test(self,review):
        pass


# load review and label files




