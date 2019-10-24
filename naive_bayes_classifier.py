from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 
from collections import Counter
import numpy as np
class NB:
    def __init__(self,reviews,labels):
        self.reviews = reviews
        self.labels = labels
        
        assert(len(self.reviews) == len(self.labels))
        
        self.pos_cp = Counter()  
        self.neg_cp = Counter()
        self.prior_pos = 0
        self.prior_neg = 0
        self.vocab = []
        self.count_vocab = 0
        self.positive_counts = Counter()
        self.negative_counts = Counter()
        self.count_positive_words = 0
        self.count_negative_words = 0
        self.pos_label_count = 0
        self.neg_label_count = 0
        
        self.process(self.reviews[:40000], self.labels[:40000])
        self.fit(self.reviews[10000:], self.labels[10000:])
        
        
    
    def clean(self,text):
        '''
            Returns the clean text after removing stop words
            :params: uncleaned text review
            :out_params: cleaned review
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
    
    def process(self,reviews,labels):
        total_counts = Counter()

        for i in range(len(reviews)):
            review = self.clean(reviews[i])  # clean the reviews first then process
            if(labels[i] == 'POSITIVE'):
                self.pos_label_count += 1
                for word in review:
                    self.positive_counts[word] += 1      # all positive words with their counts
                    total_counts[word] += 1
            else:
                self.neg_label_count += 1
                for word in review:
                    self.negative_counts[word] += 1      # all negative words with their counts
                    total_counts[word] += 1
                    
        self.vocab = list(total_counts.keys())  # vocabulary
        self.count_vocab = len(self.vocab)           # total no of vocabulary
        self.count_positive_words = len(self.positive_counts)
        self.count_negative_words = len(self.negative_counts)
        
        # call train after process
        self.train()
    
    def train(self):    
  
        # finding conditional probability of each word for each class
        for word in self.vocab:
            self.pos_cp[word] = np.log((self.positive_counts[word] + 1) / (self.count_positive_words + self.count_vocab))
            self.neg_cp[word] = np.log((self.negative_counts[word] + 1) / (self.count_negative_words + self.count_vocab))
        
        # finding prior of each class
        self.prior_pos = np.log(self.pos_label_count / len(self.labels))   # p(C==c) = Nc / N  where N = total labels/reviews
        self.prior_neg = np.log(self.neg_label_count / len(self.labels))
    
    def test(self,review):
        review = self.clean(review)
        test_pos = self.prior_pos
        test_neg = self.prior_neg
        for word in review:
            if word in self.vocab:
                test_pos += self.pos_cp[word]
                test_neg += self.neg_cp[word]
        return ('POSITIVE' if test_pos > test_neg else 'NEGATIVE') 
    
    def fit(self, test_set_reviews, test_set_labels):
        correct = 0
        for r,l in zip(test_set_reviews, test_set_labels):
            test_value = self.test(r)
            if(test_value == l):
                correct += 1 
        print("total correct: {} and incorrect: {}".format(correct,len(test_set_labels)-correct))

import csv
filename = 'IMDB_dataset_sa.csv'
with open(filename, 'r') as csvfile:
    csvreader = csv.reader(csvfile)
    reviews = []
    labels = []
    # extract rows
    i= 0
    for row in csvreader:
        if csvreader.line_num == 1:
            continue  #skip first row
        reviews.append(row[0])
        labels.append(row[1].upper())
    print("Done.")

nb = NB(reviews,labels)

