# Vader Sentiment Analysis

# Import pkgs
from bs4 import BeautifulSoup
import pandas as pd
import nltk
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# read in the review.csv file from appropriate directory
# total of 1873 reviews here
df = pd.read_csv('reviews.csv')
df.count(), df.columns

# extract the text from html in body column and store in reviewtext column
df['reviewtext'] = df['body'].apply(lambda x: BeautifulSoup(x, 'lxml').get_text())

# tokenize the text and store as new column, can clean this further later
df['tokenized_text'] = df.apply(lambda row: nltk.word_tokenize(row['reviewtext']), axis=1)

# initialize the sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

# perform sentiment analysis using Vader pkg and store as series variable
sentiment = df['reviewtext'].apply(lambda x: analyzer.polarity_scores(x))

# concatenate the sentiment variable with your existing df
df = pd.concat([df,sentiment.apply(pd.Series)],1)



###### SECOND PART - EXTRACTING KEY WORDS


import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

def pre_process(text):
    
    # lowercase
    text=text.lower()
    
    #remove tags
    text=re.sub("<!--?.*?-->","",text)
    
    # remove special characters and digits
    text=re.sub("(\d|\W)+"," ",text)
    
    return text
 
# apply pre-processing fn to the text
df['cleantext'] = df['reviewtext'].apply(lambda x:pre_process(x))
 
# get the text column change to list
docs=df['cleantext'].tolist()
 
#create a vocabulary of words, rule is to ignore words that appear in 80% of documents, 
#eliminate stop words, set max features to 10k
cv=CountVectorizer(max_df=0.80,max_features=10000)
word_count_vector=cv.fit_transform(docs)
list(cv.vocabulary_.keys())[:10]


tfidf_transformer=TfidfTransformer(smooth_idf=True,use_idf=True)
tfidf_transformer.fit(word_count_vector)
 

# you only needs to do this once, this is a mapping of index to 
feature_names=cv.get_feature_names()
 
 
def sort_coo(coo_matrix):
    tuples = zip(coo_matrix.col, coo_matrix.data)
    return sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)

def extract_topn_from_vector(feature_names, sorted_items, topn=10):
    """get the feature names and tf-idf score of top n items"""
    
    #use only topn items from vector
    sorted_items = sorted_items[:topn]
 
    score_vals = []
    feature_vals = []
    
    # word index and corresponding tf-idf score
    for idx, score in sorted_items:
        
        #keep track of feature name and its corresponding score
        score_vals.append(round(score, 3))
        feature_vals.append(feature_names[idx])
 
    #create a tuples of feature,score
    #results = zip(feature_vals,score_vals)
    results= {}
    for idx in range(len(feature_vals)):
        results[feature_vals[idx]]=score_vals[idx]
    
    return results


 
def get_keywords(idx):

    #generate tf-idf for the given document
    tf_idf_vector=tfidf_transformer.transform(cv.transform([docs[idx]]))

    #sort the tf-idf vectors by descending order of scores
    sorted_items=sort_coo(tf_idf_vector.tocoo())

    #extract only the top n; n here is 10
    keywords=extract_topn_from_vector(feature_names,sorted_items,10)
    
    return keywords


# now create empty list to hold key words
kw = []

# run keywords fn and append to list
for idx in range(len(df)):
    kw.append(get_keywords(idx))

# add the list to the df as a new column 
df['keywordsnew'] = kw


# save your csv file and explore sentiment analysis & key word results
# NOTE - it assigned sentiment on the entire review. this did not focus on key words like Watson did
# next steps are ot work with tokenized text, perform cleaning, maybe split into bigrams/trigrams and assign 
# sentiment to that either using sentiment library, our own defined lists, or building a classifier
export_csv = df.to_csv(r'reviews_sentiment.csv', index = None, header=True) #Don't forget to add '.csv' at the end of the path

