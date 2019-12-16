#########################
#  Author: Gazal Agarwal
#########################

# coding: utf-8

# In[10]:


import pandas as pd
import numpy as np
import pprint
import spacy 
nlp = spacy.load('en_core_web_sm')
user_columns=['business_id','r1','date','r2','user_id','Rating','Review','r3','user_review_id','Sentiment','Output_Score']
file_path=r"Adult_Education_validation.csv"
df = pd.read_csv(file_path,names=user_columns)
#print(df)
from textblob import TextBlob
import spacy
nlp = spacy.load('en_core_web_sm')

def get_review_score(noun_scores_dict, adj_with_noun):
    
    score = 0
    
    for adjt,nounscores in adj_with_noun.items():
        
        for noun,scores in noun_scores_dict.items():
            if str(nounscores[0]) == str(noun):
                #print(noun,' ',nounscores[1],' ',scores,' ','\n')
                score = score + (nounscores[1] * scores)
                #print(score)

    score = score / (len(adj_with_noun)+1)
    
    return score

def pos_words (sentence, token, ptag):
    #sentences = [sent for sent in sentence.sents if token in sent.string]     
    pwrds = []
    
    ajt = str(find_nearest_noun(sentence, token, ptag))
    pwrds.extend([ajt, TextBlob(str(token)).sentiment.polarity] )
    return pwrds


def find_nearest_noun(doc, word, ptag):
    for i,token in enumerate(doc):
            if str(token) == str(word):
                #print('i',' ',i,'token',' ',token,'\n')
                j = k = leftHop = rightHop = -1
                for j in range(i+1,len(doc)):
                    if doc[j].pos_ == ptag[0] or doc[j].pos_ == ptag[1]: 
                        rightHop = j-i
                        rightadj = doc[j].text
                        #print('rightHop',' ',doc[j].text,' ',rightHop)
                        break
                        
                for k in range(i-1,-1,-1):
                    if doc[k].pos_ == ptag[0] or doc[j].pos_ == ptag[1]:
                        leftHop = i-k
                        leftadj = doc[k].text
                        #print('leftHop',' ',doc[k].text,' ',leftHop)
                        break

                
                #Compare which noun is closer to adjective(left or right) and assign the adj to corresponding noun
                if(leftHop > 0 and rightHop > 0):					#If nouns exist on both sides of adjective
                    if (leftHop - rightHop) >= 0:						#If left noun is farther
                        return rightadj
                    else:									#If right noun is farther
                        return leftadj
                elif rightHop == -1:								#If noun is not found on RHS of adjective
                    return leftadj
                elif leftHop == -1:								#If noun is not found on LHS of adjective
                    return rightadj

def extract(feature_dictionary):
    countp=0
    countn=0
    for index,row in df.iterrows():
        review = row['Review']
        doc = nlp(review.lower())
        print(doc)
        noun_scores={}
        adj_list=[]
        for i,token in enumerate(doc):
            if token.pos_ not in ('NOUN','PROPN'):
                continue
            
            for j in range(i-1,len(doc)):
                if doc[j].pos_ == 'ADJ':
                    if token.lemma_ not in feature_dictionary:
                        feature_dictionary[token.lemma_] = 1
                    else:
                        feature_dictionary[token.lemma_] = feature_dictionary[token.lemma_] + 1
                        
                    if str(token) in noun_scores:
                        noun_scores[str(token)] = noun_scores[str(token)] + 1
                    else:
                        noun_scores[str(token)] = 1
                    if doc[j] not in adj_list:
                        adj_list.append(doc[j])
                    break
            
            
            
        print('\n#################NOUN SCORES################\n')
        pprint.pprint(noun_scores)
        print('\n#################ADJECTIVE LIST################\n')
        pprint.pprint(adj_list)
        
        adj_with_noun={}
        
        for adjectives in adj_list:
            adj_with_noun[adjectives] = pos_words(doc, adjectives, ['NOUN','PROPN'])
        
        print('\n#################ADJECTIVES WITH THE CORESPONDING NOUN AND POLARITY################\n')
        pprint.pprint(adj_with_noun)
            
        score = get_review_score(noun_scores, adj_with_noun)
            
        print('\n#################CALCULATED REVIEW SCORE################\n')
        print(score)
            
        rating = row['Rating']
        
        
        print('\n#################USER RATING################\n')
        print(rating)
        
        print('\n#################COMPUTED SENTIMENT################\n')
            
        if score >= 0:
            df.iloc[index, df.columns.get_loc('Sentiment')] = 'Pos'
            print('----------------->  POSITIVE REVIEW')
        elif score < 0:
            df.iloc[index, df.columns.get_loc('Sentiment')] = 'Neg'
            print('----------------->  NEGATTIVE REVIEW')
        '''    
        elif score == 0:
            df.iloc[index, df.columns.get_loc('Sentiment')] = 'Net'
            print('----------------->  NEUTRAL REVIEW')'''        
        
    
def check():
    cp1=0
    cp2=0
    cn1=0
    cn2=0
    for index,row in df.iterrows():
        rating = row['Rating']
        sentiment = row['Sentiment']
        if rating >= 3:
            if sentiment == 'Pos':
                cp1 = cp1 + 1
            else:
                cn1 = cn1 + 1    
    
        elif rating < 3:
            if sentiment == 'Neg':
                cp1 = cp1 + 1
            else:
                cn1 = cn1 + 1
                
        if sentiment == 'Pos':
            df.iloc[index, df.columns.get_loc('Output_Score')] = 1
        elif sentiment == 'Neg':
            df.iloc[index, df.columns.get_loc('Output_Score')] = 0
            
        '''       
        elif rating == 3:
            if sentiment == 'Net':
                cp1 = cp1 + 1
            else:
                cn1 = cn1 + 1'''
           
    fp = (cp1/(cp1+cn1))

    print('\n#################ACCURACY################\n')
    print('Accuracy percentage', fp)
   

if __name__ == "__main__":
    feature_dictionary = {}
    extract(feature_dictionary)
    check()
    print('\n#################FINAL DATAFRAME################\n')
    print(df)
    #print(feature_dictionary)
    
    #WRITING OUTPUT TO A CSV FILE
    file_path=r"Adult_Education_validation_Test_Results_NLP.csv"
    user_columns=['business_id','r1','date','r2','user_id','Rating','Review','r3','user_review_id','Sentiment','Output_Score']
    df.to_csv(file_path, encoding='utf-8', index=False, header=user_columns)
    
    #PRINTING THE FEATURES OF A PARTICULAR DATASET
    sorted(feature_dictionary.items(), key =lambda kv:(kv[1], kv[0]), reverse=True)[:30]


