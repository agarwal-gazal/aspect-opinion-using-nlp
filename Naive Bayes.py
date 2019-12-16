#########################
#  Author: Gazal Agarwal
#########################

# coding: utf-8

# In[5]:


#BAYESIAN CLASSISFIER

import pandas as pd 
import numpy as np 
from collections import defaultdict
import re
from sklearn.model_selection import train_test_split

def preprocess_string(str_arg):
    
    cleaned_str=re.sub('[^a-z\s]+',' ',str_arg,flags=re.IGNORECASE) #every char except alphabets is replaced
    cleaned_str=re.sub('(\s+)',' ',cleaned_str) #multiple spaces are replaced by single space
    cleaned_str=cleaned_str.lower() #converting the cleaned string to lower case
    
    return cleaned_str # eturning the preprocessed string in tokenized form

class NaiveBayes:
    
    def __init__(self,unique_classes):
        
        self.classes=unique_classes # Constructor is siMply passed with unique number of classes of the training set
        

    def addToBow(self,example,dict_index):
        
        if isinstance(example,np.ndarray): example=example[0]
     
        for token_word in example.split(): #for every word in preprocessed example
          
            self.bow_dicts[dict_index][token_word]+=1 #increment in its count
            
    def train(self,dataset,labels):
    
        self.examples=dataset
        self.labels=labels
        self.bow_dicts=np.array([defaultdict(lambda:0) for index in range(self.classes.shape[0])])
        
        #only convert to numpy arrays if initially not passed as numpy arrays - else its a useless recomputation
        
        if not isinstance(self.examples,np.ndarray): self.examples=np.array(self.examples)
        if not isinstance(self.labels,np.ndarray): self.labels=np.array(self.labels)
            
        #constructing BoW for each category
        for cat_index,cat in enumerate(self.classes):
          
            all_cat_examples=self.examples[self.labels==cat] #filter all examples of category == cat
            
            #get examples preprocessed
            
            cleaned_examples=[preprocess_string(cat_example) for cat_example in all_cat_examples]
            
            cleaned_examples=pd.DataFrame(data=cleaned_examples)
            
            #now costruct BoW of this particular category
            np.apply_along_axis(self.addToBow,1,cleaned_examples,cat_index)
            
     
        prob_classes=np.empty(self.classes.shape[0])
        all_words=[]
        cat_word_counts=np.empty(self.classes.shape[0])
        for cat_index,cat in enumerate(self.classes):
           
            #Calculating prior probability p(c) for each class
            prob_classes[cat_index]=np.sum(self.labels==cat)/float(self.labels.shape[0]) 
            
            #Calculating total counts of all the words of each class 
            count=list(self.bow_dicts[cat_index].values())
            cat_word_counts[cat_index]=np.sum(np.array(list(self.bow_dicts[cat_index].values())))+1 # |v| is remaining to be added
            
            #get all words of this category                                
            all_words+=self.bow_dicts[cat_index].keys()
                                                     
        
        #combine all words of every category & make them unique to get vocabulary -V- of entire training set
        
        self.vocab=np.unique(np.array(all_words))
        self.vocab_length=self.vocab.shape[0]
                                  
        #computing denominator value                                      
        denoms=np.array([cat_word_counts[cat_index]+self.vocab_length+1 for cat_index,cat in enumerate(self.classes)])                                                                          
    
        self.cats_info=[(self.bow_dicts[cat_index],prob_classes[cat_index],denoms[cat_index]) for cat_index,cat in enumerate(self.classes)]                               
        self.cats_info=np.array(self.cats_info)                                 
                                              
                                              
    def getExampleProb(self,test_example):                                                 
                                              
        likelihood_prob=np.zeros(self.classes.shape[0]) #to store probability w.r.t each class
        
        #finding probability w.r.t each class of the given test example
        for cat_index,cat in enumerate(self.classes): 
                             
            for test_token in test_example.split(): #split the test example and get p of each test word
                
                #get total count of this test token from it's respective training dict to get numerator value                           
                test_token_counts=self.cats_info[cat_index][0].get(test_token,0)+1
                
                #now get likelihood of this test_token word                              
                test_token_prob=test_token_counts/float(self.cats_info[cat_index][2])                              
                
                #remember why taking log? To prevent underflow!
                likelihood_prob[cat_index]+=np.log(test_token_prob)
                                              
        # we have likelihood estimate of the given example against every class but we need posterior probility
        post_prob=np.empty(self.classes.shape[0])
        for cat_index,cat in enumerate(self.classes):
            post_prob[cat_index]=likelihood_prob[cat_index]+np.log(self.cats_info[cat_index][1])                                  
      
        return post_prob
    
   
    def test(self,test_set):
        
        predictions=[] #to store prediction of each test example
        for example in test_set: 
                                              
            #preprocess the test example the same way we did for training set exampels                                  
            cleaned_example=preprocess_string(example) 
             
            #simply get the posterior probability of every example                                  
            post_prob=self.getExampleProb(cleaned_example) #get prob of this example for both classes
            
            #simply pick the max value and map against self.classes!
            predictions.append(self.classes[np.argmax(post_prob)])
                
        return np.array(predictions)
    

def main():
    user_columns=['business_id','r1','date','r2','user_id','Rating','Review','r3','user_review_id','Sentiment','Output_Score']
    file_path=r"Adult_Education_training.csv"
    test_file_path=r"Adult_Education_validation.csv"
    training_set = pd.read_csv(file_path)
    test_set = pd.read_csv(test_file_path,names=user_columns)
    #print(test_set)
    for index,row in training_set.iterrows():
        rating = row['stars']
        if rating >= 3:
            training_set.iloc[index, training_set.columns.get_loc('Sentiment')] = 1
        elif rating < 3:
            training_set.iloc[index, training_set.columns.get_loc('Sentiment')] = 0
    for index,row in test_set.iterrows():
        rating = row['Rating']
        if rating >= 3:
            test_set.iloc[index, test_set.columns.get_loc('Sentiment')] = 1
        elif rating < 3:
            test_set.iloc[index, test_set.columns.get_loc('Sentiment')] = 0
    train_labels=training_set['Sentiment'].values
    test_labels=test_set['Sentiment'].values
    train_data=training_set['text'].values
    test_data=test_set['Review'].values
    #print ("Unique Classes: ",np.unique(y_train))
    #print ("Total Number of Training Examples: ",x_train.shape)
    #train_data,test_data,train_labels,test_labels=train_test_split(x_train,y_train,shuffle=True,test_size=0.25,random_state=42,stratify=y_train)
    #print(train_data,' ',test_data,' ',train_labels,' ',test_labels)
    #test_labels, train_labels=[0,1]
    #train_data=training_set['Review'].values
    classes=np.unique(train_labels)
    nb=NaiveBayes(classes)
    print ("------------------Training In Progress------------------------")
    print ("Training Examples: ",train_data.shape)
    nb.train(train_data,train_labels)
    print ('------------------------Training Completed!')

    # Testing phase 

    pclasses=nb.test(test_data)
    test_acc=np.sum(pclasses==test_labels)/float(test_labels.shape[0])
    print ("Test Set Examples: ",test_labels.shape[0])
    print ("Test Set Accuracy: ",test_acc)
    Xtest=test_set.Review.values

    #WRITING OUTPUT TO A CSV FILE
    result_df=pd.DataFrame(list(zip(test_set.Review.values, test_set.Rating.values, pclasses)), columns = ['Review','Rating','Sentiment'])
    print('\n####################RESULTANT DATAFRAME####################\n')
    print(result_df)
    op_file_path=r"Adult_Education_validation_Test_OP_NB.csv"
    result_df.to_csv(op_file_path,index=False)
    
main()

