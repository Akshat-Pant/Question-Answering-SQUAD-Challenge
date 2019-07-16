#!/usr/bin/env python
# coding: utf-8

# In[365]:


import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from keras.models import Sequential
from keras.layers import Dense

import re

import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import string

import category_encoders as ce

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.model_selection import KFold, cross_val_predict, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer

import gensim
from gensim import corpora


# In[475]:


"""
define a function for auto ml
define a function to prepare data
define a function to clean text (also define stopwords, lemmatizer, punctuations)
"""

def auto_ML(model, features, Xtrain, Ytrain, Xtest, Ytest, folds= 10):
    model.fit(Xtrain[features], Ytrain)
    prediction= model.predict(Xtest[features])
    
    # Cross validation
    scores= cross_val_score(model, Xtrain[features], Ytrain, cv= folds)
    val_score= sum(scores/folds)
    
    print('Test prediction:\n{}'.format(prediction))
    print('-------------------------------------')
    print('Accuracy score: {}'.format(accuracy_score(Ytest, prediction)))
    print('-------------------------------------')
    print('confusion matrix:\n {}'.format(confusion_matrix(Ytest, prediction)))
    print('-------------------------------------')
    print('Cross Validation score: {}'.format(val_score))
    
    correct_df= pd.DataFrame(columns= Xtrain.columns)
    incorrect_df= pd.DataFrame(columns= Xtrain.columns)
    
    for i, pred in enumerate(prediction):
        if pred== Ytest[i]:
            correct_df= correct_df.append(Xtest.loc[i, Xtest.columns])
        else:
            incorrect_df= incorrect_df.append(Xtest.loc[i, Xtest.columns])
            
    return correct_df, incorrect_df
    
    
def prepare_data(data, split_ratio):
    split= int(data.shape[0]*split_ratio)
    
    X_train= data.loc[:split, :]
    X_test= data.loc[split: , :]
    
    print('train shape: {}'.format(X_train.shape))
    print('test shape: {}'.format(X_test.shape))
    
    return X_train, X_test



stop= stopwords.words('english')
exclude = string.punctuation 
lemma = WordNetLemmatizer()

def clean(doc):
    stop_free= ' '.join(word for word in doc.lower().split() if word not in stop)
    punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
    normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split())
    return normalized
    


# In[476]:


train= pd.read_csv('..../train.csv')
test= pd.read_csv('..../test.csv')


# In[477]:


train.head()


# In[405]:


print('training data:')
train.shape


# In[406]:


print('testing data: ')
test.shape


# ### add new features

# In[478]:


"""
Add new feature: Question length

Add new features: Extract max IR score and correscponding page from IR_Wiki Scores
Also calculate difference between highest and second highest IR score

Add target feature paren_match (true if wiki page and answer match)

We will add the following features after splitting into training and testing.
1. Apply Weight of Evidence(WOE) encoding to category variable
2. Topic modeling
"""

#Question length (training data)
lens= []
for i in range(0, train.shape[0]):
    lens.append(len(train.loc[i]['Question Text']))
lens
train['Quest len']= lens
print('Question length created for training data')

#Question length (testing data)
lens= []
for i in range(0, test.shape[0]):
    lens.append(len(test.loc[i]['Question Text']))
lens
test['Quest len']= lens
print('Question length created for testing data')



######################################################################

# Max IR score and corresponding page (training data)
wiki_page= []
wiki_page2= []
page_score= []
diff= []
for i in range(0, train.shape[0]):
    ans_score= {}
    for ii in train.loc[i]['IR_Wiki Scores'].split(', '):
        ans_score[ii.split(':')[0]]= float(ii.split(':')[1])
        
    
    page= sorted(ans_score, key= ans_score.get, reverse= True)[0]
    page2= sorted(ans_score, key= ans_score.get, reverse= True)[1]
    wiki_page.append(page)
    page_score.append(ans_score[page])
    wiki_page2.append(page2)
    diff.append(ans_score[page]- ans_score[page2])

train['Wiki_page']= wiki_page
train['Wiki_page_embed']= wiki_page
train['Wiki_page2']= wiki_page2
train['Page score']= page_score
train['Score difference']= diff
print('Max IR score and corresponding page created for training data')



# Max IR score and corresponding page (testing data)
wiki_page= []
wiki_page2= []
page_score= []
diff= []
for i in range(0, test.shape[0]):
    ans_score= {}
    for ii in test.loc[i]['IR_Wiki Scores'].split(', '):
        ans_score[ii.split(':')[0]]= float(ii.split(':')[1])
        
    
    page= sorted(ans_score, key= ans_score.get, reverse= True)[0]
    page2= sorted(ans_score, key= ans_score.get, reverse= True)[1]
    wiki_page.append(page)
    wiki_page2.append(page2)
    page_score.append(ans_score[page])
    diff.append(ans_score[page]- ans_score[page2])

test['Wiki_page']= wiki_page
test['Wiki_page_embed']= wiki_page
test['Wiki_page2']= wiki_page2
test['Page score']= page_score
test['Score difference']= diff
print('Max IR score and corresponding page created for testing data')



################################################################

# Target feature paren_match (training data). It is 1 if answer and wiki page match. 0 otherwise
train['paren_match']= 0

for i, row in train.iterrows():
    if row['Answer'] == row['Wiki_page']:
        train.loc[i, 'paren_match']= 1
print('paren_match created for training data')



#################################################################

# Create a new column to embed categories (training data)
train['category_embed']= train['category']

"""
encoding= ce.WOEEncoder(cols= ['category_embed'], impute_missing= True)
encoding.fit(train[['category_embed']], train['paren_match'])
train['category_embed']=encoding.transform(train[['category_embed']])
print('Category embeddings created for training data')



# Create a new column to embed categories (testing data)
test['category_embed']= test['category']
test['category_embed']= encoding.transform(test[['category_embed']])
print('Category embeddings created for testing data')
"""



#################################################################


# Create new topic columns for topic modeling (training data)
train['Topic 0']= 0
train['Topic 1']= 0
train['Topic 2']= 0
train['Topic 3']= 0
print('Topics created for training data')


# Create new topic columns for topic modeling (testing data)
test['Topic 0']= 0
test['Topic 1']= 0
test['Topic 2']= 0
test['Topic 3']= 0
print('Topics created for testing data')


##################################################################
##################################################################

"""
Add new feature: topics (topic modeling)
We have 4 categories in the dataset. SO lets use 4 topics
"""

"""
# Preprocessing the text first
clean_questions= [clean(row['Question Text']).split() for _, row in train.iterrows()]
print('Text preprocessing done')

# Create a word to index mapping
dictionary= corpora.Dictionary(clean_questions)
print('Word index dictionary created')

# Convert the list of documents to a BOW matrix using the dictionary prepared above
doc_term_matrix = [dictionary.doc2bow(doc) for doc in clean_questions]
print('BOW matrix created')

# Train the LDA model on the document word matrix
ldamodel = gensim.models.ldamodel.LdaModel(doc_term_matrix, num_topics=4, id2word = dictionary, passes=50)
print('LDA model trained')
"""


"""
Add topic distributions as features to train and test data
"""

"""
# Topic distributions (training data)
for i, row in train.iterrows():
    for pair in ldamodel[doc_term_matrix[i]]:
        col= 'Topic '+ str(pair[0])
        train.loc[i, col]= pair[1]
        
print('Topic distributions added to training data')    
    

# Topic distributions (testing data)

clean_test= [clean(row['Question Text']).split() for _, row in test.iterrows()]
test_bow= [dictionary.doc2bow(doc) for doc in clean_test]

for i, row in test.iterrows():
    for pair in ldamodel.get_document_topics(test_bow[i]):
        col= 'Topic '+ str(pair[0])
        test.loc[i, col]= pair[1]

print('Topic distributions added to testing data')
"""


# ## Split into training and testing

# In[479]:


X_train, X_test, y_train, y_test = train_test_split(train, train['paren_match'], test_size= 0.2, random_state= 4)
print('X train shape: {}'.format(X_train.shape))
print('y train shape: {}'.format(y_train.shape))
print('X test shape: {}'.format(X_test.shape))
print('y test shape: {}'.format(y_test.shape))

X_train= X_train.reset_index(drop= True)
y_train= y_train.reset_index(drop= True)
X_test= X_test.reset_index(drop= True)
y_test= y_test.reset_index(drop= True)

print('reindexed X train and y train for WOE embeddings')
print('reindexed X test and y test for WOE embeddings')


# In[480]:


encoding= ce.WOEEncoder(cols= ['category_embed'], impute_missing= True)
encoding.fit(X_train[['category_embed']], X_train['paren_match'])
X_train['category_embed']=encoding.transform(X_train[['category_embed']])
print('Category embeddings created for training data')

# Create a new column to embed categories (testing data)
X_test['category_embed']= encoding.transform(X_test[['category_embed']])
print('Category embeddings created for testing data')


# In[481]:


"""
Add new feature: topics (topic modeling)
We have 4 categories in the dataset. SO lets use 4 topics
"""
# Preprocessing the text first
clean_questions= [clean(row['Question Text']).split() for _, row in X_train.iterrows()]
print('Text preprocessing done')

# Create a word to index mapping
dictionary= corpora.Dictionary(clean_questions)
print('Word index dictionary created')

# Convert the list of documents to a BOW matrix using the dictionary prepared above
doc_term_matrix = [dictionary.doc2bow(doc) for doc in clean_questions]
print('BOW matrix created')

# Train the LDA model on the document word matrix
ldamodel = gensim.models.ldamodel.LdaModel(doc_term_matrix, num_topics=4, id2word = dictionary, passes=50)
print('LDA model trained')


# In[482]:


"""
Add topic distributions as features to train and test data
"""
# Topic distributions (training data)
for i, row in X_train.iterrows():
    for pair in ldamodel[doc_term_matrix[i]]:
        col= 'Topic '+ str(pair[0])
        X_train.loc[i, col]= pair[1]
        
print('Topic distributions added to training data')    
    

# Topic distributions (testing data)

clean_test= [clean(row['Question Text']).split() for _, row in X_test.iterrows()]
new_bow= [dictionary.doc2bow(doc) for doc in clean_test]

for i, row in X_test.iterrows():
    for pair in ldamodel.get_document_topics(new_bow[i]):
        col= 'Topic '+ str(pair[0])
        X_test.loc[i, col]= pair[1]

print('Topic distributions added to testing data')


# ### features

# In[483]:


features= ['Quest len', 'Page score',  'category_embed', 'Score difference', 'Topic 0', 'Topic 1', 'Topic 2', 'Topic 3']
target= ['paren_match']


# ### regularize data

# In[484]:


"""
feat_reg= ['Quest len', 'Page score',  'category_embed', 'Score difference']

scaler= StandardScaler()

scaler.fit(X_train[feat_reg].values)

X_train[feat_reg]= scaler.transform(X_train[feat_reg].values)
X_test[feat_reg]= scaler.transform(X_test[feat_reg].values)

print('regularized X train and X test')
"""

y_train= np.reshape(y_train.values, (y_train.shape[0], ))


# ### SVM

# In[358]:


svm_clf= SVC(kernel= 'rbf')
svm_clf.fit(train[features], train['paren_match'])
prediction= model.predict(test[features])


# In[359]:


prediction


# In[362]:


test['buzz']= 'false'

for i, row in test.iterrows():
    if prediction[i]>= 0.5:
        test.loc[i, 'buzz']= 'true'


# In[363]:


test.head()


# In[364]:


test.to_csv('/Users/akshatpant/Desktop/UMD/Sem 3/Comp Ling/Project/predict.csv')


# In[485]:


svm_clf= SVC(kernel= 'rbf')

corr, incorr= auto_ML(svm_clf, features, X_train, y_train, X_test, y_test, folds= 10)


# In[472]:


corr.describe()


# In[490]:


incorr.describe()


# In[471]:


incorr[incorr['category']== 'social']['Question Text']


# In[463]:


X_test['category'].value_counts()


# In[ ]:





# ### Feedforward net

# In[308]:


"""
Prepare input
"""
X_train= np.array(X_train[features])
y_train= np.array(y_train)

X_test= np.array(X_test[features])
y_test= np.array(y_test)


# In[309]:


"""
Create model
"""
model= Sequential()
model.add(Dense(15, activation='relu'))
model.add(Dense(10, activation= 'relu'))
model.add(Dense(5, activation= 'relu'))
model.add(Dense(1, activation= 'sigmoid'))

"""
compile model
"""
model.compile(loss= 'binary_crossentropy', optimizer= 'adam')

"""
Fit the model
"""
model.fit(X_train, y_train, epochs=20, batch_size=10, verbose=1, validation_split= 0.2)

"""
Calculate predictions
"""
predictions= model.predict(X_test)

"""
calculate accuracy
print confusion matrix

         Pred labels
         0     1
Target 0
labels 1

"""
#threshold to decide class
threshold= 0.5

count= 0
false_positive= 0
false_negative= 0
true_positive= 0
true_negative= 0
for i in range(0, predictions.shape[0]):
    if (predictions[i]> threshold) & (y_test[i]== 1):
        count+= 1
        true_positive+= 1
    if (predictions[i]<= threshold) & (y_test[i]== 0):
        count+= 1
        true_negative+= 1
    if (predictions[i]> threshold) & (y_test[i]== 0):
        false_positive+= 1
    if (predictions[i]<= threshold) & (y_test[i]== 1):
        false_negative+= 1
        
print('\tPred Labels\n\t0\t1\nTarget 0 {}\t{}\nlabels 1 {}\t{}'.format(true_negative, 
                                false_positive, false_negative, true_positive))

print('\n')

print(count/predictions.shape[0])


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[42]:


X_train, X_test, y_train, y_test = train_test_split(train_df, train_df[target], test_size=0.20)


# In[13]:


def prepare_seq(seq, to_ix):
    idxs= []
    for w in seq:
        if w in to_ix:
            idxs.append(to_ix[w])
        else:
            idxs.append(to_ix['<UNK>'])
    return torch.tensor(idxs, dtype=torch.long)


# In[43]:


"""
prepare word_to_index dictionary
set <pad> and <unk> value to 0
normalize text before adding to vocab
remove stopwords
"""

regex= re.compile(r"\b(\w*['\w]*[\w]*)[^\w]*")


word_to_ix= {'<UNK>': 0}

for i, row in train_df.iterrows():
    sent= train_df.loc[i, 'Question Text'].lower()
    for word in regex.findall(sent):
        if word not in word_to_ix:
            word_to_ix[word]= len(word_to_ix)- 1
    train_df.loc[i, 'Question Text']= ' '.join(regex.findall(sent))
    
word_to_ix


# In[15]:


train_df


# In[36]:


"""
Create LSTM network for the text.
Combine other features into the output of LSTM.

"""

class LSTM_special(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, batch_size, vocab_size, add_dim, out_dim= 1, num_layers= 1):
        #embedding_dim is the same as input_dim as embeddings are used as input to the LSTM
        
        super(LSTM_special, self).__init__()
        self.input_dim= embedding_dim
        self.hidden_dim= hidden_dim
        self.batch_size= batch_size
        self.add_dim= add_dim
        self.out_dim= out_dim
        self.num_layers= num_layers
        
        # Embedding layer
        self.word_embeddings= nn.Embedding(vocab_size, self.input_dim)
        
        # Define the LSTM layer
        self.lstm= nn.LSTM(self.input_dim, self.hidden_dim, self.num_layers)
        
        # Define the linear layer that maps from (hidden_dim + add_dim) to out_dim
        self.linear= nn.Linear(self.hidden_dim+self.add_dim, self.out_dim)
        
        # Define the non-linearity that converts to probability
        self.softmax= nn.Softmax()
        
    def init_hidden(self):
        """
        Initialize the hidden state (h0, c0)
        
        Before we've done anything, we dont have any hidden state.
        Refer to the Pytorch documentation to see exactly
        why they have this dimensionality.
        The axes semantics are (num_layers, minibatch_size, hidden_dim)
        """
        return (torch.zeros(self.num_layers, self.batch_size, self.hidden_dim),
               torch.zeros(self.num_layers, self.batch_size, self.hidden_dim))
        
    def forward(self, sentence, add_features):
        """
        forward pass through LSTM layer
        first pass through the embedding layer
        hidden to output space mapping by Linear layer
        lstm_out shape: [seq_len/input_len, batch_size, hidden_dim]
        self.hidden shape= (a, b) where a & b both have shape: [num_layers, batch_size, hidden_dim]
        """
        
        # Embedding layer
        embeds= self.word_embeddings(sentence)
        
        
        lstm_out, self.hidden= self.lstm(embeds.view(len(sentence), self.batch_size, -1), self.hidden)
        
        """
        Take the output from the last layer of the LSTM and 
        concatenate the additional features to them.
        Map them to output space.
        Apply non linearity like softmax
        """
        # get the output from the last timestep
        lstm_out= lstm_out[-1].view(self.batch_size, -1)
        
        # concatenate additional features to lstm output
        new_features= torch.cat((lstm_out, add_features.view(self.batch_size, -1)), 1)
        
        # map to output space
        y_pred= self.linear(new_features)
        
        # apply non linearity
        output= self.softmax(y_pred)
        
        
        return output
        
        
        
        


# In[37]:


"""
Initialize the model
Train the LSTM
"""

#################
# initialize the model
#################

model= LSTM_special(embedding_dim= 50, hidden_dim= 30, batch_size= 1, vocab_size= len(word_to_ix), add_dim= 5,) 

#################
# define the loss function
#################

loss_fn= torch.nn.MSELoss()
optimiser= torch.optim.Adam(model.parameters(), lr= 0.01)
num_epochs= 4

###################
#Train the model
###################

hist= np.zeros(num_epochs)

regex= re.compile(r"\b(\w*['\w]*[\w]*)[^\w]*")


for i in range(0, num_epochs):
    
    for ii, row in train_df.iterrows():
        
        #clear the stored gradients
        model.zero_grad()
        
        #clear the history of LSTM. Reset the hidden state
        model.hidden= model.init_hidden()
        
        # get the inputs ready for lstm
        sent= train_df.loc[ii, 'Question Text'].lower()
        sentence= prepare_seq(regex.findall(sent), word_to_ix)
        
        #additional features (time invariant)
        additional_features= [train_df.loc[ii, f] for f in features]
        additional_features= torch.tensor(additional_features)
        #additional_features= additional_features.type(torch.FloatTensor)
        
        #print((additional_features.values))
        #print((additional_features.values.view(1, -1)))
        
        # forward pass
        pred= model(sentence, additional_features)
        
        ans= torch.tensor(row['paren_match'])
        ans= ans.type(torch.FloatTensor).view(1,1)
        
        loss= loss_fn(pred, ans)
        
        hist[i]= loss.item()
        
        loss.backward()
    
        optimiser.step()
        print('Epoch: {}, row: {}'.format(i, ii))
        
    print('Epoch: ', i, 'MSE: ', loss.item())
        
        
        
        
        
        


# In[ ]:





# In[38]:


for ii, row in train_df.iterrows():
        
    #clear the stored gradients
    model.zero_grad()
        
    #clear the history of LSTM. Reset the hidden state
    model.hidden= model.init_hidden()
        
    # get the inputs ready for lstm
    sent= train_df.loc[ii, 'Question Text'].lower()
    sentence= prepare_seq(regex.findall(sent), word_to_ix)
        
    #additional features (time invariant)
    additional_features= [train_df.loc[ii, f] for f in features]
    additional_features= torch.tensor(additional_features)
    #additional_features= additional_features.type(torch.FloatTensor)
        
    #print((additional_features.values))
    #print((additional_features.values.view(1, -1)))
        
    # forward pass
    pred= model(sentence, additional_features)
    print(pred)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




