#!/usr/bin/env python
# coding: utf-8

# <h1>Dawood Sarfraz </h1>
# <h1> Duplicate Questions using KNN, Decision Tree, MLP,  RandomForestClassifier and XGBoostClassifier with Advance Features </h1>

# # Dataset Description
# The ground truth is the set of labels that have been supplied by human experts. The ground truth labels are inherently subjective, as the true meaning of sentences can never be known with certainty. Human labeling is also a 'noisy' process, and reasonable people will disagree. As a result, the ground truth labels on this dataset should be taken to be 'informed' but not 100% accurate, and may include incorrect labeling. We believe the labels, on the whole, to represent a reasonable consensus, but this may often not be true on a case by case basis for individual items in the dataset.
# 
# <b>Please note:</b> 
# All of the questions in the training set are genuine examples from Quora.
# 
# # Data fields
# <b>* id - </b> the id of a training set question pair </br>
# <b>* qid1, qid2 - </b>- unique ids of each question (only available in train.csv) </br>
# <b>* question1, question2 - </b>- the full text of each question </br>
# <b>* is_duplicate - </b> the target variable, set to 1 if question1 and question2 have essentially the same meaning, and 0 otherwise.</br>

# <h1> Without Feature Engineering </h1>

# In[10]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import re
from bs4 import BeautifulSoup

import warnings
warnings.filterwarnings('ignore')


# In[11]:


df = pd.read_csv("train.csv")


# In[12]:


df.shape


# In[13]:


df.head(5)


# In[14]:


df.tail(5)


# In[15]:


df.sample(5)


# In[16]:


df.info()


# In[17]:


df.isnull().sum()


# In[18]:


df = df.dropna()


# In[19]:


df.shape


# In[20]:


df.isnull().sum()


# In[21]:


df.duplicated().sum()


# In[22]:


print(df["is_duplicate"].value_counts())
print((df["is_duplicate"].value_counts()/df["is_duplicate"].count())*100)


# In[23]:


df["is_duplicate"].value_counts().plot(kind="bar")


# In[24]:


df["is_duplicate"].value_counts().plot(kind="pie")


# In[25]:


qid = pd.Series(df["qid1"].tolist() + df["qid2"].tolist())
print("# of Unique Questions",np.unique(qid).shape[0])


# In[26]:


x = qid.value_counts()>1
print("# of Questions Qepeated",x[x].shape[0])


# In[27]:


plt.hist(qid.value_counts().values,bins=90,color="brown")
plt.yscale("log")
plt.show()


# In[28]:


new_df = df


# In[29]:


new_df.shape


# In[30]:


new_df.isnull().sum()


# In[31]:


new_df = new_df.dropna()


# In[32]:


new_df = df.sample(30000) 


# In[33]:


new_df.shape


# In[34]:


new_df.isnull().sum()


# In[35]:


new_df.duplicated().sum()


# In[36]:


new_df.shape


# In[37]:


ques_df = new_df[['question1','question2']]
ques_df.head()


# In[38]:


from sklearn.feature_extraction.text import CountVectorizer
# merge texts of questions asked
questions = list(ques_df['question1']) + list(ques_df['question2'])

# if You have Good laptop increase No. of max_features
cv = CountVectorizer(max_features=3000)#creating 3000 here for Question1 7 3000 for Question2 in end would double
q1_array, q2_array = np.vsplit(cv.fit_transform(questions).toarray(),2)


# In[39]:


temp_data1 = pd.DataFrame(q1_array, index= ques_df.index) # here q1_array back to data frame
temp_data2 = pd.DataFrame(q2_array, index= ques_df.index) # here q2_array back to data frame
temp_data = pd.concat([temp_data1, temp_data2], axis=1) # concating data frames here to make sigle data frame
temp_data.shape 


# In[40]:


temp_data['is_duplicate'] = new_df['is_duplicate']


# In[41]:


temp_data.shape


# In[42]:


temp_data.head(5)


# In[43]:


temp_data.tail(5)


# In[44]:


temp_data.sample(5)


# In[45]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
X_train, X_test, y_train, y_test = train_test_split(temp_data.iloc[:,0:-1].values, temp_data.iloc[:,-1],
                                                    test_size=0.2,random_state= 42)


# In[46]:


from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier()
rfc.fit(X_train, y_train)
y_pred = rfc.predict(X_test)
accuracy = accuracy_score(y_test,y_pred) * 100
print("Accuracy of Random Forest",accuracy)


# In[47]:


from xgboost import XGBClassifier
xgb = XGBClassifier()
xgb.fit(X_train, y_train)
y_pred = xgb.predict(X_test)
accuracy = accuracy_score(y_test,y_pred) * 100
print("Accuracy of Random Forest",accuracy)


# In[48]:


'''from sklearn import svm

# Create an SVM classifier
svm_classifier = svm.SVC(kernel='linear')

# Train the model
svm_classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred = svm_classifier.predict(X_test)

# Evaluate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
'''


# In[49]:


from sklearn.neural_network import MLPClassifier

# Create an MLP classifier
mlp = MLPClassifier(hidden_layer_sizes=(10, 10), max_iter=1000, random_state=42)

# Train the classifier
mlp.fit(X_train, y_train)

# Make predictions on the test set
y_pred = mlp.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
accuracy = accuracy  * 100
print("Accuracy:", accuracy )


# In[50]:


from sklearn.neighbors import KNeighborsClassifier

# Create a KNN classifier object
knn = KNeighborsClassifier(n_neighbors=3)

# Train the classifier
knn.fit(X_train, y_train)

# Make predictions on the test set
y_pred = knn.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred) * 100
print("Accuracy:", accuracy)


# In[51]:


from sklearn.tree import DecisionTreeClassifier

# Create a decision tree classifier
clf = DecisionTreeClassifier()

# Train the classifier
clf.fit(X_train, y_train)

# Make predictions on the testing data to model
y_pred = clf.predict(X_test)

# Calculate the accuracy of the classifier
accuracy = accuracy_score(y_test, y_pred) * 100
print("Accuracy:", accuracy)


# In[ ]:





# In[ ]:





# <h1> Advance Feature Engineering </h1>

# In[52]:


df =pd.read_csv("train.csv")


# In[53]:


new_df = df.sample(20000,random_state=2)


# In[54]:


new_df.shape


# In[55]:


new_df.head(5)


# In[56]:


new_df.tail(5)


# In[ ]:





# In[57]:


def questions_preprocessing(question):
    
    question = str(question).lower().strip()

    # Decontracting words
    contractions = { 
    "ain't": "am not",
    "aren't": "are not",
    "can't": "can not",
    "can't've": "can not have",
    "'cause": "because",
    "could've": "could have",
    "couldn't": "could not",
    "couldn't've": "could not have",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hadn't've": "had not have",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he would",
    "he'd've": "he would have",
    "he'll": "he will",
    "he'll've": "he will have",
    "he's": "he is",
    "how'd": "how did",
    "how'd'y": "how do you",
    "how'll": "how will",
    "how's": "how is",
    "i'd": "i would",
    "i'd've": "i would have",
    "i'll": "i will",
    "i'll've": "i will have",
    "i'm": "i am",
    "i've": "i have",
    "isn't": "is not",
    "it'd": "it would",
    "it'd've": "it would have",
    "it'll": "it will",
    "it'll've": "it will have",
    "it's": "it is",
    "let's": "let us",
    "ma'am": "madam",
    "mayn't": "may not",
    "might've": "might have",
    "mightn't": "might not",
    "mightn't've": "might not have",
    "must've": "must have",
    "mustn't": "must not",
    "mustn't've": "must not have",
    "needn't": "need not",
    "needn't've": "need not have",
    "o'clock": "of the clock",
    "oughtn't": "ought not",
    "oughtn't've": "ought not have",
    "shan't": "shall not",
    "sha'n't": "shall not",
    "shan't've": "shall not have",
    "she'd": "she would",
    "she'd've": "she would have",
    "she'll": "she will",
    "she'll've": "she will have",
    "she's": "she is",
    "should've": "should have",
    "shouldn't": "should not",
    "shouldn't've": "should not have",
    "so've": "so have",
    "so's": "so as",
    "that'd": "that would",
    "that'd've": "that would have",
    "that's": "that is",
    "there'd": "there would",
    "there'd've": "there would have",
    "there's": "there is",
    "they'd": "they would",
    "they'd've": "they would have",
    "they'll": "they will",
    "they'll've": "they will have",
    "they're": "they are",
    "they've": "they have",
    "to've": "to have",
    "wasn't": "was not",
    "we'd": "we would",
    "we'd've": "we would have",
    "we'll": "we will",
    "we'll've": "we will have",
    "we're": "we are",
    "we've": "we have",
    "weren't": "were not",
    "what'll": "what will",
    "what'll've": "what will have",
    "what're": "what are",
    "what's": "what is",
    "what've": "what have",
    "when's": "when is",
    "when've": "when have",
    "where'd": "where did",
    "where's": "where is",
    "where've": "where have",
    "who'll": "who will",
    "who'll've": "who will have",
    "who's": "who is",
    "who've": "who have",
    "why's": "why is",
    "why've": "why have",
    "will've": "will have",
    "won't": "will not",
    "won't've": "will not have",
    "would've": "would have",
    "wouldn't": "would not",
    "wouldn't've": "would not have",
    "y'all": "you all",
    "y'all'd": "you all would",
    "y'all'd've": "you all would have",
    "y'all're": "you all are",
    "y'all've": "you all have",
    "you'd": "you would",
    "you'd've": "you would have",
    "you'll": "you will",
    "you'll've": "you will have",
    "you're": "you are",
    "you've": "you have",
    "'ve": " have",
    "n't": " not",
    "'re": " are",
    "'ll": " will"  
    }

    question_decontracted = []

    for word in question.split():
        if word in contractions:
            word = contractions[word]
            
        question_decontracted.append(word)

    question = ' '.join(question_decontracted)
    
    
    # Replace certain special characters with their string equivalents
    question = question.replace('%', ' percent')
    question = question.replace('$', ' dollar ')
    question = question.replace('₹', ' rupee ')
    question = question.replace('€', ' euro ')
    question = question.replace('@', ' at ')
    question = question.replace('R$',  'Brazilian Real')
    question = question.replace('S$',  'Singapore Dollar')
    question = question.replace('NZ$', 'New Zealand Dollar')
    question = question.replace('HK$', 'Hong Kong Dollar')
    question = question.replace('₩',   'South Korean Won')                                
    question = question.replace('₺', 'Turkish Lira')
    question = question.replace('₽',  'Russian Ruble')
    question = question.replace('zł',  'Polish Zloty')
    question = question.replace('Kč',  'Czech Koruna')
    question = question.replace('₪',   'Israeli Shekel')
    question = question.replace('¥',   'Chinese Yuan')
    question = question.replace('₣',   'Swiss Franc')
                                
        
    # The pattern '[math]' appears around 900 times in the whole dataset.
    question = question.replace('[math]', '')
    
    # Replacing some numbers with string equivalents (not perfect, can be done better to account for more cases)
    question = question.replace(',000,000,000 ', 'b ')
    question = question.replace(',000,000 ', 'm ')
    question = question.replace(',000 ', 'k ')
    
    # re is regualar Experssion
    question = re.sub(r'([0-9]+)000000000', r'\1b', question)
    question = re.sub(r'([0-9]+)000000', r'\1m', question)
    question = re.sub(r'([0-9]+)000', r'\1k', question)
    # Removing HTML tags
    question = BeautifulSoup(question)
    question = question.get_text()
    
    # Remove punctuations
    pattern = re.compile('\W')
    question = re.sub(pattern, ' ', question).strip()

    
    return question
    


# In[58]:


questions_preprocessing("That's Great <b>done</b>?")


# In[59]:


new_df['question1'] = new_df['question1'].apply(questions_preprocessing)
new_df['question2'] = new_df['question2'].apply(questions_preprocessing)


# In[60]:


new_df.shape


# In[61]:


new_df.sample(5)


# In[62]:


new_df.head(5)


# In[63]:


new_df.tail(5)


# In[64]:


new_df['chars_in_q1'] = new_df['question1'].str.len() 
new_df['chars_in_q2'] = new_df['question2'].str.len()


# In[65]:


new_df.shape


# In[66]:


new_df.sample(5)


# In[67]:


new_df.head(5)


# In[68]:


new_df.tail(5)


# In[ ]:





# In[69]:


new_df['words_no_words_in_q1'] = new_df['question1'].apply(lambda row: len(row.split(" ")))
new_df['words_no_words_in_q2'] = new_df['question2'].apply(lambda row: len(row.split(" ")))


# In[70]:


new_df.shape


# In[71]:


new_df.head(5)


# In[72]:


new_df.tail(5)


# In[73]:


new_df.tail(5)


# In[74]:


def common_words_in_questions(row):
    word1 = set(map(lambda word: word.lower().strip(), row['question1'].split(" ")))
    word2 = set(map(lambda word: word.lower().strip(), row['question2'].split(" ")))    
    length = len(word1 & word2)
    return length


# In[75]:


new_df['common_words_qs'] = new_df.apply(common_words_in_questions, axis=1)


# In[76]:


new_df.shape


# In[77]:


new_df.head(5)


# In[78]:


new_df.sample(5)


# In[79]:


new_df.tail(5)


# In[80]:


def total_words_in_questions(row):
    word1 = set(map(lambda word: word.lower().strip(), row['question1'].split(" ")))
    word2 = set(map(lambda word: word.lower().strip(), row['question2'].split(" ")))    
    length = (len(word1) + len(word2))
    return length


# In[81]:


new_df['total_words_in_questions'] = new_df.apply(total_words_in_questions, axis=1)


# In[82]:


new_df.shape


# In[83]:


new_df.sample(5)


# In[84]:


new_df.head(5)


# In[85]:


new_df.tail(5)


# In[86]:


new_df['shared_words_in_questions'] = round(new_df['common_words_qs']/new_df['total_words_in_questions'],2)


# In[87]:


new_df.shape


# In[88]:


new_df.head(5)


# In[89]:


new_df.tail(5)


# In[90]:


new_df.sample(5)


# In[91]:


# Advanced Feature adding
from nltk.corpus import stopwords

def token_features_fetching_from_questions(row):
    
    question1 = row['question1']
    question2 = row['question2']
    
    SAFE_DIV = 0.0000001 

    STOP_WORDS = stopwords.words("english")
    
    token_features = [0.0]*8 # bcz of 8 features 0-7
    
    # Converting the Sentence into Tokens: 
    question1_tokens = question1.split()
    question2_tokens = question2.split()
    
    if len(question1_tokens) == 0 or len(question2_tokens) == 0:
        return token_features

    # Get the non-stopwords in Questions
    question1_words = set([word for word in question1_tokens if word not in STOP_WORDS])
    question2_words = set([word for word in question2_tokens if word not in STOP_WORDS])
    
    #Get the stopwords in Questions
    question1_stops = set([word for word in question1_tokens if word in STOP_WORDS])
    question2_stops = set([word for word in question2_tokens if word in STOP_WORDS])
    
    # Get the common non-stopwords from Question pair
    common_word_count = len(question1_words.intersection(question2_words))
    
    # Get the common stopwords from Question pair
    common_stop_count = len(question1_stops.intersection(question2_stops))
    
    # Get the common Tokens from Question pair
    common_token_count = len(set(question1_tokens).intersection(set(question2_tokens)))
    
    token_features[0] = common_word_count / (min(len(question1_words), len(question2_words)) + SAFE_DIV)
    token_features[1] = common_word_count / (max(len(question1_words), len(question2_words)) + SAFE_DIV)
    token_features[2] = common_stop_count / (min(len(question1_stops), len(question2_stops)) + SAFE_DIV)
    token_features[3] = common_stop_count / (max(len(question1_stops), len(question2_stops)) + SAFE_DIV)
    token_features[4] = common_token_count / (min(len(question1_tokens), len(question2_tokens)) + SAFE_DIV)
    token_features[5] = common_token_count / (max(len(question1_tokens), len(question2_tokens)) + SAFE_DIV)
    
    # Last word of Q1 AND Q2 is SAME or NOT
    token_features[6] = int(question1_tokens[-1] == question2_tokens[-1])
    
    # First word of Q1 AND Q2 is AME or NOT
    token_features[7] = int(question1_tokens[0] == question2_tokens[0])
    
    return token_features


# In[92]:


token_features = new_df.apply(token_features_fetching_from_questions, axis=1)

new_df["common_words_count_min"]       = list(map(lambda x: x[0], token_features))
new_df["common_words_count_max"]       = list(map(lambda x: x[1], token_features))
new_df["common_stopwords_count_min"]       = list(map(lambda x: x[2], token_features))
new_df["common_stopwords_count_max"]       = list(map(lambda x: x[3], token_features))
new_df["common_token_count_min"]       = list(map(lambda x: x[4], token_features))
new_df["common_token_count_max"]       = list(map(lambda x: x[5], token_features))
new_df["last_word_matching"]  = list(map(lambda x: x[6], token_features))
new_df["first_word_matching"] = list(map(lambda x: x[7], token_features))


# In[93]:


new_df.shape


# In[94]:


new_df.sample(5)


# In[95]:


new_df.head(5)


# In[96]:


new_df.tail(5)


# In[97]:


import distance

def fetch_length_features(row):
    
    question1 = row['question1']
    question2 = row['question2']
    
    length_features = [0.0]*3
    
    # Converting the Sentence into Tokens: 
    question1_tokens = question1.split()
    question2_tokens = question2.split()
    
    if len(question1_tokens) == 0 or len(question2_tokens) == 0:
        return length_features
    
    
    #Average Token Length of both Questions
    length_features[0] = (len(question1_tokens) + len(question2_tokens))/2
    
    # Absolute length features
    length_features[1] = abs(len(question1_tokens) - len(question2_tokens))
    
    strs = list(distance.lcsubstrings(question1, question2))
    length_features[2] = len(strs[0]) / (min(len(question1), len(question2)) + 1)
    
    return length_features
    


# In[98]:


length_features = new_df.apply(fetch_length_features, axis=1)

new_df['average_length_of_question'] = list(map(lambda x: x[0], length_features))
new_df['absolute_difference_of_qs_length'] = list(map(lambda x: x[1], length_features))
new_df['longest_substring_ratio'] = list(map(lambda x: x[2], length_features))


# In[99]:


new_df.shape


# In[100]:


new_df.head(5)


# In[101]:


new_df.sample(5)


# In[102]:


new_df.tail(5)


# In[103]:


# Fuzzy Features
from fuzzywuzzy import fuzz

def fetch_fuzzy_features_from_questions(row):
    
    question1 = row['question1']
    question2 = row['question2']
    
    fuzzy_features = [0.0]*4
    

    # fuzz_partial_ratio btw question 1 and Question 2
    fuzzy_features[0] = fuzz.partial_ratio(question1, question2)

    # fuzz_ratio btw question 1 and Question 2
    fuzzy_features[1] = fuzz.QRatio(question1, question2)
    
    # token_sort_ratio btw question 1 and Question 2
    fuzzy_features[2] = fuzz.token_sort_ratio(question1, question2)

    # token_set_ratio btw question 1 and Question 2
    fuzzy_features[3] = fuzz.token_set_ratio(question1, question2)

    return fuzzy_features


# In[104]:


fuzzy_features = new_df.apply(fetch_fuzzy_features_from_questions, axis=1)

# Creating new feature columns for fuzzy features in already given data set
new_df['fuzz_partial_ratio'] = list(map(lambda x: x[0], fuzzy_features))
new_df['fuzz_ratio'] = list(map(lambda x: x[1], fuzzy_features))
new_df['token_sort_ratio'] = list(map(lambda x: x[2], fuzzy_features))
new_df['token_set_ratio'] = list(map(lambda x: x[3], fuzzy_features))


# In[108]:


new_df.shape


# In[109]:


new_df.head(5)


# In[110]:


new_df.sample(5)


# In[111]:


new_df.tail(5)


# In[112]:


sns.pairplot(new_df[['common_token_count_min', 'common_words_count_min', 'common_stopwords_count_min', 'is_duplicate']], hue='is_duplicate',diag_kind="hist")


# In[113]:


sns.pairplot(new_df[['common_token_count_max', 'common_words_count_max', 'common_stopwords_count_max', 'is_duplicate']],hue='is_duplicate',diag_kind="kde")


# In[114]:


sns.pairplot(new_df[['last_word_matching', 'first_word_matching', 'is_duplicate']],hue='is_duplicate')


# In[115]:


sns.pairplot(new_df[['average_length_of_question', 'absolute_difference_of_qs_length','longest_substring_ratio', 'is_duplicate']],hue='is_duplicate')


# In[116]:


sns.pairplot(new_df[['fuzz_ratio', 'fuzz_partial_ratio','token_sort_ratio','token_set_ratio', 'is_duplicate']],hue='is_duplicate')


# In[117]:


from sklearn.preprocessing import MinMaxScaler

X = MinMaxScaler().fit_transform(new_df[['common_stopwords_count_min', 'common_stopwords_count_max' ,'common_words_count_min', 'common_words_count_max', 'common_token_count_min' , 'common_token_count_max' , 'last_word_matching', 'first_word_matching' , 'absolute_difference_of_qs_length' , 'average_length_of_question' , 'token_set_ratio' , 'token_sort_ratio' , 'fuzz_ratio' , 'fuzz_partial_ratio' , 'longest_substring_ratio']])
y = new_df['is_duplicate'].values


# In[118]:


# Using TSNE for Dimentionality reduction for 15 Features(Generated after cleaning the data) to 3 dimention

from sklearn.manifold import TSNE

tsne2d = TSNE(
    n_components=2,init='random',random_state=42 ,method='barnes_hut', n_iter=2000,verbose=2,angle=0.5
).fit_transform(X)


# In[119]:


new_df1 = pd.DataFrame({'x':tsne2d[:,0], 'y':tsne2d[:,1] ,'label':y})

# draw the plot in appropriate place in the grid
sns.lmplot(data=new_df1, x='x', y='y', hue='label', fit_reg=False,palette="Set1",markers=['o','.'])


# In[120]:


tsne3d = TSNE(
    n_components=3, init='random', random_state=42, method='barnes_hut', n_iter=1000, verbose=2,
    angle=0.5
).fit_transform(X)


# In[121]:


import plotly.graph_objs as pgo
import plotly.tools as ptls
import plotly.offline as po
po.init_notebook_mode(connected=True)

trace1 = pgo.Scatter3d(x=tsne3d[:,0], y=tsne3d[:,1],z=tsne3d[:,2],mode='markers',marker=dict(sizemode='diameter',
        color = y,colorscale = 'Portland',colorbar = dict(title = 'duplicate'),
        line=dict(color='rgb(255, 255, 255)'),opacity=0.75) )

data=[trace1]
layout=dict(height=1000, width=1000, title='3D embedding with engineered features')
fig=dict(data=data, layout=layout)
po.iplot(fig, filename='3DBubble')


# In[122]:


ques_df = new_df[['question1','question2']]


# In[123]:


ques_df.shape


# In[124]:


ques_df.head(5)


# In[125]:


ques_df.sample(5)


# In[126]:


ques_df.tail(5)


# In[127]:


final_data = new_df.drop(columns=['id','qid1','qid2','question1','question2'])


# In[128]:


final_data.shape


# In[129]:


final_data.head(5)


# In[130]:


final_data.sample(5)


# In[131]:


final_data.tail(5)


# In[132]:


from sklearn.feature_extraction.text import CountVectorizer
# merge texts
questions = list(ques_df['question1']) + list(ques_df['question2'])

cv = CountVectorizer(max_features=3000)
q1_array, q2_array = np.vsplit(cv.fit_transform(questions).toarray(),2)


# In[133]:


temp_df1 = pd.DataFrame(q1_array, index= ques_df.index)
temp_df2 = pd.DataFrame(q2_array, index= ques_df.index)
temp_df = pd.concat([temp_df1, temp_df2], axis=1)


# In[134]:


temp_df.shape


# In[135]:


temp_data.sample(5)


# In[ ]:





# In[136]:


final_data = pd.concat([final_data, temp_df], axis=1)


# In[137]:


final_data.shape


# In[138]:


final_data.sample(5)


# In[139]:


from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(final_data.iloc[:,1:].values,final_data.iloc[:,0].values,test_size=0.2,random_state=1)
a = final_data.iloc[:,1:]


# In[140]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
rfc = RandomForestClassifier()
rfc.fit(X_train,y_train)
y_pred = rfc.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)


# In[141]:


"""from sklearn import svm

svm_classifier = svm.SVC(kernel='linear')

# Train the model
svm_classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred = svm_classifier.predict(X_test)

# Evaluate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
"""


# In[142]:


from xgboost import XGBClassifier
xgb = XGBClassifier()
xgb.fit(X_train,y_train)
y_pred = xgb.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)


# In[143]:


from sklearn.tree import DecisionTreeClassifier

# Create a decision tree classifier
clf = DecisionTreeClassifier()

# Train the classifier
clf.fit(X_train, y_train)

# Make predictions on the testing data to model
y_pred = clf.predict(X_test)

# Calculate the accuracy of the classifier
accuracy = accuracy_score(y_test, y_pred) * 100
print("Accuracy:", accuracy)


# In[144]:


from sklearn.neural_network import MLPClassifier

# Create an MLP classifier
mlp = MLPClassifier(hidden_layer_sizes=(10, 10), max_iter=1000, random_state=42)

# Train the classifier
mlp.fit(X_train, y_train)

# Make predictions on the test set
y_pred = mlp.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
accuracy = accuracy  * 100
print("Accuracy:", accuracy )


# In[145]:


from sklearn.neighbors import KNeighborsClassifier

# Create a KNN classifier object
knn = KNeighborsClassifier(n_neighbors=3)

# Train the KNN classifier
knn.fit(X_train, y_train)

# Make predictions on the test set
y_pred = knn.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)


# In[146]:


from sklearn.metrics import confusion_matrix


# In[147]:


# for random forest model
confusion_matrix(y_test,y_pred)


# In[149]:


# for xgboost model
confusion_matrix(y_test,y_pred)


# In[150]:


def test_common_words_in_question(ques1,ques2):
    word1 = set(map(lambda word: word.lower().strip(), q1.split(" ")))
    word2 = set(map(lambda word: word.lower().strip(), q2.split(" ")))   
    length = len(word1 & word2)
    return length


# In[168]:


def test_total_words(ques1,ques2):
    word1 = set(map(lambda word: word.lower().strip(), ques1.split(" ")))
    word2 = set(map(lambda word: word.lower().strip(), ques2.split(" ")))    
    length = (len(word1) + len(word2))
    return length


# In[178]:


# Advanced Feature adding
from nltk.corpus import stopwords

def test_token_features_fetching_from_questions(question1,question2):

    
    SAFE_DIV = 0.0000001 

    STOP_WORDS = stopwords.words("english")
    
    token_features = [0.0]*8 # bcz of 8 features 0-7
    
    # Converting the Sentence into Tokens: 
    question1_tokens = question1.split()
    question2_tokens = question2.split()
    
    if len(question1_tokens) == 0 or len(question2_tokens) == 0:
        return token_features

    # Get the non-stopwords in Questions
    question1_words = set([word for word in question1_tokens if word not in STOP_WORDS])
    question2_words = set([word for word in question2_tokens if word not in STOP_WORDS])
    
    #Get the stopwords in Questions
    question1_stops = set([word for word in question1_tokens if word in STOP_WORDS])
    question2_stops = set([word for word in question2_tokens if word in STOP_WORDS])
    
    # Get the common non-stopwords from Question pair
    common_word_count = len(question1_words.intersection(question2_words))
    
    # Get the common stopwords from Question pair
    common_stop_count = len(question1_stops.intersection(question2_stops))
    
    # Get the common Tokens from Question pair
    common_token_count = len(set(question1_tokens).intersection(set(question2_tokens)))
    
    token_features[0] = common_word_count / (min(len(question1_words), len(question2_words)) + SAFE_DIV)
    token_features[1] = common_word_count / (max(len(question1_words), len(question2_words)) + SAFE_DIV)
    token_features[2] = common_stop_count / (min(len(question1_stops), len(question2_stops)) + SAFE_DIV)
    token_features[3] = common_stop_count / (max(len(question1_stops), len(question2_stops)) + SAFE_DIV)
    token_features[4] = common_token_count / (min(len(question1_tokens), len(question2_tokens)) + SAFE_DIV)
    token_features[5] = common_token_count / (max(len(question1_tokens), len(question2_tokens)) + SAFE_DIV)
    
    # Last word of Q1 AND Q2 is SAME or NOT
    token_features[6] = int(question1_tokens[-1] == question2_tokens[-1])
    
    # First word of Q1 AND Q2 is AME or NOT
    token_features[7] = int(question1_tokens[0] == question2_tokens[0])
    
    return token_features


# In[179]:


import distance

def test_fetch_length_features(question1,question2):
    
    length_features = [0.0]*3
    
    # Converting the Sentence into Tokens: 
    question1_tokens = question1.split()
    question2_tokens = question2.split()
    
    if len(question1_tokens) == 0 or len(question2_tokens) == 0:
        return length_features
    
    
    #Average Token Length of both Questions
    length_features[0] = (len(question1_tokens) + len(question2_tokens))/2
    
    # Absolute length features
    length_features[1] = abs(len(question1_tokens) - len(question2_tokens))
    
    strs = list(distance.lcsubstrings(question1, question2))
    length_features[2] = len(strs[0]) / (min(len(question1), len(question2)) + 1)
    
    return length_features
    


# In[180]:


# Fuzzy Features
from fuzzywuzzy import fuzz

def test_fetch_fuzzy_features_from_questions(question1, question2):
    
    fuzzy_features = [0.0]*4
    

    # fuzz_partial_ratio btw question 1 and Question 2
    fuzzy_features[0] = fuzz.partial_ratio(question1, question2)

    # fuzz_ratio btw question 1 and Question 2
    fuzzy_features[1] = fuzz.QRatio(question1, question2)
    
    # token_sort_ratio btw question 1 and Question 2
    fuzzy_features[2] = fuzz.token_sort_ratio(question1, question2)

    # token_set_ratio btw question 1 and Question 2
    fuzzy_features[3] = fuzz.token_set_ratio(question1, question2)

    return fuzzy_features


# In[181]:


def query_point_creator(question1,question2):
    
    input_query = []
    
    # preprocess
    question1 = questions_preprocessing(q1)
    question2 = questions_preprocessing(q2)
    
    # fetch basic features
    input_query.append(len(question1))
    input_query.append(len(question2))
    
    input_query.append(len(question1.split(" ")))
    input_query.append(len(question2.split(" ")))
    
    input_query.append(test_common_words_in_question(question1,question2))
    input_query.append(test_total_words(question1,question2))
    input_query.append(round(test_common_words_in_question(question1,question2)/test_total_words(question1,question2),2))
    
    # fetch token features
    token_features = test_token_features_fetching_from_questions(question1,question2)
    input_query.extend(token_features)
    
    # fetch length based features
    length_features = test_fetch_length_features(question1,question2)
    input_query.extend(length_features)
    
    # fetch fuzzy features
    fuzzy_features = test_fetch_fuzzy_features_from_questions(question1,question2)
    input_query.extend(fuzzy_features)
    
    # bag of words feature for q1
    question1_bow = cv.transform([question1]).toarray()
    
    # bag oo words feature for q2
    question2_bow = cv.transform([question2]).toarray()
    
    
    
    return np.hstack((np.array(input_query).reshape(1,22),question1_bow,question2_bow))


# In[ ]:





# In[ ]:





# In[ ]:




