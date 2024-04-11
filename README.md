# Cyber-Bully-Tweet-Classifier
AI model for the classification and detection of cyberbullying tweets, contributing to the creation of a safer online environment.
# Step 1 : Download the dataset(Cyber bully tweet dataset.csv)
# Step 2 : Write the code

import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import re 
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px 
import emoji
import string
import nltk
from PIL import Image
from collections import Counter
from wordcloud import WordCloud, ImageColorGenerator, STOPWORDS
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
import pickle

# %%
df = pd.read_csv('cyberbullying_tweets.csv')

# %%
df.info()

# %%
df.head(10)

# %%
df.tail()

# %%
df.isnull().sum()

# %%
df['cyberbullying_type'].value_counts()

# %%
df = df.rename(columns={
    'tweet_text':'text','cyberbullying_type':'sentiment'
})

# %%
df.head()

# %%
df['sentiment_encoded']=df['sentiment'].replace({
'religion': 1,
'age': 2,
'ethnicity':3,
'gender':4,
'other_cyberbullying':5,
'not_cyberbullying':6
})

# %%
df.tail(10)
# %%

nltk.download('punkt')
# %%

nltk.download('wordnet')

# %%
import nltk
nltk.download('stopwords')


# %%
stop_words = set(stopwords.words('english'))

# %%
###preprocessing of text
# function to remove emojis
def strip_emoji(text):
    return emoji.replace_emoji(text,replace="")

# %%
#function to convert text to Lowercase, remove (unwanted characters, ursi, non-utf stuff, numbers, stopwrors)

def strip_all_entities (text):
    text= text.replace('\r', '').replace('\n','').lower()
    text= re.sub(r"(?:\@|https?|-\://)\S+",'', text)
    text= re.sub(r" [^\x00-\x7f]",r'', text)
    text= re.sub('[0-9]+','',text)
    
    stopchars =string.punctuation 
    table=str.maketrans('','', stopchars)
    text=text.translate(table)
    
    text=[word for word in text.split() if word not in stop_words] 
    text=' '.join(text)
    
    return text

# import re

# def strip_all_entities(text):
#     text = text.replace('\r', '').replace('\n','').lower()
#     text = re.sub(r"(?:@[A-Za-z0-9_]+|https?://\S+)", '', text)  # Corrected regular expression
#     text = re.sub(r"[^\x00-\x7f]", '', text)
#     text = re.sub('[0-9]+', '', text)

#     stopchars = string.punctuation 
#     table = str.maketrans('', '', stopchars)
#     text = text.translate(table)

#     text = [word for word in text.split() if word not in stop_words] 
#     text = ' '.join(text)

#     return text


# %%
#function to remove contractions
def decontract(text):
    
    text = re.sub(r"cant\'t'","can not", text)
    text = re.sub(r"n\'t", "not", text)
    text = re.sub(r"\'re", " are", text) 
    text = re.sub(r"\'s", " is", text)
    text = re.sub(r"\'d", " would", text)
    text = re.sub(r"\'ll", "will", text)
    text = re.sub(r"\'t", " not", text) 
    text = re.sub(r"\'ve", "have", text)
    text = re.sub(r"\'m", " am", text)
    
    return text

# %%
#junction to clean hastags
def clean_hashtags (tweet):
    now_tweet=" ".join(word.strip() for word in re.split('#(?!(?:hashtag)\b)[\w-]+(?=(?:\s+#[\w-]+)*\s*$)', tweet)) 
    new_tweet2 = " ".join(word.strip() for word in re.split('#|_', now_tweet))
   
    return new_tweet2


# %%
# function to filter special charoters
def filter_chars(a):
    sent = []
    for word in a.split(" "):
        if ('$' in word) | ('&' in word):
            sent.append("")
        else:
            sent.append(word)
    return ' '.join(sent)

# %%
#removing sequences and applying stemming
def remove_mult_spaces(text):
    return re.sub("\s\s+", " ", text)

def stemmer(text):
    tokenized = nltk.word_tokenize(text)
    ps = PorterStemmer()
    return ' '.join([ps.stem(words) for words in tokenized])

def lemmatize(text):
    tokenized = nltk.word_tokenize(text)
    lm = WordNetLemmatizer()
    return ' '.join([lm.lemmatize(words) for words in tokenized])


# %%
# using all functions

def preprocess(text):
    text = strip_emoji(text)
    text = decontract(text)
    text = strip_all_entities(text)
    text = clean_hashtags(text)
    text = filter_chars(text)
    text = remove_mult_spaces(text)
    text = stemmer(text)
    text = lemmatize(text)
    return text



# %%
print("Column Names:", df.columns)

# %%
##cleaned text
df['cleaned_text'] = df['text'].apply(preprocess)
df.head()

# %%
# dealing with duplicates
df['cleaned_text'].duplicated().sum()

# %%
df.drop_duplicates('cleaned_text',inplace = True)

# %%
# tokenization
df['tweet_list'] = df['cleaned_text'].apply(word_tokenize)
df.head()

# %%
#EDA

#checking length of various text
text_len = []
for text in df.tweet_list:
    tweet_len = len(text)
    text_len.append(tweet_len)
df['text_len'] = text_len
df.head()

# %%
plt.figure(figsize=(15,8))
ax = sns.countplot(x='text_len', data = df, palette='mako')
plt.title('Count of words in tweets', fontsize=20)
plt.yticks([])
ax.bar_label(ax.containers[0])
plt.ylabel('count')
plt.xlabel('')
plt.show()


# %%
#removing text without words
df = df[df['text_len']!=0]

# %%
df.shape

# %%
#function to create a word cloud

def plot_wordcloud(cyberbullying_type):
    string = ""
    for i in df[df.sentiment == cyberbullying_type].cleaned_text.values:
        string = string + " " + i.strip()

    wordcloud = WordCloud(background_color = 'white',max_words = 2000, max_font_size = 256,random_state=42).generate(string)
    #plot the WordCLoud image
    plt.figure(figsize = (8, 8), facecolor = None)
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.tight_layout(pad = 0)
    plt.title(cyberbullying_type)
    plt.show()
    del string

# %%
#splitting data based on sentiment for EDA
not_cyberbullying_type = df [df['sentiment']=='not_cyberbullying']
gender_type = df [df['sentiment']=='gender']
religion_type = df[df['sentiment']=='religion']
other_cyberbullying_type = df[df['sentiment']=='other_cyberbullying']
age_type = df[df['sentiment']=='age']
ethnicity_type = df[df['sentiment']=='ethnicity']


# %%
#EDA
gender = Counter([item for sublist in gender_type ['tweet_list'] for item in sublist])
top20_gender = pd.DataFrame(gender.most_common(20)) 
top20_gender.columns = ['Top Words', 'Count']
top20_gender.head(20)


# %%
fig = plt.figure(figsize=(15,8))
sns.barplot(data=top20_gender, y="Count", x="Top Words")
plt.title("Top 20 words in Gender Cyberbullying")


# %%
plot_wordcloud('gender')

# %%
religion = Counter([item for sublist in religion_type['tweet_list'] for item in sublist])
top20_religion = pd.DataFrame(religion.most_common(20))
top20_religion.columns = ['Top Words', 'Count']
top20_religion.style.background_gradient(cmap='Greens')


# %%
fig = plt.figure(figsize=(15,8))
sns.barplot(data=top20_religion, y="Count", x="Top Words")
plt.title("Top 20 words in Religion Cyberbullying")


# %%
plot_wordcloud('religion')

# %%
#age based cyber bulllying
age = Counter([item for sublist in age_type['tweet_list'] for item in sublist]) 
top20_age = pd.DataFrame(age.most_common(20))
top20_age.columns = ['Top Words', 'Count']
top20_age.style.background_gradient(cmap='Greens')

# %%
fig = plt.figure(figsize=(15,8))
sns.barplot(data=top20_age, y="Count", x="Top Words") 
plt.title("Top 20 words in Age Cyberbullying")

# %%
plot_wordcloud('age')

# %%
#ethnicity based on cyber bullying
ethnicity = Counter([item for sublist in ethnicity_type['tweet_list'] for item in sublist]) 
top20_ethnicity = pd.DataFrame(ethnicity.most_common(20))
top20_ethnicity.columns = ['Top Words', 'Count']
top20_ethnicity.style.background_gradient(cmap='Greens')

# %%
fig = plt.figure(figsize=(15,8))
sns.barplot(data=top20_ethnicity, y="Count", x="Top Words") 
plt.title("Top 20 words in Age Cyberbullying")

# %%


# %%
plot_wordcloud('ethnicity')

# %%
other_cyberbullying = Counter([item for sublist in other_cyberbullying_type['tweet_list'] for item in sublist]) 
top20_other_cyberbullying = pd.DataFrame(other_cyberbullying.most_common(20))
top20_other_cyberbullying.columns = ['Top Words', 'Count']
top20_other_cyberbullying.style.background_gradient(cmap='Greens')

# %%
fig = plt.figure(figsize=(15,8))
sns.barplot(data=top20_other_cyberbullying, y="Count", x="Top Words") 
plt.title("Top 20 words in Age Cyberbullying")

# %%
plot_wordcloud('other_cyberbullying')

# %%
not_cyberbullying = Counter([item for sublist in not_cyberbullying_type['tweet_list'] for item in sublist]) 
top20_not_cyberbullying = pd.DataFrame(not_cyberbullying.most_common(20))
top20_not_cyberbullying.columns = ['Top Words', 'Count']
top20_not_cyberbullying.style.background_gradient(cmap='Greens')

# %%

fig = plt.figure(figsize=(15,8))
sns.barplot(data=top20_not_cyberbullying, y="Count", x="Top Words") 
plt.title("Top 20 words in Age Cyberbullying")

# %%

plot_wordcloud('not_cyberbullying')

# %%
df.head()

# %%
sentiments = ["religion", "age", "ethnicity", "gender", "other_cyberbullying", "not_cyberbullying"]

# %%
#modelling
#splitting data in test and train
X,Y = df['cleaned_text'],df['sentiment_encoded']

# %%
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3, stratify = Y, random_state = 42) 
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

# %%
#tf-idf vectorization
tf_idf = TfidfVectorizer()
X_train_tf = tf_idf.fit_transform(X_train)
X_test_tf = tf_idf.transform(X_test)
print(X_train_tf.shape) 
print(X_test_tf.shape)

# %%
with open('tfidf_vectorizer.pkl','wb') as f:
    pickle.dump(tf_idf, f)

# %%
#trying different models
#Logistics
log_reg = LogisticRegression()


# %%
log_cv_score = cross_val_score(log_reg, X_train_tf,y_train, cv=5, scoring='f1_macro',n_jobs=-1)

# %%
mean_log_cv = np.mean(log_cv_score)
mean_log_cv

# %%
#support vector
lin_svc = LinearSVC()


# %%
lin_svc_cv_score = cross_val_score(lin_svc,X_train_tf,y_train, cv=5, scoring='f1_macro',n_jobs=-1) 
mean_lin_svc_cv = np.mean(lin_svc_cv_score) 
mean_lin_svc_cv

# %%
#naive bayes classifier
multiNB = MultinomialNB()

# %%
multiNB_cv_score = cross_val_score(multiNB, X_train_tf,y_train,cv=5, scoring='f1_macro',n_jobs=-1)
mean_multiNB_cv = np.mean(multiNB_cv_score)
mean_multiNB_cv 

# %%
#decision tree classifier
dtree = DecisionTreeClassifier()

# %%
dtree_cv_score = cross_val_score(dtree,X_train_tf,y_train, cv=5, scoring='f1_macro',n_jobs=-1) 
mean_dtree_cv = np.mean(dtree_cv_score) 
mean_dtree_cv

# %%
#random forest classifier
rand_forest = RandomForestClassifier()

# %%
rand_forest_cv_score = cross_val_score(rand_forest, X_train_tf,y_train,cv=5, scoring='f1_macro', n_jobs=-1) 
mean_rand_forest_cv = np.mean(rand_forest_cv_score)
mean_rand_forest_cv

# %%
#ada boost classifier
adab = AdaBoostClassifier()

# %%
adab_cv_score = cross_val_score(adab, X_train_tf,y_train,cv=5, scoring='f1_macro',n_jobs=-1)
mean_adab_cv = np.mean(adab_cv_score)
mean_adab_cv

# %%
#By trying different models we can see logistic regression, svm and random forest classifier performed similarly, so among these
#tuning svc
svc1 =  LinearSVC()
param_grid = {'C': [0.0001,0.001,0.01,0.1,1,10], 
              'loss':['hinge', 'squared_hinge'],
              'fit_intercept': [True, False]}
grid_search = GridSearchCV(svc1,param_grid,cv=5,scoring='f1_macro',n_jobs=-1, verbose=0, return_train_score=True) 
grid_search.fit(X_train_tf,y_train)

# %%
grid_search.best_estimator_

# %%
grid_search.best_score_

# %%
#evaluation
lin_svc.fit(X_train_tf,y_train)
y_pred = lin_svc.predict(X_test_tf)

# %%
import pickle 
with open('model.pkl','wb') as f:
    pickle.dump(lin_svc, f)

# %%
def print_confusion_matrix(confusion_matrix, class_names, figsize = (10,7), fontsize=14):
    df_cm = pd.DataFrame(confusion_matrix, index=class_names, columns=class_names)
    fig = plt.figure(figsize=figsize)
    try:
        heatmap = sns.heatmap(df_cm, annot=True, fmt="d")
    except ValueError:
        raise ValueError("Confusion matrix values must be integers.")
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize) 
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=fontsize)
    plt.ylabel('Truth')
    plt.xlabel('Prediction')


# %%
cm = confusion_matrix(y_test,y_pred)
print_confusion_matrix(cm,sentiments)

# %%
print('Classification Report:\n', classification_report(y_test, y_pred, target_names=sentiments))

#prediction

import pickle

# Load the TF-IDF vectorizer
with open('tfidf_vectorizer.pkl', 'rb') as f:
    tfidf_vectorizer = pickle.load(f)

# Load the trained model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# Preprocess the new text input
def preprocess_text(text):
    # Apply the same preprocessing steps as you did for your training data
    text = preprocess(text)
    return text

# Vectorize the preprocessed text
def vectorize_text(text):
    # Convert the text to TF-IDF vector
    text_vector = tfidf_vectorizer.transform([text])
    return text_vector

# Function to predict the sentiment
def predict_sentiment(text):
    # Preprocess the text
    preprocessed_text = preprocess_text(text)
    # Vectorize the preprocessed text
    text_vector = vectorize_text(preprocessed_text)
    # Make predictions using the trained model
    prediction = model.predict(text_vector)
    return prediction

# Example usage
new_text = input("Enter your text")
predicted_sentiment = predict_sentiment(new_text)
print("Predicted Sentiment:", predicted_sentiment)


