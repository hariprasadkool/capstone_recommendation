# -*- coding: utf-8 -*-
"""model.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1KrBWu33_FieuagWYyLd5W_Tfv64-ZKtU

## Capstone Project
- Sentiment Based Product Recommendation System

This is divide into -
1. Data processing, text processing, feature extraction, model buildibng
2. Building a recommendation system
3. Improving the recommendations using the sentiment analysis model choosen

### 1. Data processing, text processing, feature extraciton, model building
"""

#from google.colab import drive
#drive.mount('/content/gdrive')

# Importing  libraries

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt



pd.set_option('display.max_columns',200)
pd.set_option('display.max_rows',100)

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix, classification_report

import nltk
from nltk.corpus import wordnet
import string
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.tokenize import WhitespaceTokenizer
from nltk.stem import WordNetLemmatizer
from wordcloud import WordCloud

nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('omw-1.4')

# Reading 
# reviews_dataset=pd.read_csv('/content/gdrive/MyDrive/sentiment/sample30.csv')
reviews_dataset=pd.read_csv('./data/sample30.csv')

reviews_dataset.head()

# Checking for missing values in data
reviews_dataset.info()

#plotting application data distribution
plt.figure(figsize=(20,10))
sns.countplot(x='name', data= reviews_dataset)
plt.show()

# Concat review title and review text which would be used for sentiment analysis
for_sentiment_analysis=reviews_dataset
for_sentiment_analysis['reviews_title_text']= reviews_dataset['reviews_title'].fillna('') +" "+ reviews_dataset['reviews_text']

# Drop one row where user_sentiment is null
for_sentiment_analysis=for_sentiment_analysis[for_sentiment_analysis['user_sentiment'].isnull()== False]
for_sentiment_analysis.reset_index(drop=True)

# Function that returns the wordnet object value corresponding to the POS tag

def get_wordnet_pos(pos_tag):
    if pos_tag.startswith('J'):
        return wordnet.ADJ
    elif pos_tag.startswith('V'):
        return wordnet.VERB
    elif pos_tag.startswith('N'):
        return wordnet.NOUN
    elif pos_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN
    
# Function for clean the text

def clean_text(text):

    text = text.lower()
    text = [word.strip(string.punctuation) for word in text.split(" ")]
    text = [word for word in text if not any(c.isdigit() for c in word)]
    # remove stop words
    stop = stopwords.words('english')
    text = [x for x in text if x not in stop]
    text = [t for t in text if len(t) > 0]
    
    # pos tag text
    pos_tags = pos_tag(text)
    # lemmatize text
    text = [WordNetLemmatizer().lemmatize(t[0], get_wordnet_pos(t[1])) for t in pos_tags]
    # remove words with only one letter
    text = [t for t in text if len(t) > 1]
    
    # join all
    text = " ".join(text)
    return(text)

# clean 
for_sentiment_analysis["reviews_clean"] = for_sentiment_analysis.apply(lambda x: clean_text(x['reviews_title_text']),axis=1)

for_sentiment_analysis.head(5)

words_per_review = for_sentiment_analysis.reviews_clean.apply(lambda x: len(x.split(" ")))

print('Average words:', words_per_review.mean())

"""On an average there are around 20 words per review"""

percent_val = 100 * for_sentiment_analysis['user_sentiment'].value_counts()/len(for_sentiment_analysis)
percent_val.plot.bar()
plt.show()

"""We observe data is highly imabalanced based on user sentiment  """

percent_val2 = 100 * for_sentiment_analysis['reviews_rating'].value_counts()/len(for_sentiment_analysis)
percent_val2.plot.bar()
plt.show()

""" 70% of the records the user has rated 5"""

# Word clouds for postive and negative reviews 

sns.set(font_scale=2)
plt.figure(figsize = (15,10))

plt.subplot(1, 2, 1)
plt.title('Positive')
positive_reviews=for_sentiment_analysis.loc[for_sentiment_analysis.user_sentiment=='Positive',['reviews_clean']]
word_cloud_text = ''.join(positive_reviews['reviews_clean'])
wordcloud = WordCloud(max_font_size=100, 
                      max_words=100, 
                      background_color="grey", 
                      scale = 10, 
                      width=1000, 
                      height=400 
                     ).generate(word_cloud_text)

plt.imshow(wordcloud, 
           interpolation="bilinear") 
plt.axis("off")
plt.tight_layout()


plt.subplot(1, 2, 2)
plt.title('Negative')
negative_reviews=for_sentiment_analysis.loc[for_sentiment_analysis.user_sentiment=='Negative',['reviews_clean']]

word_cloud_text = ''.join(negative_reviews['reviews_clean'])

wordcloud = WordCloud(max_font_size=100, 
                      max_words=100, 
                      background_color="grey", 
                      scale = 10, 
                      width=1000, 
                      height=400 
                     ).generate(word_cloud_text)

plt.imshow(wordcloud, 
           interpolation="bilinear") 
plt.axis("off")
plt.tight_layout()

"""Positive reviews are mostly - 'great', 'promotion', 'love', 'use'. 
 Negative reviews are mostly - 'bad', 'horrible', 'little', 'nothing'
"""

# Mapping positive sentiment as 1 and negative as 0 

for_sentiment_analysis['Sentiment_coded'] = np.where(for_sentiment_analysis.user_sentiment == 'Positive',1,0)

# Printing the counts of each class
for_sentiment_analysis['Sentiment_coded'].value_counts()

for_sentiment_analysis.head(5)

# Converting the clean & processed review text to features using Tf-idf

from sklearn.feature_extraction.text import TfidfVectorizer

### Creating a python object of the class CountVectorizer
tfidf_count = TfidfVectorizer(tokenizer= word_tokenize, # type of tokenization
                               stop_words=stopwords.words('english'), # List of stopwords
                               ngram_range=(1,1)) # number of n-grams

tfidf_data = tfidf_count.fit_transform(for_sentiment_analysis["reviews_clean"])

# Saving the vectorizer so that it can be used later while deploying the model

import pickle

# Save to file in the current working directory
pkl_filename = "Tfidf_pickle.pkl"
with open(pkl_filename, 'wb') as file:
    pickle.dump(tfidf_count, file)

# Load from file
with open(pkl_filename, 'rb') as file:
    pickled_tfidf_vectorizer = pickle.load(file)

# Splitting the data into train and test

X_train_tfidf, X_test_tfidf, y_train_tfidf, y_test_tfidf = train_test_split(tfidf_data,
                                                                            for_sentiment_analysis['Sentiment_coded'],
                                                                            test_size = 0.3,
                                                                            random_state = 50)

print(X_train_tfidf.shape)
print(X_test_tfidf.shape)
print(y_train_tfidf.shape)
print(y_test_tfidf.shape)

"""Build 3 different ML models to predict sentiment based on title and text reviews

- **1 Logistic Regression model**
"""

# Training the data using Logistic Regression model and checking the performance based on recall 

lr = LogisticRegression()
  
# train the model on train set
lr.fit(X_train_tfidf, y_train_tfidf.ravel())

predictions = lr.predict(X_train_tfidf)
  
# Confusion matrix 
confusion = confusion_matrix(y_train_tfidf, predictions)
print(confusion)


# print classification report
print(classification_report(y_train_tfidf, predictions))

predictions = lr.predict(X_test_tfidf)
  
# Confusion matrix 
confusion = confusion_matrix(y_test_tfidf, predictions)
print(confusion)


# print classification report
print(classification_report(y_test_tfidf, predictions))
print("Accuracy : ",accuracy_score(y_test_tfidf, predictions))
print("Recall: ",recall_score(y_test_tfidf, predictions))

"""handle the issue of class imbalance. We will use SMOTE technique """

import six
import sys
sys.modules['sklearn.externals.six'] = six

print("Before OverSampling, counts of label '1': {}".format(sum(y_train_tfidf == 1)))
print("Before OverSampling, counts of label '0': {} \n".format(sum(y_train_tfidf == 0)))

#  imblearn library
from imblearn.over_sampling import SMOTE

sm = SMOTE(random_state = 2)
X_train_res, y_train_res = sm.fit_resample(X_train_tfidf, y_train_tfidf.ravel())

print('AfterOverSampling, the shape of train_X: {}'.format(X_train_res.shape))
print('AfterOverSampling, the shape of train_y: {} \n'.format(y_train_res.shape))

print("AfterOverSampling, counts of label '1': {}".format(sum(y_train_res == 1)))
print("AfterOverSampling, counts of label '0': {}".format(sum(y_train_res == 0)))

"""- ** Building the Logistic Regression model**"""

# Training  imbalance

lr1 = LogisticRegression(solver='lbfgs', max_iter=1000)
lr1.fit(X_train_res, y_train_res.ravel())
predictions1 = lr1.predict(X_test_tfidf)

# Confusion matrix 
confusion = confusion_matrix(y_test_tfidf, predictions1)
print(confusion)

# print classification report

print(classification_report(y_test_tfidf, predictions1))
print("Accuracy : ",accuracy_score(y_test_tfidf, predictions1))
print("Recall: ",recall_score(y_test_tfidf, predictions1))

"""The Recall is more than 80% for both the classes. Hence we will choose imbalance handled Logistic Regression model"""

# Saving the model as it will be used later while deploying
import pickle

# Save to file in the current working directory
pkl_filename = "classification_pickle.pkl"
with open(pkl_filename, 'wb') as file:
    pickle.dump(lr1, file)

# Load from file
with open(pkl_filename, 'rb') as file:
    pickled_model = pickle.load(file)

"""- **2. Random Forest Classifier**"""

from sklearn.ensemble import RandomForestClassifier

clf=RandomForestClassifier(n_estimators=300)

clf.fit(X_train_res,y_train_res)

y_pred_RandomForest=clf.predict(X_test_tfidf)

confusion = confusion_matrix(y_test_tfidf, y_pred_RandomForest)
print(confusion)

print(classification_report(y_test_tfidf, y_pred_RandomForest))

"""- Although the accuracy is high but Recall for minority class is too low
- Hyperparamter tuning needs to be done here
"""

# Create the parameter grid based on the results of random search 
from sklearn.model_selection import GridSearchCV
param_grid = {
    'max_depth': [10,20,30],
    'min_samples_leaf': [100,125,150,175],
    'min_samples_split': [200,250,300],
    'n_estimators': [250,350,500], 
    'max_features': [10,15]
}

rf = RandomForestClassifier()
grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, cv = 3, scoring="recall", n_jobs = -1,verbose = 1)

# Fit the grid search
grid_search.fit(X_train_res,y_train_res)

# printing the optimal accuracy 
print('We can get recall of',grid_search.best_score_,'using',grid_search.best_params_)

from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(bootstrap=True,
                             max_depth=30,
                             min_samples_leaf=150, 
                             min_samples_split=300,
                             max_features=10,
                             n_estimators=500)

# fit
rfc.fit(X_train_res,y_train_res)

predictions_hpt_RF = rfc.predict(X_test_tfidf)
# Confusion matrix 
confusion = confusion_matrix(y_test_tfidf, predictions_hpt_RF)
print(confusion)

# print classification report
print(classification_report(y_test_tfidf, predictions_hpt_RF))
print("Accuracy : ",accuracy_score(y_test_tfidf, predictions_hpt_RF))
print("Recall: ",recall_score(y_test_tfidf, predictions_hpt_RF))

"""The minority class Recall score has improved but the model is not as good as the class imbalance handled Logistic Regression model

**4. Naive Bayes Classifier**
"""

from sklearn.naive_bayes import MultinomialNB

MNB=MultinomialNB()
MNB.fit(X_train_res,y_train_res)

predicted_MNB=MNB.predict(X_test_tfidf)

confusion = confusion_matrix(y_test_tfidf, predicted_MNB)
print(confusion)

# print classification report
print(classification_report(y_test_tfidf, predicted_MNB))

print("Accuracy : ",accuracy_score(y_test_tfidf, predicted_MNB))
print("Recall: ",recall_score(y_test_tfidf, predicted_MNB))

"""- After building  different ML models to predict the sentiment based on review text and title, the best model is class imbalance handled Logistic Regression model since the minority class Recall score is highest for this model (82%) 
- choosing class imbalance handled Logistic Regression model for all the future predictions and model deployment

### 2. Build a recommendation system

- User based recommendation
-  user based prediction and evaluation
- item based recommendation
- item based prediction & evaluation
"""

for_user_based_reco= for_sentiment_analysis[for_sentiment_analysis['reviews_username'].isnull()== False]
for_user_based_reco.reset_index(drop=True)

train, test = train_test_split(for_user_based_reco, test_size=0.30, random_state=31)

# Pivot the train ratings' dataset into matrix format in which columns are products and the rows are user IDs.
df_pivot = train.pivot_table(
    index='reviews_username',
    columns='id',
    values='reviews_rating'
).fillna(0)
print(df_pivot.shape)
df_pivot.head(5)

"""Create dummy train and dummy test dataset
- Dummy train will be used later for prediction of the products which have not been rated by the user. To ignore the products rated by the user, we will mark it as 0 during prediction. The products not rated by user is marked as 1 for prediction in dummy train dataset. 

- Dummy test will be used for evaluation. To evaluate, we will only make prediction on the products rated by the user. So, this is marked as 1. This is just opposite of dummy_train.
"""

# Copy the train dataset into dummy_train
dummy_train = train.copy()

dummy_train.head()

# The products not rated by user is marked as 1 for prediction. 
dummy_train['reviews_rating'] = dummy_train['reviews_rating'].apply(lambda x: 0 if x>=1 else 1)

# Convert the dummy train dataset into matrix format.
dummy_train = dummy_train.pivot_table(
    index='reviews_username',
    columns='id',
    values='reviews_rating'
).fillna(1)

dummy_train.head()

"""**Cosine Similarity**

Cosine Similarity is a measurement that quantifies the similarity between two vectors
"""

from sklearn.metrics.pairwise import pairwise_distances

# Creating the User Similarity Matrix using pairwise_distance function.
user_correlation = 1 - pairwise_distances(df_pivot, metric='cosine')
user_correlation[np.isnan(user_correlation)] = 0
print(user_correlation)

user_correlation.shape

"""**Prediction user user**

prediction for the users who are positively related with other users, and not with the users who are negatively related as we are interested in the users who are more similar to the current users. So, ignoring the correlation for values less than 0.
"""

user_correlation[user_correlation<0]=0
user_correlation

"""Rating predicted by the user (for products rated as well as not rated) is the weighted sum of correlation with the product rating (as present in the rating dataset). """

user_predicted_ratings = np.dot(user_correlation, df_pivot.fillna(0))
user_predicted_ratings

user_predicted_ratings.shape

"""Since we are interested only in the products not rated by the user, we will ignore the products rated by the user by making it zero. """

user_final_rating = np.multiply(user_predicted_ratings,dummy_train)
user_final_rating.head()

"""Finding top 5 recommendation for a user

"""

# Take the user name as input.
user_input = str(input("Enter your user name"))
print(user_input)

d = user_final_rating.loc[user_input].sort_values(ascending=False)[0:5]
d

mapping=for_user_based_reco[['id','name']]
mapping = pd.DataFrame.drop_duplicates(mapping)
mapping.head()

d = pd.merge(d,mapping, left_on='id', right_on='id', how = 'left')
d

"""**Evaluation  User-user recommendation system**

Evaluation will we same as you have seen above for the prediction.
"""

# Find out the common users of test and train dataset.
common = test[test.reviews_username.isin(train.reviews_username)]

common.head()

# convert into the user-product matrix.
common_user_based_matrix = common.pivot_table(index='reviews_username',columns='id',values='reviews_rating')
common_user_based_matrix.head()

# Convert the user_correlation matrix into dataframe.
user_correlation_df = pd.DataFrame(user_correlation)
user_correlation_df.head()

user_correlation_df['reviews_username'] = df_pivot.index

user_correlation_df.set_index('reviews_username',inplace=True)
user_correlation_df.head()

common.head(1)

list_name = common.reviews_username.tolist()

user_correlation_df.columns = df_pivot.index.tolist()

user_correlation_df_1 =  user_correlation_df[user_correlation_df.index.isin(list_name)]

user_correlation_df_1.shape

user_correlation_df_2 = user_correlation_df_1.T[user_correlation_df_1.T.index.isin(list_name)]

user_correlation_df_3 = user_correlation_df_2.T

print(user_correlation_df_3.shape)
user_correlation_df_3.head()

user_correlation_df_3[user_correlation_df_3<0]=0

common_user_predicted_ratings = np.dot(user_correlation_df_3, common_user_based_matrix.fillna(0))
common_user_predicted_ratings

dummy_test = common.copy()

dummy_test['reviews_rating'] = dummy_test['reviews_rating'].apply(lambda x: 1 if x>=1 else 0)

dummy_test = dummy_test.pivot_table(index='reviews_username',columns='id',values='reviews_rating').fillna(0)
print(dummy_test.shape)

print(common_user_based_matrix.shape)
common_user_based_matrix.head()

print(dummy_test.shape)
dummy_test.head()

common_user_predicted_ratings = np.multiply(common_user_predicted_ratings,dummy_test)
common_user_predicted_ratings.head()

"""Calculating the RMSE for only the products rated by user. For RMSE, normalising the rating to (1,5) range


"""

from sklearn.preprocessing import MinMaxScaler
from numpy import *

X  = common_user_predicted_ratings.copy() 
X = X[X>0]

scaler = MinMaxScaler(feature_range=(1, 5))
print(scaler.fit(X))
y = (scaler.transform(X))
print(y)

common_ = common.pivot_table(index='reviews_username',columns='id',values='reviews_rating')

# Finding total non-NaN value
total_non_nan = np.count_nonzero(~np.isnan(y))

rmse = (sum(sum((common_ - y )**2))/total_non_nan)**0.5
print(rmse)

"""RMSE for User based recommendation system is ~ 2.1

**Item based recommendation system**

### taking the transpose of the rating matrix to normalize the rating around the mean for differnt product.
"""

df_pivot = train.pivot_table(
   index='reviews_username',
    columns='id',
    values='reviews_rating'
).T

df_pivot.head()

# finding the cosine similarity using pairwise distances approach

from sklearn.metrics.pairwise import pairwise_distances

# Item Similarity Matrix
item_correlation = 1 - pairwise_distances(df_pivot.fillna(0), metric='cosine')
item_correlation[np.isnan(item_correlation)] = 0
print(item_correlation)

item_correlation.shape

"""Filtering the correlation only for which the value is greater than 0. (Positively correlated)"""

item_correlation[item_correlation<0]=0
item_correlation

"""**Prediction item-item**"""

item_predicted_ratings = np.dot((df_pivot.fillna(0).T),item_correlation)
item_predicted_ratings

print(item_predicted_ratings.shape)
print(dummy_train.shape)

"""Filtering the rating only for products not rated by the user for recommendation"""

item_final_rating = np.multiply(item_predicted_ratings,dummy_train)
item_final_rating.head()

"""Finding the top 5 recommendation for the user"""

# Take the user ID as input
user_input = str(input("Enter your user name"))
print(user_input)

# Recommending the Top 5 products to the user.
d = item_final_rating.loc[user_input].sort_values(ascending=False)[0:5]
d

mapping= for_user_based_reco[['id','name']]
mapping = pd.DataFrame.drop_duplicates(mapping)
mapping.head()

d = pd.merge(d,mapping, left_on='id', right_on='id', how = 'left')
d

"""**Evaluation item -item**

# evaluation will be same as you have seen above for prediciton
"""

test.columns

common = test[test.id.isin(train.id	)]
print(common.shape)
common.head()

common_item_based_matrix = common.pivot_table(index='reviews_username', columns='id', values='reviews_rating').T

common_item_based_matrix.shape

item_correlation_df = pd.DataFrame(item_correlation)
item_correlation_df.head(5)

item_correlation_df['id'] = df_pivot.index
item_correlation_df.set_index('id',inplace=True)
item_correlation_df.head()

list_name = common.id.tolist()

item_correlation_df.columns = df_pivot.index.tolist()

item_correlation_df_1 =  item_correlation_df[item_correlation_df.index.isin(list_name)]

item_correlation_df_2 = item_correlation_df_1.T[item_correlation_df_1.T.index.isin(list_name)]

item_correlation_df_3 = item_correlation_df_2.T

item_correlation_df_3.head()

item_correlation_df_3[item_correlation_df_3<0]=0
common_item_predicted_ratings = np.dot(item_correlation_df_3, common_item_based_matrix.fillna(0))
print(common_item_predicted_ratings.shape)
common_item_predicted_ratings

"""Dummy test will be used for evaluation. To evaluate, we will only make prediction on the products rated by the user. So, this is marked as 1. This is just opposite of dummy_train"""

dummy_test = common.copy()
dummy_test['reviews_rating'] = dummy_test['reviews_rating'].apply(lambda x: 1 if x>=1 else 0)
dummy_test = dummy_test.pivot_table(index='reviews_username', columns='id', values='reviews_rating').T.fillna(0)
common_item_predicted_ratings = np.multiply(common_item_predicted_ratings,dummy_test)

# The products not rated is marked as 0 for evaluation. And make the item- item matrix representaion.

common_ = common.pivot_table(index='reviews_username', columns='id', values='reviews_rating').T

from sklearn.preprocessing import MinMaxScaler
from numpy import *

X  = common_item_predicted_ratings.copy() 
X = X[X>0]

scaler = MinMaxScaler(feature_range=(1, 5))
print(scaler.fit(X))
y = (scaler.transform(X))

print(y)

# Finding total non-NaN value
total_non_nan = np.count_nonzero(~np.isnan(y))

rmse = (sum(sum((common_ - y )**2))/total_non_nan)**0.5
print(rmse)

"""RMSE for User-based recommendation system is lesser as compared to Item-based recommendation system, hence we will select User-based recommendation system

Finding top 20 recommendations for a selected user using User-based recommendation system
"""

# Take the user ID as input
user_input = str(input("Enter your user name"))
print(user_input)

"""** 20 Recommendations**"""

recommendations = user_final_rating.loc[user_input].sort_values(ascending=False)[0:20]
mapping= for_user_based_reco[['id','name']]
mapping = pd.DataFrame.drop_duplicates(mapping)
recommendations = pd.merge(recommendations,mapping, left_on='id', right_on='id', how = 'left')
recommendations

import pickle

user_final_rating.to_pickle("user_final_rating_pickle.pkl")
pickled_user_final_rating = pd.read_pickle("user_final_rating_pickle.pkl")
pickled_user_final_rating

# Save to file in the current working directory

mapping.to_pickle("mapping_pickle.pkl")
pickled_mapping = pd.read_pickle("mapping_pickle.pkl")
pickled_mapping

for_sentiment_analysis.to_pickle("data_pickle.pkl")
pickled_reviews_data = pd.read_pickle("data_pickle.pkl")
pickled_reviews_data

"""### 3. Improving the recommendations using the sentiment analysis model

Fine-Tuning the recommendation system and recommending top 5 products to the user based on highest percentage of positive sentiments using Sentiment Analysis model developed earlier
"""

# Predicting sentiment for the recommended products using the Logistic Regression model developed earlier

improved_recommendations= pd.merge(recommendations,pickled_reviews_data[['id','reviews_clean']], left_on='id', right_on='id', how = 'left')
test_data_for_user = pickled_tfidf_vectorizer.transform(improved_recommendations['reviews_clean'])
sentiment_prediction_for_user= pickled_model.predict(test_data_for_user)
sentiment_prediction_for_user = pd.DataFrame(sentiment_prediction_for_user, columns=['Predicted_Sentiment'])
improved_recommendations= pd.concat([improved_recommendations, sentiment_prediction_for_user], axis=1)

# For 20 recommended products, calculating the percentage of positive sentiments 

a=improved_recommendations.groupby('id')
b=pd.DataFrame(a['Predicted_Sentiment'].count()).reset_index()
b.columns = ['id', 'Total_reviews']
c=pd.DataFrame(a['Predicted_Sentiment'].sum()).reset_index()
c.columns = ['id', 'Total_predicted_positive_reviews']
improved_recommendations_final=pd.merge( b, c, left_on='id', right_on='id', how='left')
improved_recommendations_final['Positive_sentiment_rate'] = improved_recommendations_final['Total_predicted_positive_reviews'].div(improved_recommendations_final['Total_reviews']).replace(np.inf, 0)
improved_recommendations_final= improved_recommendations_final.sort_values(by=['Positive_sentiment_rate'], ascending=False )
improved_recommendations_final=pd.merge(improved_recommendations_final, pickled_mapping, left_on='id', right_on='id', how='left')

# Filtering out the top 5 products with the highest percentage of positive review
improved_recommendations_final.head(5)