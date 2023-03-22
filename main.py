# Leonardo Daniel Riojas Sánchez A00825968

import pandas as pd
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV

# Preparing data
mylist = []

for chunk in pd.read_csv('IMDB Dataset.csv', delimiter=',', chunksize=20000):
    mylist.append(chunk)
df_review = pd.concat(mylist, axis=0)
del mylist
# print(df_review)

# Imbalanced data
df_positive = df_review[df_review['sentiment'] == 'positive'][:4500]
df_negative = df_review[df_review['sentiment'] == 'negative'][:500]

df_review_imb = pd.concat([df_positive, df_negative])
# print(df_review_imb.value_counts(['sentiment']))

# Handling Imbalanced Data
rus = RandomUnderSampler(random_state=0)
df_review_bal, df_review_bal['sentiment'] = rus.fit_resample(df_review_imb[['review']],
                                                             df_review_imb['sentiment'])

print(df_review_imb.value_counts('sentiment'))
print(df_review_bal.value_counts('sentiment'))

# Splitting data into train and test
train, test = train_test_split(df_review_bal, test_size=0.33, random_state=42)
train_x, train_y = train['review'], train['sentiment']
test_x, test_y = test['review'], test['sentiment']
# print(train_y.value_counts())

# Text Representation (Bag of Words)
# #Count Vectorizer
text = ["I love writing code in Python. I love Python code",
        "I hate writing code in Java. I hate Java code"]

df = pd.DataFrame({'review': ['review1', 'review2'], 'text': text})
cv = CountVectorizer(stop_words='english')
cv_matrix = cv.fit_transform(df['text'])
# df_dtm = pd.DataFrame(cv_matrix.toarray(), index=df['review'].values, columns=cv.get_feature_names_out())
# print(df_dtm)

# #Tfidf
text = ["I love writing code in Python. I love Python code",
        "I hate writing code in Java. I hate Java code"]

df = pd.DataFrame({'review': ['review1', 'review2'], 'text': text})
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['text'])
df_dtm = pd.DataFrame(tfidf_matrix.toarray(), index=df['review'].values, columns=tfidf.get_feature_names_out())
# print(df_dtm)

# Turning our text data into numerical vectors
tfidf = TfidfVectorizer(stop_words='english')
train_x_vector = tfidf.fit_transform(train_x)
# also fit the test_x_vector
test_x_vector = tfidf.transform(test_x)
# print(test_x_vector)

# cv = CountVectorizer(stop_words='english')
# train_x_vector = cv.fit_transform(train_x)
# test_x_vector = cv.transform(test_x)
# print(pd.DataFrame.sparse.from_spmatrix(train_x_vector, index=train_x.index, columns=tfidf.get_feature_names_out()))

# MODEL SELECTION
# #SVM
svc = SVC(kernel='linear')
svc.fit(train_x_vector, train_y)

# svc.predict(train_x_vector[0])

# ##Testing
#print(svc.predict(tfidf.transform(['A good movie'])))
#print(svc.predict(tfidf.transform(['An excellent movie'])))
# print(svc.predict(tfidf.transform(['"I did not like this movie at all I gave this movie away"'])))

# #Decision Tree
dec_tree = DecisionTreeClassifier()
dec_tree.fit(train_x_vector, train_y)

# #Naive Bayes
gnb = GaussianNB()
gnb.fit(train_x_vector.toarray(), train_y)

# Logistic regression
log_reg = LogisticRegression()
log_reg.fit(train_x_vector, train_y)

# Model Evaluation
# Mean Accuracy
print(svc.score(test_x_vector, test_y))
print(dec_tree.score(test_x_vector, test_y))
print(gnb.score(test_x_vector.toarray(), test_y))
print(log_reg.score(test_x_vector, test_y))

# svc.score('Test samples', 'True labels')

# F1 Score
f1_score(test_y, svc.predict(test_x_vector),
         labels=['positive', 'negative'],
         average=None)

# f1_score(y_true, y_pred, average=None)

# Classification report
print(classification_report(test_y,
                            svc.predict(test_x_vector),
                            labels=['positive', 'negative']))
# classification_report(y_true, y_pred)

# #Confusion MatrixConfusion Matrix
conf_mat = confusion_matrix(test_y,
                            svc.predict(test_x_vector),
                            labels=['positive', 'negative'])
print("Confusion Matrix")
print(conf_mat)

# Specificity: the specificity gets the True Negatives / Total
tn = conf_mat[1][1]
fp = conf_mat[0][1]
total_n = tn + fp
specificity = tn / total_n
print("Specificity:", specificity)

# Tuning the Model
# #GridSearchCV
'''
parameters = {'C': [1,4,8,16,32] ,'kernel':['linear', 'rbf']}
svc = SVC()
svc_grid = GridSearchCV(svc,parameters, cv=5,)
#              refit=True, verbose=0)
svc_grid.fit(train_x_vector, train_y)
print(svc_grid.best_params_)
print(svc_grid.best_estimator_)
'''
