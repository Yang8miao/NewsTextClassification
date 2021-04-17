import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from sklearn import svm
import time

start_time = time.time()

NROWS = 200000
train_rate = 0.8
train_sample = int(NROWS*train_rate)

train_df = pd.read_csv('../data/train_set.csv', sep='\t', nrows=NROWS)
print('read_csv Done!')

vectorizer = TfidfVectorizer(ngram_range=(1,3),max_features=3000)
train_test = vectorizer.fit_transform(train_df['text'])

clf = svm.SVC(kernel='linear')
clf.fit(train_test[:train_sample],train_df['label'].values[:train_sample])

value_pred = clf.predict(train_test[train_sample:])
print('f1_score = ',f1_score(train_df['label'].values[train_sample:],value_pred,average='macro'))

end_time = time.time()

print('cost time = ',end_time-start_time)