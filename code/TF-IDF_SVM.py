import pandas as pd
from pandas import DataFrame as df
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
import time

start_time = time.time()

train_df = pd.read_csv('../data/train_set.csv', sep='\t')

print('read_csv Done!')

vectorizer = TfidfVectorizer(ngram_range=(1,3),max_features=3000)
train_test = vectorizer.fit_transform(train_df['text'])

print('TF-IDF Done!')

clf = svm.SVC(kernel='linear')
clf.fit(train_test,train_df['label'])

print('clf.fit Done!')

#输出测试集结果
test_df = pd.read_csv('../data/test_a.csv', sep='\t')
test_test = vectorizer.fit_transform(test_df['text'])
test_pred = clf.predict(test_test)

output_df = df(test_pred,columns=['label'])
output_df.to_csv('../output/test_a_output.csv',sep='\t',index=False)

end_time = time.time()

print('cost time = ',end_time-start_time)