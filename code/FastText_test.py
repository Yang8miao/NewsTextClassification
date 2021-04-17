import pandas as pd
from sklearn.metrics import f1_score
import fasttext
import time

start_time = time.time()

NROWS = 200000
train_rate = 0.8
train_sample = int(NROWS*train_rate)

train_df = pd.read_csv('../data/train_set.csv', sep='\t', nrows=NROWS)
print('read_csv Done!')

train_df['label_ft'] = '__label__'+train_df['label'].astype(str)
train_df[['label_ft','text']].iloc[:train_sample].to_csv(
    '../data/fastText_train_set.csv',index=False,header=False,sep='\t')

model = fasttext.train_supervised(
    '../data/fastText_train_set.csv',lr=1,wordNgrams=2,minCount=1,epoch=1,loss='hs')

value_pred = [model.predict(x)[0][0].split('__')[-1] for x in train_df.iloc[train_sample:]['text']]
print(f1_score(train_df['label'].values[train_sample:].astype(str),value_pred,average='macro'))

end_time = time.time()

print('cost time = ',end_time-start_time)
