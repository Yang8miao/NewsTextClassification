import pandas as pd
import fasttext
import time
from pandas import DataFrame as df

start_time = time.time()

train_df = pd.read_csv('../data/train_set.csv', sep='\t')
print('read_csv Done!')

train_df['label_ft'] = '__label__'+train_df['label'].astype(str)
train_df[['label_ft','text']].to_csv(
    '../data/fastText_last_train_set.csv',index=False,header=False,sep='\t')

model = fasttext.train_supervised(
    '../data/fastText_last_train_set.csv',lr=1,wordNgrams=2,minCount=1,epoch=100,loss='hs')

#输出测试集结果
test_df = pd.read_csv('../data/test_a.csv', sep='\t')

test_pred = [model.predict(x)[0][0].split('__')[-1] for x in test_df['text']]
output_df = df(test_pred,columns=['label'])
output_df.to_csv('../output/FastText_test_a_output.csv',sep='\t',index=False)

end_time = time.time()

print('cost time = ',end_time-start_time)
