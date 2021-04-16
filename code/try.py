import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import time
import re

start_time = time.time()

train_df = pd.read_csv('../data/train_set.csv', sep='\t')

print('read_csv Done!')

train_df['sentence_num']=train_df['text'].apply(lambda x:len(re.split('3750|900|648',x)))
#
# train_df['label'].value_counts().plot(kind='bar')
# plt.show()

print(train_df.describe())


for index in range(14):

    all_lines = ' '.join(list(train_df[train_df['label']==index]['text']))
    word_count = Counter(all_lines.split(' '))

    word_count = sorted(word_count.items(), key=lambda x: x[1], reverse=True)

    print('index = ',index,'word_count:', word_count)


end_time = time.time()

print('cost time = ',end_time-start_time)

