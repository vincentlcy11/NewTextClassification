from time import strftime, localtime

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import f1_score

train_df = pd.read_csv('./data/train_set.csv', sep='\t', nrows=None)
test_df = pd.read_csv('./data/test_a.csv', sep='\t', nrows=None)
print("数据集加载完毕" + strftime("%Y-%m-%d %H:%M:%S", localtime()))

# 文本特征提取
vectorizer = CountVectorizer(max_features=4000)
train_text = vectorizer.fit_transform(train_df['text'])
test_text = vectorizer.fit_transform(test_df['text'])
print("文本特征转换完毕" + strftime("%Y-%m-%d %H:%M:%S", localtime()))

# 采用岭回归训练
clf = RidgeClassifier()
clf.fit(train_text[:150000], train_df['label'].values[:150000])
print("岭回归训练完毕" + strftime("%Y-%m-%d %H:%M:%S", localtime()))

# 预测并计算准确度
val_pred = clf.predict(train_text[150000:])
score = f1_score(train_df['label'].values[150000:], val_pred, average='macro')
print('f1值为' + str(score))

pred = pd.DataFrame()
pred['label'] = clf.predict(test_text)
pred.to_csv('submit1.0.csv', index=None)

