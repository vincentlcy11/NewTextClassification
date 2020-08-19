import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold, train_test_split, KFold
from lightgbm import LGBMClassifier
from sklearn.metrics import f1_score
import warnings
from time import strftime, localtime

warnings.filterwarnings('ignore')

print("训练开始" + strftime("%Y-%m-%d %H:%M:%S", localtime()))
train_df = pd.read_csv('./data/train_set.csv', sep='\t', nrows=None)
test_df = pd.read_csv('./data/test_a.csv', sep='\t', nrows=None)
print("数据集加载完毕" + strftime("%Y-%m-%d %H:%M:%S", localtime()))

tfidf = TfidfVectorizer(ngram_range=(1, 3), max_features=10000)
train_tfidf = tfidf.fit_transform(train_df['text'])
test_tfidf = tfidf.fit_transform(test_df['text'])
train_tfidf_df = pd.DataFrame()
test_tfidf_df = pd.DataFrame()
train_tfidf_df['tfidf'] = train_tfidf
test_tfidf_df['tfidf'] = test_tfidf
train_tfidf_df.to_csv('./data/train_tfidf.csv', index=None)
test_tfidf_df.to_csv('./data/train_tfidf.csv', index=None)
print("文本特征转换完毕" + strftime("%Y-%m-%d %H:%M:%S", localtime()))

skf = StratifiedKFold(n_splits=5, random_state=7)
test_pred = np.zeros((test_tfidf.shape[0], 14), dtype=np.float32)
for idx, (train_index, valid_index) in enumerate(skf.split(train_tfidf, train_df['label'].values)):
    x_train_, x_valid_ = train_tfidf[train_index], train_tfidf[valid_index]
    y_train_, y_valid_ = train_df['label'].values[train_index], train_df['label'].values[valid_index]

    clf = LGBMClassifier()
    clf.fit(x_train_, y_train_)
    val_pred = clf.predict(x_valid_)

    print(f1_score(y_valid_, val_pred, average='macro'))
    test_pred += clf.predict_proba(test_tfidf)
    print("交叉检验预测完毕一次" + strftime("%Y-%m-%d %H:%M:%S", localtime()))

pred = pd.DataFrame()
pred['label'] = test_pred.argmax(1)
pred.to_csv('submit2.0.csv', index=None)
print("输出预测值完毕" + strftime("%Y-%m-%d %H:%M:%S", localtime()))
