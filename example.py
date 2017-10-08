'''
Created on 2017年10月4日

@author: Mi
'''
import csv
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

#get data and train data 从csv获取数据并且训练
def get_data(filepath):
    data_set = []
    vector = []
    with open(filepath) as f:
        f.readline()
        f_csv = csv.reader(f)
        for line in f_csv:
            #吧独立的词合并成一个字符串
            data = ' '.join(line[0:-1])
            #加入训练集
            data_set.append(data)
            #生成Y向量
            car_vector = line[-1]
            if car_vector == 'yes':
                vector.append(1)
            elif car_vector == 'no':
                vector.append(0)
            else :
                break
    print('===========start training================')
    #词频统计 向量化
    count_vect = CountVectorizer(stop_words="english",decode_error='ignore')
    X_train_counts = count_vect.fit_transform(data_set)
#     print(X_train_counts.shape)
    #tf-idf方法加权
    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
#     print(X_train_tfidf.shape)
#     X = np.array(X_train_tfidf)
#     print(X.shape)
#     y = np.array(vector)
#     print(y.shape)
    
    X = X_train_tfidf.toarray()
    y = vector
    #选用高斯分布
    clf = GaussianNB()
    #训练
    clf.fit(X, y)
    return clf,count_vect,tfidf_transformer

#test 测试
def test(filepath):
    #使用训练好的训练器
    clf,count_vect,tfidf_transformer = get_data(filepath)
    print('======Training terminated, now start predict==================')
    #输入新的情况
    doc_to_pridict = [['overcast', 'cool', 'normal','strong']]
    print('==predict the possibility of the condition ==')
    print('outlook    temp    humidity    wind     class')
    print(doc_to_pridict)

    #转化为字符串处理
    docs_new = [' '.join(doc_to_pridict[0][:])]
    #词频特征提取
    X_new_counts = count_vect.transform(docs_new)
    #生成tf-idf向量
    X_new_tfidf = tfidf_transformer.transform(X_new_counts)
    #预测
    predicted = clf.predict(X_new_tfidf.toarray())
    #预测结果处理
    if predicted == [0]:
        result = 'no'
    elif predicted == [1]:
        result = 'yes'
    print('========The answer is ============')
    print(result)

            
if __name__ == '__main__':
    test(r'nbtest.csv')
