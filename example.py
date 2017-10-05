'''
Created on 2017年10月4日

@author: Mi
'''
import csv
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

def get_data(filepath):
    data_set = []
    vector = []
    with open(filepath) as f:
        f.readline()
        f_csv = csv.reader(f)
        for line in f_csv:
            data = ' '.join(line[0:-1])
            data_set.append(data)
            car_vector = line[-1]
            if car_vector == 'yes':
                vector.append(1)
            elif car_vector == 'no':
                vector.append(0)
            else :
                break
    print(data_set)
    print(vector)
    count_vect = CountVectorizer(stop_words="english",decode_error='ignore')
    X_train_counts = count_vect.fit_transform(data_set)
#     print(X_train_counts.shape)
    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
#     print(X_train_tfidf.shape)
#     X = np.array(X_train_tfidf)
#     print(X.shape)
#     y = np.array(vector)
#     print(y.shape)
    X = X_train_tfidf.toarray()
    y = vector
    clf = GaussianNB()
    clf.fit(X, y)
    
    doc_to_pridict = [['overcast', 'cool', 'normal','strong']]
    print('==predict the possibility of the condition ==')
    print('outlook    temp    humidity    wind     class')
    docs_new = [' '.join(doc_to_pridict[0][0:])]
    X_new_counts = count_vect.transform(docs_new)
    X_new_tfidf = tfidf_transformer.transform(X_new_counts)

    predicted = clf.predict(X_new_tfidf.toarray())
    
    if predicted == [0]:
        result = 'no'
    elif predicted == [1]:
        result = 'yes'
    print(result)
            
if __name__ == '__main__':
    get_data(r'nbtest.csv')