from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import recall_score, precision_score
from imblearn.over_sampling import SMOTE


import numpy as np
import csv
import json
import argparse

def read_file(file,jsonfile):
    """here we read data from file and creat X and y numpy-arrays"""
    data_X, data_Y=[],[]
    with open(file, 'r') as csvfile:
        reader=csv.reader(csvfile)
        for row in reader:
            #print(row)
            row=row[0].split(' ')
            data_X.append([float(x) for x in row])
    X=np.array(data_X)


    for line in open(jsonfile, 'r'):
        record = json.loads(line)
        data_Y.append(record['overall'])

    Y=np.array(data_Y)
    return X, Y

def preprocess_log_reg(X_data, Y_data):
    """here data are divided in train-test and normalized """


    X_train, X_test, y_train, y_test = train_test_split(
        X_data, Y_data, train_size=0.8, test_size=0.2)

    sm = SMOTE( )
    X_train_res, y_train_res = sm.fit_sample(X_train, y_train)


    return X_train_res, X_test, y_train_res, y_test


def preprocess_svm(X_data, Y_data):
    """here data are divided in train-test and normalized """


    X_train, X_test, y_train, y_test = train_test_split(
        X_data, Y_data, train_size=0.8, test_size=0.2)


    return X_train, X_test, y_train, y_test


def fit_log_reg(X_train, y_train, X_test, y_test):
    mod = LogisticRegression(multi_class='multinomial', solver='sag')
    mod.fit(X_train, y_train)


    score_train = mod.score(X_train, y_train)
    recall_train = recall_score(y_train, mod.predict(X_train), average='weighted')
    precision_train = precision_score(y_train, mod.predict(X_train), average='weighted')

    score_test=mod.score(X_test, y_test)
    recall_test = recall_score(y_test, mod.predict(X_test), average='weighted')
    precision_test = precision_score(y_test, mod.predict(X_test), average='weighted')


    return score_train, recall_train, precision_train, score_test, recall_test, precision_test



def fit_svm(X_train, y_train, X_test, y_test):

    print('svm called now')
    clf = svm.SVC(decision_function_shape='ovo')
    clf.fit(X_train, y_train)
    print('svm model created', clf)

    score_train = clf.score(X_train, y_train)
    print('score train got')
    recall_train = recall_score(y_train, clf.predict(X_train), average='weighted')
    precision_train = precision_score(y_train, clf.predict(X_train), average='weighted')

    score_test = clf.score(X_test, y_test)
    print('score test got')
    recall_test = recall_score(y_test, clf.predict(X_test), average='weighted')
    precision_test = precision_score(y_test, clf.predict(X_test), average='weighted')

    return score_train, recall_train, precision_train, score_test, recall_test, precision_test



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help='give the name of the model: log_reg or svm', default='both')
    args = parser.parse_args()

    path = '/Users/teaxhaferi/Desktop/Tea/'

    X, y = read_file(path + 'vector_results.txt', path + 'reviews.json')

    precision_train_list = []
    recall_train_list = []
    precision_test_list = []
    recall_test_list = []

    if args.model == 'log_reg':


        for i in range(2):



            X_train, X_test, y_train, y_test=preprocess_svm(X, y)
            print('preprocess done')


            score_train, recall_train, precision_train, score_test, recall_test, precision_test = fit_log_reg(X_train,y_train,X_test,y_test)

            precision_train_list.append(precision_train)
            recall_train_list.append(recall_train)
            precision_test_list.append(precision_test)
            recall_test_list.append(recall_test)


            print('for trial number i=',i,'training accuracy, recall, precision are:' ,score_train, recall_train, precision_train)
            print("testing accuracy, recall, precision are:", score_test, recall_test, precision_test)
        mean_precision_train=np.mean(precision_train_list)
        mean_recall_train=np.mean(recall_train_list)
        mean_precision_test=np.mean(precision_test_list)
        mean_recall_test=np.mean(recall_test_list)

        print("All the means are:",mean_precision_train,mean_recall_train,mean_precision_test,mean_recall_test)

    elif args.model=='svm':

        for i in range(2):

            X_train, X_test, y_train, y_test=preprocess_svm(X, y)
            print('preprocess done')


            score_train, recall_train, precision_train, score_test, recall_test, precision_test = fit_svm(X_train,y_train,X_test,y_test)


            precision_train_list.append(precision_train)
            recall_train_list.append(recall_train)
            precision_test_list.append(precision_test)
            recall_test_list.append(recall_test)


            print('for trial number i=',i,'training accuracy, recall, precision are:' ,score_train, recall_train, precision_train)
            print("testing accuracy, recall, precision are:", score_test, recall_test, precision_test)

        mean_precision_train = np.mean(precision_train_list)
        mean_recall_train = np.mean(recall_train_list)
        mean_precision_test = np.mean(precision_test_list)
        mean_recall_test = np.mean(recall_test_list)

        print("All the means are:", mean_precision_train, mean_recall_train, mean_precision_test, mean_recall_test)

    else:
        print("Please give me the model name!")
