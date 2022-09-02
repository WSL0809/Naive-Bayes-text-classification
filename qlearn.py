# -*- coding: utf-8 -*-
"""


"""

import os
import time
import random
import jieba
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_curve
from model import q_learn
import sklearn.externals
from sklearn.metrics.pairwise import cosine_similarity
import joblib


def shibie(y_predict):
    list = []
    
    list.append('文本分类识别为：财经的概率'+str(y_predict[0][0]))
    list.append('文本分类识别为：科技的概率'+str(y_predict[0][1]))
    list.append('文本分类识别为：汽车的概率'+str(y_predict[0][2]))
    list.append('文本分类识别为：房产的概率'+str(y_predict[0][3]))  
    list.append('文本分类识别为：体育的概率'+str(y_predict[0][4]))
    list.append('文本分类识别为：娱乐的概率'+str(y_predict[0][5]))
    list.append('文本分类识别为：其他的概率'+str(y_predict[0][6]))
    return list
#读取停用词
def make_words_set(words_file):
    words_set = set()
    with open(words_file, 'r', encoding='utf-8') as fp:
        for line in fp.readlines():
            word = line.strip()
            if len(word)>0 and word not in words_set: # 去重
                words_set.add(word)
    return words_set
#停用词写入txt
def words_set_write(words_file,words_set):
    with open(words_file, 'w',encoding='utf-8') as fp:
        for word in words_set:
            fp.write(word+'\n')
#读入文件，分词后写入中间表
def read_write(oldfile,newfile):
    with open(oldfile,'r',encoding='utf-8') as fp:   
        wd = fp.read()
        wd = wd.replace('腾讯科技', '')
        wd = wd.replace('腾讯财经', '')
        wd = wd.replace('腾讯体育', '')
        wd = wd.replace('腾讯汽车', '')
        wd = wd.replace('腾讯娱乐', '')
        wd = wd.replace('腾讯房产', '')
        wd = wd.replace('人民网', '')
        wd = wd.replace('新华网', '')
        wd = wd.replace('中新网', '')
        wd_cut = jieba.cut(wd)
        wd_result = ' '.join(wd_cut)
        with open(newfile, 'w',encoding='utf-8') as fw:
            fw.write(wd_result)
        fw.close()
    fp.close() 

def process(result):
    result = result.replace('腾讯科技', '')
    result = result.replace('腾讯财经', '')
    result = result.replace('腾讯体育', '')
    result = result.replace('腾讯汽车', '')
    result = result.replace('腾讯娱乐', '')
    result = result.replace('腾讯房产', '')
    result = result.replace('人民网', '')
    result = result.replace('新华网', '')
    result = result.replace('中新网', '')
    wd_cut = jieba.cut(result)
    wd_result = ' '.join(wd_cut)
    tests = []
    tests.append(wd_result)
    return tests

def main(result):

    if not (os.path.exists("./model.pkl") and os.path.exists("./vector.pkl")):
        # 分词后写入中间表--训练集
        oldpath = r'.\training'
        newpath = r'.\mid\train'
        files = os.listdir(oldpath)
        for f in files:
            oldfile = os.path.join(oldpath, f)
            newfile = os.path.join(newpath, f)
            read_write(oldfile, newfile)
        # 读取中间表数据--训练集
        trains = []
        y_train = []
        files = os.listdir(newpath)
        for f in files:
            newfile = os.path.join(newpath, f)
            with open(newfile, 'r', encoding='utf-8') as fp:
                result = fp.read()
                trains.append(result)
                y_train.append(f[0].split('_')[0])
            fp.close()
        oldpath = r'.\test'
        newpath = r'.\mid\test'
        files = os.listdir(oldpath)
        for f in files:
            oldfile = os.path.join(oldpath, f)
            newfile = os.path.join(newpath, f)
            read_write(oldfile, newfile)
        tests = []
        y_test = []
        files = os.listdir(newpath)
        for f in files:
            newfile = os.path.join(newpath, f)
            with open(newfile, 'r', encoding='utf-8') as fp:
                result = fp.read()
                tests.append(result)
                y_test.append(f[0].split('_')[0])
            fp.close()
        stop_words = make_words_set(r'.\stop_words.txt')
        stop_word = list(stop_words)
        
        vector = TfidfVectorizer(stop_words=stop_word)
        x_train = vector.fit_transform(trains)
        

        print("---------------x_train--------------")
        print(x_train)

        joblib.dump(vector, "./vector.pkl")
        clf = q_learn(x_train, y_train)
        joblib.dump(clf, "./model.pkl")
        print("训练完毕，模型已经保存")
    else:
        tests = process(result)
        clf = joblib.load("./model.pkl")
        vector = joblib.load("./vector.pkl")
        

        shibie(clf.predict_proba(vector.transform(tests)[0]))


