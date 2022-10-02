#encoding:utf-8
from ctypes import sizeof
from enum import unique
from re import sub
import pandas as pd
import numpy as np 
import collections

# 加载数据
def load_data():
    data_path = './watermelon3_0.csv'
    df = pd.read_csv(data_path)
    # 删除属性
    del df['编号']
    del df['密度']
    del df['含糖率']
    data = df.values
    # print(data.shape, type(data))
    return data 

def cal_prior(data):
    labels = data[:, -1]
    data_count = collections.Counter(labels)
    for label in data_count.keys():
        data_count[label] = data_count[label]/len(labels)
    print(data_count)
    NBClassify = dict.fromkeys(data_count.keys()) 
    for k in NBClassify.keys():
        NBClassify[k] = {}
    for label in data_count.keys():
        # print(label)
        sub_data = data[data[:, -1] == label]
        sub_data = np.array(sub_data)
        print(sub_data)



# 确定种类数量
def ana_data(data):
    labels = data[:, -1]
    # print(type(labels))
    # print(labels)
    # unique, count = np.unique(labels)
    # print(unique)
    # print(count)
    # data_count = dict(zip(unique, count))
    # print(data_count)
    data_count2 = collections.Counter(labels)
    print(data_count2, data_count2.keys(), data_count2.values())
    BaseClassify = dict.fromkeys(data_count2.keys()) # 
    for label in BaseClassify.keys():
        BaseClassify[label] = (data_count2[label]+1)/float(len(labels)+len(BaseClassify.keys()))
    print(BaseClassify)

    NBClassify = dict.fromkeys(data_count2.keys()) # 
    for la in NBClassify.keys():
        NBClassify[la] = {}
    for label in data_count2.keys():
        # print(label)
        sub_data = data[data[:, -1] == label]
        sub_data = np.array(sub_data)
        for k in range(sub_data.shape[1]):
            NBClassify[label][k] = dict()
            tags = list(set(data[:, k]))
            d = sub_data[:, k]
            tag_count = collections.Counter(sub_data[:, k])
            for tag in tags:
                # 平滑处理
                NBClassify[label][k][tag] = (tag_count[tag]+1)/float(len(d)+len(tags)) 
        # print(sub_data)
    # print(NBClassify)
    print(NBClassify['是'])
    # 求分母各概率值

# 计算概率

# 找出超父


if __name__ == '__main__':
    data = load_data()
    ana_data(data)
