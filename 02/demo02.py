#encoding:utf-8

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

    label1 = data[:, 0]
    cnt1 = collections.Counter(label1)
    print(cnt1)

    data1 = np.array(data)
    classify1 = {}
    for i in range(data1.shape[1]):
        classify1[i] = dict()
        tags = list(set(data[:, i]))
        d = data1[:, i]
        tag_count = collections.Counter(data1[:, i])
        for tag in tags:
            # 平滑处理
            classify1[i][tag] = (tag_count[tag]+1)/float(len(d)+len(tags)) 
    print("!!!!!!!!!!!!!!!!!!", classify1)

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

    for i in range(len(NBClassify.keys())):
        print(i, NBClassify.keys())
    # test = ['浅白', '稍蜷', '浊响', '稍糊', '凹陷', '硬滑']
    test = ['青绿','蜷缩','浊响','清晰','凹陷','硬滑']
    tags = list(NBClassify.keys())

    # 创建混淆矩阵
    con_matrix = np.array([[0.1, 0.2], [0.3, 0.4]])
    R = dict.fromkeys(data_count2.keys())
    print("R ", R)
    print("ddddddddddd")
    for j, label in enumerate(tags):
        R[label] = 0
        sum = np.math.log(BaseClassify[label], 2)
        for k, tag in enumerate(test):
            # 对数似然避免下溢
            print(j, k, tag, BaseClassify[label], NBClassify[label][k][tag], classify1[k][tag])
            # sum += BaseClassify[label] * NBClassify[label][k][tag] / classify1[k][tag]
            sum = sum + np.math.log(NBClassify[label][k][tag], 2) - np.math.log(classify1[k][tag], 2)
        R[label] += 2 * sum 
        print("--------result : ", sum)
        print("R: ", R[label], R)
    
    # 求分母各概率值

def test_data(data):
    predict_vec = list()
    for sample in data:
        for k, tag in enumerate(sample):
            print(k, tag)
    # for lable in classify1.keys():
    #     for k, tag in enumerate(data):
    #         pass        

def testNB(data, PG, PB, NBClassify):
    predict_vec = list()
    for sample in data:
        pg = np.math.log(PG, 2)
        pb = np.math.log(PB, 2)
        for label in NBClassify.keys():
            for k, tag in enumerate(sample):
                if label == '是':
                    pg += np.math.log(NBClassify[label][k][tag], 2)
                else :
                    pb += np.math.log(NBClassify[label][k][tag], 2)
        if pg > pb:
            predict_vec.append('是')
        else:
            predict_vec.append('否')
    return np.array(predict_vec)

# 计算概率

# 找出超父


if __name__ == '__main__':
    data = load_data()
    ana_data(data)
    test = [['浅白', '稍蜷', '浊响', '稍糊', '凹陷', '硬滑'],]
    for i in test[0]:
        print(i)
    test_data(test)
