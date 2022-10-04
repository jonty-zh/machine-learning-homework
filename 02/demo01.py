from ctypes import sizeof
import pandas as pd 
import numpy as np 
import collections

# 支持样本属性数量可以任意设定
def load_data(rate=0.9):
    data_path = './watermelon3_0.csv'
    df = pd.read_csv(data_path)
    # 删除属性
    del df['编号']
    del df['密度']
    del df['含糖率']
    data = df.values
    return data 

def trainNB(data):
    labels = data[:, -1]
    PGood = sum([1 for l in labels if l=='是']) / len(labels)
    PBad = 1 - PGood

    NBClassify = {'是': {}, '否': {}}
    for label in NBClassify.keys():
        sub_data = data[data[:, -1] == label]
        sub_data = np.array(sub_data)
        for k in range(sub_data.shape[1]):
            NBClassify[label][k] = dict()
            tags = list(set(data[:, k]))
            d = sub_data[:, k]
            tag_count = collections.Counter(sub_data[:, k])
            for tag in tags:
                # NBClassify[label][k][tag] = (tag_count[tag]+1)/float(len(d)+sizeof(tag)) 
                NBClassify[label][k][tag] = (sum([1 for i in d if i==tag])+1)/len(d)
    print(NBClassify) # 先验概率 
    return PGood, PBad, NBClassify

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


if __name__ == '__main__':
    # test = [['浅白', '稍蜷', '浊响', '稍糊', '凹陷', '硬滑'],]
    confusion_matrix = []

    test = [['青绿', '蜷缩', '浊响', '清晰', '凹陷', '软粘'],]
    data = load_data()
    PG, PB, NBClassify = trainNB(data)
    predict_vec = testNB(test, PG, PB, NBClassify)
    print(test)
    print(predict_vec)

# TODO: 混淆矩阵可任意设定，混淆矩阵
