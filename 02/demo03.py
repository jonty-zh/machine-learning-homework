# -*- encoding: utf-8 -*-
'''
@Author: 张云涛 S322517454
@Time: 2022/10/04
@File: 
'''
import numpy as np
import pandas as pd
import collections

def load_data(data_path):
    # data_path = './watermelon3_0.csv'
    df = pd.read_csv(data_path)
    # 删除属性
    del df['编号']
    del df['密度']
    del df['含糖率']
    data = df.values
    # print(data.shape, type(data))
    return data 

class BayesClassifier:
    def __init__(self, data):
        self.data = data
        self.data_cnt = collections.Counter(self.data[:, -1])

    def cal_px(self):
        data = self.data
        p_x = dict()
        all_data = np.array(data)
        for k in range(all_data.shape[1]):
            p_x[k] = dict()
            tags = list(set(all_data[:, k]))
            d = all_data[:, k]
            tag_cnt = collections.Counter(d)
            for tag in tags:
                # 平滑处理
                p_x[k][tag] = (tag_cnt[tag]+1)/float(len(d)+len(tags))
        print("P(X)\t: ", p_x)
        self.p_x = p_x
        return p_x

    def cal_py(self):
        data = self.data
        data_cnt = self.data_cnt
        labels = data[:, -1]
        # 计算P(Y)
        p_y = dict.fromkeys(data_cnt.keys())
        for tag in p_y.keys():
            p_y[tag] = data_cnt[tag]/len(labels)
        print("P(Y)\t: ", p_y)
        self.p_y = p_y
        return p_y
        
    def cal_pxy(self):
        # 计算P(Xi|Y)
        data = self.data
        data_cnt = self.data_cnt
        p_xy = dict.fromkeys(data_cnt.keys())
        for tag in p_xy.keys():    
            p_xy[tag] = dict()
        for label in data_cnt.keys():   # 分别计算个类别情况下的条件概率
            sub_data = data[data[:, -1] == label]
            sub_data = np.array(sub_data)
            for k in range(sub_data.shape[1]):
                p_xy[label][k] = dict()
                d = sub_data[:, k]
                tags = list(set(data[:, k]))    # sub_data 避免某些类别不存在
                tag_cnt = collections.Counter(d)
                for tag in tags:
                    # 平滑处理
                    p_xy[label][k][tag] = (tag_cnt[tag]+1)/float(len(d) + len(tags))
        print("P(Xi|Y)\t: ", p_xy)
        self.p_xy = p_xy
        return p_xy

    def cal_pyx(self, test_data):
        # test_data = ['浅白', '稍蜷', '浊响', '稍糊', '凹陷', '硬滑']
        tags = list(self.data_cnt.keys())
        print(tags)
        p_xy = self.cal_pxy()
        p_y = self.cal_py()
        p_x = self.cal_px()

        p_yx = dict.fromkeys(self.data_cnt.keys())
        for j, label in enumerate(tags):
            p_yx[label] = np.math.log(p_y[label], 2)
            for k, tag in enumerate(test_data):
                # 采用对数运算，避免下溢
                p_yx[label] = p_yx[label] + np.math.log(p_xy[label][k][tag], 2) - np.math.log(p_x[k][tag], 2);
        print("P(Y|Xi)\t: ", p_yx)
        self.p_yx = p_yx
        return p_yx

    def cal_r(self, con_matrix):
        r = dict.fromkeys(con_matrix.keys())
        for p_tag in con_matrix.keys():   # 是否
            r[p_tag] = 0
            for tag in con_matrix[p_tag].keys():
                # print(con_matrix[p_tag][tag], ' x ', self.p_yx[tag], ' + ', con_matrix[p_tag][tag], ' * ', self.p_yx[tag])
                r[p_tag] += con_matrix[p_tag][tag] * self.p_yx[tag]
                print(tag)
        print(r)
        # 按values排序
        res = sorted(r.items(), key=lambda kv:(kv[1], kv[0]))
        print("排序后: ", res)
        print(res[0][0])

    def test_classifier(self, test_data):
        predict_vec = list()
        for sub_test in test_data:
            self.cal_pyx(sub_test)

            pass
        pass
if __name__ == '__main__':
    data_path = './watermelon3_0.csv'
    train_data = load_data(data_path)
    test_data = ['浅白', '稍蜷', '浊响', '稍糊', '凹陷', '硬滑']
    test_data = ['青绿', '蜷缩', '浊响', '清晰', '凹陷', '软粘']
    classifier = BayesClassifier(train_data)
    classifier.cal_pyx(test_data)
    # 混淆矩阵
    con_matrix ={'是':{'是':0, '否':1}, '否':{'是':1, '否':0}}
    classifier.cal_r(con_matrix)

