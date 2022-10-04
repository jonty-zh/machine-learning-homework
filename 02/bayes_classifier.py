# -*- encoding: utf-8 -*-
# @Author: 张云涛 S322517454
# @Date: 2022/10/04
# @File: bayes_classifier.py

import numpy as np
import pandas as pd
import collections

def load_data(data_path):
    df = pd.read_csv(data_path)
    # 删除属性
    del df['编号']
    del df['密度']
    del df['含糖率']
    data = df.values
    return data 

class BayesClassifier:
    """贝叶斯分类器"""
    def __init__(self, data):
        self.data = data
        self.data_cnt = collections.Counter(self.data[:, -1])

    def cal_px(self):
        """计算先验概率P(Xi)"""
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
        # print("P(X)\t: ", p_x)
        return p_x

    def cal_py(self):
        """计算先验概率P(Y)"""
        data = self.data
        data_cnt = self.data_cnt
        labels = data[:, -1]
        p_y = dict.fromkeys(data_cnt.keys())
        for tag in p_y.keys():
            p_y[tag] = data_cnt[tag]/len(labels)
        # print("P(Y)\t: ", p_y)
        return p_y
        
    def cal_pxy(self):
        # 计算先验概率P(Xi|Y)
        data = self.data
        data_cnt = self.data_cnt
        p_xy = dict.fromkeys(data_cnt.keys())
        for tag in p_xy.keys():    
            p_xy[tag] = dict()
        for label in data_cnt.keys():   
            sub_data = data[data[:, -1] == label]
            sub_data = np.array(sub_data)
            for k in range(sub_data.shape[1]):
                p_xy[label][k] = dict()
                d = sub_data[:, k]
                tags = list(set(data[:, k]))   
                tag_cnt = collections.Counter(d)
                for tag in tags:
                    # 平滑处理，避免概率为0的情况
                    p_xy[label][k][tag] = (tag_cnt[tag]+1)/float(len(d) + len(tags))
        # print("P(Xi|Y)\t: ", p_xy)
        return p_xy

    def cal_pyx(self, test_data):
        """计算P(Y|X)"""
        tags = list(self.data_cnt.keys())
        p_xy = self.cal_pxy()
        p_y = self.cal_py()
        p_x = self.cal_px()
        print("先验概率:")
        print("P(X):\t", p_x)
        print("P(Y):\t", p_y)
        print("P(X|Y):\t", p_xy)

        p_yx = dict.fromkeys(self.data_cnt.keys())
        for j, label in enumerate(tags):
            p_yx[label] = np.math.log(p_y[label], 2)
            for k, tag in enumerate(test_data):
                # 采用对数运算，避免下溢
                p_yx[label] = p_yx[label] + np.math.log(p_xy[label][k][tag], 2) - np.math.log(p_x[k][tag], 2);
        # print("P(Y|Xi)\t: ", p_yx)
        print("后验概率:\nP(Y|X):\t", p_yx)
        return p_yx

    def cal_r(self, con_matrix, test_item):
        '''计算条件风险'''
        p_yx = self.cal_pyx(test_item)
        r = dict.fromkeys(con_matrix.keys())
        for p_tag in con_matrix.keys():   # 是否
            r[p_tag] = 0
            for tag in con_matrix[p_tag].keys():
                r[p_tag] += con_matrix[p_tag][tag] * p_yx[tag]
        # 对条件风险值进行排序，返回最小值对应种类
        res = sorted(r.items(), key=lambda kv:(kv[1], kv[0]))
        print("条件风险:\nR:\t", res)
        return res[0][0]

    def test_classifier(self, con_matrix, test_data):
        """测试分类器"""
        predict_vec = list()
        for i, item in enumerate(test_data):
            print("\n", 20*'-', " 第{0}次测试 ".format(i+1), 20*'-')
            print("混淆矩阵:\t", con_matrix)
            print("测试数据:\t", item)
            res = self.cal_r(con_matrix, item)
            predict_vec.append(res)
        print("\n测试结果:\t", predict_vec)
        return predict_vec


if __name__ == '__main__':
    data_path = './watermelon.csv'
    train_data = load_data(data_path)
    test_data = [['青绿', '蜷缩', '浊响', '清晰', '凹陷', '软粘'], ['浅白', '稍蜷', '浊响', '稍糊', '凹陷', '硬滑']]
    classifier = BayesClassifier(train_data)
    # 混淆矩阵
    con_matrix ={'是':{'是':0, '否':1}, '否':{'是':1, '否':0}}
    # classifier.cal_r(con_matrix, test_data)
    res = classifier.test_classifier(con_matrix, test_data)
    print("res: ", res)
