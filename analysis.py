# -*- coding: utf-8 -*-
# @Time : 2022/8/17 15:19
# @Author : Yumo
# @File : analysis.py
# @Project: cotrainKT
# @Comment :
import itertools
import numpy as np
import pandas as pd

dic = dict()
for i in range(1676*2):
    dic[i] = 0

path = './KTDataset/assist2009/assist2009_pid_train.csv'
with open(path, 'r', encoding='UTF-8-sig') as train:
    for lens, ques, skill, ans in itertools.zip_longest(*[train] * 4):
        lens = int(lens.strip().strip(','))
        skill = skill.split(',')
        ans = ans.split(',')
        for j in range(lens):
            if ans[j] == '1':
                dic[int(skill[j])-1] += 1
            else:
                dic[1675 + int(skill[j])] += 1
print(dic)
a = sum(dic)

"""
将字典对象保存为excel
"""
# 提取字典中的两列值key是键值，value是cont【key】对应的值
key = list(dic.keys())
value = list(dic.values())

# 利用pandas模块先建立DateFrame类型，然后将两个上面的list存进去
result_excel = pd.DataFrame()
result_excel["skill"] = key
result_excel["num"] = value
# 写入excel
result_excel.to_csv('sednet.csv')
