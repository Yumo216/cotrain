# -*- coding: utf-8 -*-
# @Time : 2022/8/22 11:21
# @Author : Yumo
# @File : balance.py
# @Project: cotrainKT
# @Comment :
import numpy as np
import tqdm
import torch

import itertools
testpath = '../KTDataset/assist2009/assist2009_pid_test'

path = '../../KTDataset/lessnum.csv'
ln = np.array(list(open(path))).astype(int)

dic = dict()
for i in range(1, 204 + 1):
    dic[i] = 0

with open(testpath, 'r', encoding='UTF-8-sig') as train:
    for lens, _, skill, ans in tqdm.tqdm(itertools.zip_longest(*[train] * 4), desc='loading train data:    ',
                                       mininterval=2):
        lens = int(lens.strip().strip(','))
        skill = skill.split(',').astype(int)
        ans = ans.split(',').astype(int)
        for j in range(lens):
            if ans[j] == '1' and dic[int(skill[j])] < ln[int(skill[j])]:
                dic[int(skill[j])] += 1
            elif ans[j] == '0' and dic[int(skill[j])] < ln[int(skill[j])]:
                dic[102 + int(skill[j])] += 1
            else:
                skill[j] = ans[j] = 0

