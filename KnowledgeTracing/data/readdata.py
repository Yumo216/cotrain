
import numpy as np
from KnowledgeTracing.data.DKTDataSet import DKTDataSet
import itertools
import tqdm


class DataReader():
    def __init__(self, path, maxstep, numofques):
        self.path = path
        self.maxstep = maxstep
        self.numofques = numofques

    def getTrainData(self):
        trainqus = np.array([])
        trainskil = np.array([])
        trainans = np.array([])
        with open(self.path, 'r', encoding='UTF-8-sig') as train:
            for len, ques ,skill,  ans in tqdm.tqdm(itertools.zip_longest(*[train] * 4), desc='loading train data:    ',
                                               mininterval=2):
                len = int(len.strip().strip(','))
                ques = np.array(ques.strip().strip(',').split(',')).astype(int)
                skill = np.array(skill.strip().strip(',').split(',')).astype(int)
                ans = np.array(ans.strip().strip(',').split(',')).astype(int)
                mod = 0 if len % self.maxstep == 0 else (self.maxstep - len % self.maxstep)
                zero = np.zeros(mod) - 1
                ques = np.append(ques, zero)
                skill = np.append(skill, zero)
                ans = np.append(ans, zero)
                trainqus = np.append(trainqus, ques).astype(int)
                trainskil = np.append(trainskil, skill).astype(int)
                trainans = np.append(trainans, ans).astype(int)
        return trainqus.reshape([-1, self.maxstep]),trainskil.reshape([-1, self.maxstep]), trainans.reshape([-1, self.maxstep])

    '''和上面的一样'''

    def getTestData(self):
        testqus = np.array([])
        testskil = np.array([])
        testans = np.array([])
        with open(self.path, 'r', encoding='UTF-8-sig') as test:
            for len,  ques,skill,  ans in tqdm.tqdm(itertools.zip_longest(*[test] * 4), desc='loading train data:    ',
                                                mininterval=2):
                len = int(len.strip().strip(','))
                ques = np.array(ques.strip().strip(',').split(',')).astype(int)
                skill = np.array(skill.strip().strip(',').split(',')).astype(int)
                ans = np.array(ans.strip().strip(',').split(',')).astype(int)
                mod = 0 if len % self.maxstep == 0 else (self.maxstep - len % self.maxstep)
                zero = np.zeros(mod) - 1
                ques = np.append(ques, zero)
                skill = np.append(skill, zero)
                ans = np.append(ans, zero)
                testqus = np.append(testqus, ques).astype(int)
                testskil = np.append(testskil, skill).astype(int)
                testans = np.append(testans, ans).astype(int)
        return testqus.reshape([-1, self.maxstep]),testskil.reshape([-1, self.maxstep]), testans.reshape([-1, self.maxstep])
