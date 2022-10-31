import numpy as np
from torch.utils.data.dataset import Dataset
from KnowledgeTracing.Constant import Constants as C
import torch


class DKTDataSet(Dataset):

    def __init__(self, ques, skill, ans):
        self.ques = ques
        self.skill = skill
        self.ans = ans
        self.num_ques = C.NUM_Q
        self.num_skill = C.NUM_S

    def __len__(self):
        return len(self.ques)  # 返回dataset的长度

    def __getitem__(self, index):
        questions = self.ques[index]
        skill = self.skill[index]
        answers = self.ans[index]
        onehot = self.onehot(questions, skill, answers)
        return onehot

    def onehot(self, ques, skill, answers):
        result = torch.zeros(C.MAX_STEP, 3 + C.MAX_STEP).cuda()
        for i in range(C.MAX_STEP):
                if answers[i] == 1:
                    result[i][0] = ques[i]
                    result[i][1] = skill[i]
                    result[i][2] = 1
                elif answers[i] == 0:
                    result[i][0] = ques[i]
                    result[i][1] = skill[i]
                    result[i][2] = 0
                for j in range(i+1):
                        if skill[j] == skill[i]:
                            result[i][3 + j] = 1
        c = result.cpu().numpy()
        return result  # [50,q_s_a_50]

        # result = torch.zeros(C.MAX_STEP, 2 * self.num_ques + 2 * self.num_skill + C.MAX_STEP).cuda()
        # for i in range(C.MAX_STEP):
        #     if answers[i] > 0:
        #         result[i][ques[i] - 1] = 1
        #         result[i][2 * self.num_ques + skill[i] - 1] = 1
        #     elif answers[i] == 0:
        #         result[i][ques[i] + self.num_ques - 1] = 1
        #         result[i][2 * self.num_ques + skill[i] + self.num_skill - 1] = 1
        #     for j in range(i+1):
        #             if skill[j] == skill[i]:
        #                 result[i][2 * self.num_ques + 2 * self.num_skill + j] = 1


        # a = result[:,2 * self.num_ques + 2 * self.num_skill:].cpu().numpy()
        # b = result[:,2 * self.num_ques:2 * self.num_ques + 2 * self.num_skill].cpu().numpy()

