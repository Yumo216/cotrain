import tqdm
import torch
from KnowledgeTracing.Constant import Constants as C
import torch.nn as nn
from sklearn import metrics
import logging

logger = logging.getLogger('main.eval')


def performance(ground_truth, prediction):
    fpr, tpr, thresholds = metrics.roc_curve(ground_truth.detach().cpu().numpy(),
                                             prediction.detach().cpu().numpy())
    auc = metrics.auc(fpr, tpr)
    acc = metrics.accuracy_score(ground_truth.detach().cpu().numpy(), torch.round(prediction).detach().cpu().numpy())
    print('auc: ' + str(auc) + ' acc: ' + str(acc))
    logger.info('\nauc: ' + str(auc) + 'acc: ' + str(acc))
    return auc, acc


class lossFunc(nn.Module):
    def __init__(self, input, hidden, max_step, device):
        super(lossFunc, self).__init__()
        self.crossEntropy = nn.BCELoss()
        self.L2 = nn.MSELoss()
        self.q = C.NUM_Q
        self.s = C.NUM_S
        self.pid = 2 * self.q
        self.input = input
        self.hidden = hidden
        self.sig = nn.Sigmoid()
        self.max_step = max_step
        self.device = device

    def forward(self, p_s, p_q, batch):
        pred_s = self.sig(p_s)  # [64,50,s]
        pred_q = self.sig(p_q)

        qsa = batch[:, :, :3].long()
        loss_pl = torch.Tensor([0.0]).cuda()
        loss_gt = torch.Tensor([0.0]).cuda()
        loss = torch.Tensor([0.0]).cuda()
        prediction = torch.tensor([], device=self.device)
        ground_truth = torch.tensor([], device=self.device)
        for student in range(batch.shape[0]):
            """大矩阵"""
            index = torch.tensor([i for i in range(self.max_step - 1)], dtype=torch.long, device=self.device)
            s_index = qsa[student, 1:, 1] - 1
            q_index = qsa[student, 1:, 0] - 1  # 题号变下标，减一
            ps = pred_s[student, index, s_index]
            pq = pred_q[student, index, q_index]
            """50道题"""
            # ps = pred_s[student].squeeze(1)
            # pq = pred_q[student].squeeze(1)

            p = (pq + ps) / 2
            s = qsa[student, 1:, 1]
            a = qsa[student, 1:, 2].float()
            # 不考虑补充的零
            for i in range((C.MAX_STEP - 1) - 1, -1, -1):
                if s[i] > 0:
                    p = p[:i + 1]
                    ps = ps[:i + 1]
                    pq = pq[:i + 1]
                    a = a[:i + 1]
                    break
            """Pseudo-Labelling"""
            # loss_pl += 0.01 * (torch.sum((ps - pq) ** 2))  # mse L2 均方误差
            # loss_pl += 0.001 * (torch.sum(torch.abs(ps-pq)) + torch.sum(torch.abs(pq-ps)))  # mae L1 平均绝对误差
            loss_gt += self.crossEntropy(ps, a) + self.crossEntropy(pq, a)
            loss = loss_gt

            prediction = torch.cat([prediction, p])
            ground_truth = torch.cat([ground_truth, a])
        return loss, prediction, ground_truth


def train_epoch(model, trainLoader, optimizer, loss_func, device):
    global loss
    for batch in tqdm.tqdm(trainLoader, desc='Training:    ', mininterval=2):
        # shape of a batch:[batch_size, max_step, 2 * ques + 2 * skill]
        batch = batch.to(device)
        ps, pq = model(batch)
        loss, _, _ = loss_func(ps, pq, batch)
        # p = model(batch)
        # loss, _, _ = loss_func(p, batch)
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()

    return model, optimizer


def test_epoch(model, testLoader, loss_func, device):
    ground_truth = torch.tensor([], device=device)
    prediction = torch.tensor([], device=device)
    loss = torch.tensor([], device=device)
    for batch in tqdm.tqdm(testLoader, desc='Testing:     ', mininterval=2):
        batch = batch.to(device)
        ps, pq = model(batch)
        loss, p, a = loss_func(ps, pq, batch)
        # p = model(batch)
        # loss, p, a = loss_func(p, batch)
        prediction = torch.cat([prediction, p])
        ground_truth = torch.cat([ground_truth, a])
    performance(ground_truth, prediction)
    print('loss:', loss.item())
    return performance(ground_truth, prediction)
