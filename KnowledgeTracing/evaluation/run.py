import sys
from KnowledgeTracing.data.dataloader import getLoader
from KnowledgeTracing.Constant import Constants as C
import torch.optim as optim
from KnowledgeTracing.evaluation import eval
import torch
import numpy as np
import logging
from datetime import datetime

torch.cuda.set_device(0)
sys.path.append('../')

device = torch.device('cuda')
print('Dataset: ' + C.DATASET + ', Question: ' + str(C.NUM_Q) + ', Skill: ' + str(C.NUM_S) + '\n')
''' save log '''
logger = logging.getLogger('main')
logger.setLevel(level=logging.DEBUG)
date = datetime.now()
handler = logging.FileHandler(
    f'log/{date.year}_{date.month}_{date.day}_result.log')
handler.setLevel(logging.INFO)
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.info('This is a new training log')
logger.info('\nDataset: ' + str(C.DATASET))

'''set random seed'''
SEED = 0
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
np.random.seed(SEED)



from KnowledgeTracing.model.ckt.model import DKT
# from KnowledgeTracing.model.DKT.RNNModel import DKT
model = DKT(C.INPUT, C.HIDDEN, C.LAYERS, C.OUTPUT).cuda()


optimizer = optim.Adam(model.parameters(), lr=C.LR)

loss_func = eval.lossFunc(C.INPUT, C.HIDDEN, C.MAX_STEP, device).cuda()

trainLoaders, testLoaders = getLoader(C.DATASET)


best_auc = 0
for epoch in range(C.EPOCH):
    print('epoch: ' + str(int(epoch)+1))
    model, optimizer = eval.train_epoch(model,trainLoaders, optimizer, loss_func,device)
    with torch.no_grad():
        auc, acc = eval.test_epoch(model, testLoaders, loss_func, device)
        if best_auc < auc:
            best_auc = auc
            best_acc = acc
            best_epoch = epoch + 1
            # torch.save(model, '../model/save' + C.H + 'model.pkl')

        print('Best auc at present: %f  acc:  %f  Best epoch: %d' % (best_auc, best_acc, best_epoch))

