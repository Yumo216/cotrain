
import sys

sys.path.append('../')
import torch.utils.data as Data
from KnowledgeTracing.Constant import Constants as C
from KnowledgeTracing.data.readdata import DataReader
from KnowledgeTracing.data.DKTDataSet import DKTDataSet

'''在深度学习的训练过程中，我们需要将数据分批的放入到训练网络中，批数量的大小也被成为batch_size
通过pytorch提供的dataloader方法，可以自动实现一个迭代器，每次返回一组batch_size个样本和标签来进行训练。'''


def getTrainLoader(train_data_path):
    handle = DataReader(train_data_path, C.MAX_STEP, C.NUM_S)
    trainques, trainskil, trainans = handle.getTrainData()
    dtrain = DKTDataSet(trainques, trainskil, trainans)
    trainLoader = Data.DataLoader(dtrain, batch_size=C.BATCH_SIZE, shuffle=True)

    return trainLoader


def getTestLoader(test_data_path):
    handle = DataReader(test_data_path, C.MAX_STEP, C.NUM_S)
    testques, testskil, testans = handle.getTestData()
    dtest = DKTDataSet(testques, testskil, testans)
    testLoader = Data.DataLoader(dtest, batch_size=C.BATCH_SIZE, shuffle=False)
    return testLoader


def getLoader(dataset):
    trainLoaders = []
    testLoaders = []
    if dataset == 'assist2009':
        trainLoader = getTrainLoader(C.Dpath + '/assist2009/assist2009_pid_train.csv')
        trainLoaders.append(trainLoader)
        testLoader = getTestLoader(C.Dpath + '/assist2009/assist2009_pid_test.csv')
        testLoaders.append(testLoader)
    elif dataset == 'assist2012':
        trainLoader = getTrainLoader(C.Dpath + '/assist2012/assist2012_pid_train.csv')
        trainLoaders.append(trainLoader)
        testLoader = getTestLoader(C.Dpath + '/assist2012/assist2012_pid_test.csv')
        testLoaders.append(testLoader)
    elif dataset == 'assist2017':
        trainLoader = getTrainLoader(C.Dpath + '/assist2017/assist2017_pid_train.csv')
        trainLoaders.append(trainLoader)
        testLoader = getTestLoader(C.Dpath + '/assist2017/assist2017_pid_test.csv')
        testLoaders.append(testLoader)
    elif dataset == 'assistednet':
        trainLoader = getTrainLoader(C.Dpath + '/assistednet/assistednet_pid_train.csv')
        trainLoaders.append(trainLoader)
        testLoader = getTestLoader(C.Dpath + '/assistednet/assistednet_pid_test.csv')
        testLoaders.append(testLoader)

    return trainLoaders[0], testLoaders[0]
