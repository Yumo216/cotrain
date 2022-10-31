Dpath = '../../KTDataset'

datasets = {
    'assist2009' : 'assist2009',
    'assist2012' : 'assist2012',
    'assist2017' : 'assist2017',
    'assistednet': 'assistednet',
}

# question number of each dataset
question = {
    'assist2009' : 16891,
    'assist2012' : 37125,
    'assist2017' : 3162,
    'assistednet': 12372,
}

skill = {
    'assist2009' : 101,
    'assist2012' : 188,
    'assist2017' : 102,
    'assistednet': 1901,
}

DATASET = datasets['assistednet']
NUM_Q = question[DATASET]
NUM_S = skill[DATASET]

# the max step of RNN model
MAX_STEP = 50
BATCH_SIZE = 64
LR = 0.001
EPOCH = 50
INITIAL = 256  # 送入超图前初始化的维度

# DKT
INPUT = NUM_S * 2
HIDDEN = 256
LAYERS = 1
OUTPUT = NUM_S
# SAKT
heads = 8
length = MAX_STEP
hidden = HIDDEN
questions = NUM_Q
skill = NUM_S
dropout = 0.1

