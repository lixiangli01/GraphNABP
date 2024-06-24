import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.parameter import Parameter
import math
from sklearn.model_selection import KFold
from sklearn import metrics
import warnings
from torch.autograd import Variable
import pickle
warnings.filterwarnings("ignore",category=DeprecationWarning)

SEED = 2020
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.set_device(0)
    torch.cuda.manual_seed(SEED)


class ProDataset(Dataset):
    def __init__(self, dataframe):
        self.names = dataframe['name'].values
        self.sequences = dataframe['seq'].values
        self.labels = dataframe['label'].values

    def __getitem__(self, index):
        sequence_name = self.names[index]
        sequence = self.sequences[index]
        #label = np.array(self.labels[index])
        label = self.labels[index]

        # sequence_embedding = #embedding(sequence_name, sequence, EMBEDDING)
        # #structural_features = #get_dssp_features(sequence_name)
        graph = load_graph(sequence_name)
        # #node_features = np.concatenate([sequence_embedding, structural_features], axis = 1)
        # #graph = load_graph(sequence_name)
        node_features=np.array(loadembedding(sequence_name))

        return sequence_name, sequence, label, node_features, graph
    def __len__(self):
        return len(self.labels)

class ProDataTestset(Dataset):
    def __init__(self, dataframe):
        self.names = dataframe['name'].values
        self.sequences = dataframe['seq'].values
        self.labels = dataframe['label'].values

    def __getitem__(self, index):
        sequence_name = self.names[index]
        sequence = self.sequences[index]
        #label = np.array(self.labels[index])
        label = self.labels[index]

        # sequence_embedding = #embedding(sequence_name, sequence, EMBEDDING)
        # #structural_features = #get_dssp_features(sequence_name)
        graph = load_graph_test(sequence_name)
        # #node_features = np.concatenate([sequence_embedding, structural_features], axis = 1)
        # #graph = load_graph(sequence_name)
        node_features=np.array(loadtestembedding(sequence_name))

        return sequence_name, sequence, label, node_features, graph
    def __len__(self):
        return len(self.labels)      

class net_prot_gat(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear0 = nn.ModuleList([nn.Linear(256, 256) for _ in range(4)])
        self.linear1 = nn.ModuleList([nn.Linear(256, 256) for _ in range(4)])
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.w_attn = nn.ModuleList([nn.Linear(256, 256) for _ in range(4)])
        self.linear_final = nn.Linear(256, 256)

    def forward(self, x, adj):
        adj[:, list(range(512)), list(range(512))] = 1
        for l in range(4):#7个头
            x0 = x
            adj_attn = self.sigmoid(torch.einsum('bij,bkj->bik', self.w_attn[l](x), x))
            adj_attn = adj_attn + 1e-5 * torch.eye(512).to(x.device)
            adj_attn = torch.einsum('bij,bij->bij', adj_attn, adj)
            adj_attn_sum = torch.einsum('bij->bi', adj_attn)
            adj_attn = torch.einsum('bij,bi->bij', adj_attn, 1/adj_attn_sum)

            x = torch.einsum('bij,bjd->bid', adj_attn, x)
            x = self.relu(self.linear0[l](x))
            x = self.relu(self.linear1[l](x))

            # x += x0
            x = x+x0

        x = self.linear_final(x)
        return x
    
def load_graph_test(sequence_name):

    
#    #dismap = np.load(Feature_Path + "distance_map/" + sequence_name + ".npy") 
    try:
        dismap = np.load( "/home/xli/NABProt/task1ab/dataft/test255/testPDB255dist/" + sequence_name[1:] + ".npy")
    except:
        dismap = np.load( "/home/xli/NABProt/task1ab/dataft/test255/testPDB255dist/" + sequence_name+ ".npy")
    mask = ((dismap >= 0) * (dismap <= 14))
    
    adjacency_matrix = mask.astype(np.int)
    # elif MAP_TYPE == "c":
    #     adjacency_matrix = norm_dis(dismap)
    #     adjacency_matrix = mask * adjacency_matrix

    norm_matrix = normalize(adjacency_matrix.astype(np.float32))
    norm_matrix=padded_adj(norm_matrix).astype(np.float32)
    return norm_matrix

def loadtestembedding(name):
    try:
        fea = np.load( "/home/xli/NABProt/task1ab/T5model/DNA/DNAtestfeafile/255fea/" + name + ".npy")
    except:
        fea = np.load( "/home/xli/NABProt/task1ab/T5model/RNA/RNAtestfile/255fea/" +name + ".npy")
    fea=fea[1:-1]
    if fea.shape[0]>=512:
        fea=fea[:512,:]
    else:
        padded_matrix = np.zeros((512, 1024))
        padded_matrix[:fea.shape[0], :] = fea
        fea=padded_matrix
    return fea

class simplemodel(nn.Module):
    def __init__(self):
        super(simplemodel,self).__init__()
        # self.hidden_dim=512
        self.lstm=nn.LSTM(1024,128,batch_first=True,bidirectional=True,num_layers=2)

        self.gat = net_prot_gat()
        self.fc1 = nn.Linear(1024, 256)

        self.fc = nn.Sequential(nn.Linear(512, 128),
                                        nn.LeakyReLU(0.1),
                                        nn.Dropout(0.1),
                                        nn.Linear(128, 32),
                                        nn.LeakyReLU(0.1),
                                        nn.Dropout(0.1),
                                        nn.Linear(32, 2))

        self.criterion = nn.CrossEntropyLoss()
        #self.optimizer = torch.optim.SGD(self.parameters(),lr=0.001)
        self.optimizer = torch.optim.Adam(self.parameters(),lr=0.001) 
           
    def forward(self,x,adj):#x:(b,len,fea_len)
        
        outputlstm,(h,c)=self.lstm(x)#x:(b,len,fea1024);output:(b,len,fea256)
        x=self.fc1(x)
        outputgat=self.gat(x,adj)
        #print(outputgat.shape)
        output=torch.cat([outputlstm,outputgat],dim=2)

        output=torch.mean(output,dim=1)#output:(64,256)
        x=self.fc(output)
        return x




def norm_dis(mx): # from SPROF
    return 2 / (1 + (np.maximum(mx, 4) / 4))

def normalize(mx):
    rowsum = np.array(mx.sum(1))
    r_inv = (rowsum ** -0.5).flatten()
    r_inv[np.isinf(r_inv)] = 0
    r_mat_inv = np.diag(r_inv)
    result = r_mat_inv @ mx @ r_mat_inv
    return result



def padded_adj(original_matrix):#输入是（103，103）
    #original_matrix = np.zeros((103, 103))
    dim=original_matrix.shape[0]
    # 创建一个512x512的零矩阵
    if dim >=512:
        return original_matrix[:512,:512]
    else:
        padded_matrix = np.zeros((512, 512))

    # 将原始矩阵的值复制到新的矩阵中
        padded_matrix[:dim, :dim] = original_matrix

    # 打印结果
    # print(padded_matrix)
        return padded_matrix


def load_graph(sequence_name):

    
#    #dismap = np.load(Feature_Path + "distance_map/" + sequence_name + ".npy") 

    try:
        dismap = np.load( "/mnt/raid5/data3/xli/three_work/task1/distrain_DNA/" + sequence_name + ".npy")
    except:
        dismap = np.load( "/mnt/raid5/data3/xli/three_work/task1/dis_task1RNA_train_pos_3948/" + sequence_name + ".npy")
    mask = ((dismap >= 0) * (dismap <= 14))
    
    adjacency_matrix = mask.astype(np.int)
    # elif MAP_TYPE == "c":
    #     adjacency_matrix = norm_dis(dismap)
    #     adjacency_matrix = mask * adjacency_matrix

    norm_matrix = normalize(adjacency_matrix.astype(np.float32))
    norm_matrix=padded_adj(norm_matrix).astype(np.float32)
    return norm_matrix



def to_var(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return x


def loadembedding(name):
    fea = np.load( "/mnt/raid5/data3/xli/three_work/task1T5fea/T5fea/T5fea/" + name + ".npy")
    fea=fea[1:-1]
    if fea.shape[0]>=128:
        fea=fea[:128,:]
        padded_matrix = np.zeros((512, 1024))
        padded_matrix[:fea.shape[0], :] = fea
        fea=padded_matrix
    else:
        padded_matrix = np.zeros((512, 1024))
        padded_matrix[:fea.shape[0], :] = fea
        fea=padded_matrix
    return fea

# def embeddingfea(namelist):
#     batchfea=np.array([loadembedding(i) for i in namelist])
#     return batchfea



# model.train() 
def train_one_epoch(model, data_loader):
    n=0
    epoch_loss_train=0
    for data in data_loader:
        model.optimizer.zero_grad()
        # optimizer.zero_grad()
        
        name,_,label, node_features, graph=data

        # namelist=list(name)

        # print(seq)#元组
        # seqlist=list(seq)
        # seqlist=[genstr(seq) for seq in seqlist]

        # seq=np.array(seq)
        # seq=torch.from_numpy(seq)
        # seq=to_var(seq)
        # label=to_var(label)#(64,514,1024)的张量截取（64，512，1024）

        # embedding = embeddingfea(namelist)
        # embedding = np.array(embedding)
        embedding=node_features
        #embedding=torch.from_numpy(embedding)

        #embedding=torch.squeeze(embedding)#????????

        #embedding=embedding[:,1:-1,:]#获得（batch,512,1024）
        embedding=torch.tensor(embedding,dtype=torch.float)
        embedding=to_var(embedding)

        y_true=list(label)
        y_true=np.array([int(i) for i in y_true])
        y_true=torch.from_numpy(y_true)
        y_true=to_var(y_true)
        
        graph=to_var(graph)

        y_pred =model(embedding,graph)
        loss=model.criterion(y_pred, y_true)
        loss.backward()
        # print(loss,n)
        model.optimizer.step()
        epoch_loss_train += loss.item()
        n += 1

    epoch_loss_train_avg = epoch_loss_train / n
    # print(epoch_loss_train_avg)
    return epoch_loss_train_avg


    
def evaluate(model, data_loader):
    model.eval()

    epoch_loss = 0.0
    n = 0
    valid_pred = []
    valid_true = []
    pred_dict = {}


    for data in data_loader:
        with torch.no_grad():
            name,_,label, node_features, graph=data
            # namelist=list(name)

            #embedding = embeddingfea(namelist)
            embedding=node_features
            embedding=torch.tensor(embedding,dtype=torch.float)
            embedding=to_var(embedding)

            y_true=list(label)
            y_true=np.array([int(i) for i in y_true])
            y_true=torch.from_numpy(y_true)
            y_true=to_var(y_true)
            
            graph=to_var(graph)

            y_pred =model(embedding,graph)
            loss=model.criterion(y_pred, y_true)

            softmax = torch.nn.Softmax(dim=1)
            y_pred = softmax(y_pred)
            y_pred = y_pred.cpu().detach().numpy()

            y_true = y_true.cpu().detach().numpy()
            valid_pred += [pred[1] for pred in y_pred]
            valid_true += list(y_true)
            pred_dict[name[0]] = [pred[1] for pred in y_pred]


            epoch_loss += loss.item()
            n += 1
    epoch_loss_avg = epoch_loss / n
    return epoch_loss_avg, valid_true, valid_pred, pred_dict


def analysis(y_true, y_pred, best_threshold = None):
    if best_threshold == None:
        best_f1 = 0
        best_threshold = 0
        for threshold in range(0, 100):
            threshold = threshold / 100
            binary_pred = [1 if pred >= threshold else 0 for pred in y_pred]
            binary_true = y_true
            f1 = metrics.f1_score(binary_true, binary_pred)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold

    binary_pred = [1 if pred >= best_threshold else 0 for pred in y_pred]
    binary_true = y_true
    # binary evaluate
    binary_acc = metrics.accuracy_score(binary_true, binary_pred)
    precision = metrics.precision_score(binary_true, binary_pred)
    recall = metrics.recall_score(binary_true, binary_pred)
    f1 = metrics.f1_score(binary_true, binary_pred)
    AUC = metrics.roc_auc_score(binary_true, y_pred)
    precisions, recalls, thresholds = metrics.precision_recall_curve(binary_true, y_pred)
    AUPRC = metrics.auc(recalls, precisions)
    mcc = metrics.matthews_corrcoef(binary_true, binary_pred)

    results = {
        'binary_acc': binary_acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'AUC': AUC,
        'AUPRC': AUPRC,
        'mcc': mcc,
        'threshold': best_threshold
    }
    return results

# LAYER = 8#8
# INPUT_DIM = 1024
# HIDDEN_DIM = 256
# NUM_CLASSES = 2
# DROPOUT = 0.1#0.1
# LAMBDA = 1.5
# ALPHA = 0.7
# VARIANT = True
# LEARNING_RATE = 1E-5
# WEIGHT_DECAY = 0








# trainset=pd.read_csv('/mnt/raid5/data3/xli/three_work/task3/new_train.csv')
# trainset['label'] = trainset['label'].replace(-1, 0)
# #train_loader = DataLoader(dataset=ProDataset(trainset), batch_size=1, shuffle=True, num_workers=2)

# sequence_names = trainset['name'].values
# sequence_labels = trainset['label'].values


def gentraindataframe():#防止变量
    with open(r'/mnt/raid5/data3/xli/three_work/task1/dlseq/task1_train_pos7594.pkl', 'rb') as f:
        traintposset1 = pickle.load(f)

    with open(r'/mnt/raid5/data3/xli/three_work/task1/dlseq/task1_train_neg10188.pkl', 'rb') as f1:
        traintnegset1 = pickle.load(f1)

    with open(r'/mnt/raid5/data3/xli/three_work/task1/dlseq_RNA/task1RNA_pos3948.pkl', 'rb') as f2:
        traintnegset2 = pickle.load(f2)
    
    # print(traintest.keys())
    # train_dataframe = pd.DataFrame(traintest)
    # print(train_dataframe,)
    trainposset={}
    trainposset.update(traintposset1)
    
    # print(trainset['P45771'])
    trainposlist=list(trainposset.keys())

    namelist=[]
    seqlist=[]
    labellist=[]

    for i in trainposlist:
        # seqstr=trainset[i][0]
        # label=trainset[i][1]
        # seqlist.append(seqstr)
        # labellist.append(1)
        namelist.append(i)
        seqlist.append(trainposset[i])
        labellist.append(1)
    
    trainnegset={}
    trainnegset.update(traintnegset1)
    trainnegset.update(traintnegset2)
    trainneglist=list(trainnegset.keys())
    for j in trainneglist:
        namelist.append(j)
        seqlist.append(trainnegset[j])
        labellist.append(0)

    traindict={'name':namelist,'seq':seqlist,'label':labellist}
    train_dataframe = pd.DataFrame(traindict)
    print(train_dataframe)
    return train_dataframe

# def gentest474DNAdataframe():#防止变量
#     with open(r'/home/xli/NABProt/task1ab/dataft/testdata/test474pos175.pkl', 'rb') as f1:
#         testposset1 = pickle.load(f1)

#     with open(r'/home/xli/NABProt/task1ab/dataft/testdata/test474pos8.pkl', 'rb') as f2:
#         testposset2 = pickle.load(f2)

#     with open(r'/home/xli/NABProt/task1ab/dataft/testdata/test474neg223.pkl', 'rb') as f3:
#         testnegset3 = pickle.load(f3)

#     with open(r'/home/xli/NABProt/task1ab/dataft/testdata/test474pos68.pkl', 'rb') as f4:
#         testnegset4 = pickle.load(f4)

#     testposset={}
#     testposset.update(testposset1)
#     testposset.update(testposset2)
#     # labelpos=[1 for i in range(183)]

#     testposlist=list(testposset.keys())

#     namelist=[]
#     seqlist=[]
#     labellist=[]
#     for i in testposlist:
#         namelist.append(i)
#         seqlist.append(testposset[i])
#         labellist.append(1)
    

#     testnegset={}
#     testnegset.update(testnegset3)
#     testnegset.update(testnegset4)

#     testneglist=list(testnegset.keys())
#     for j in testneglist:
#         namelist.append(j)
#         seqlist.append(testnegset[j])
#         labellist.append(0)
#     testdict={'name':namelist,'seq':seqlist,'label':labellist}
#     test_dataframe = pd.DataFrame(testdict)
#     print(test_dataframe)
#     return test_dataframe
def getRNAlist():
    namelist=[]
    seqlist=[]
    dir='/home/xli/NABProt/task1ab/dataft/test255/pos70.fasta'
    with open(dir,'r') as f:
        for line in f:
            if line[0]=='>':
                namelist.append(line[1:-1])
            else:
                seqlist.append(line[:-1])
    # print(namelist)
    # print(seqlist)
    return namelist,seqlist

def gentest255DNAdataframe():#防止变量
    with open(r'/home/xli/NABProt/task1ab/dataft/test255/test255pos93.pkl', 'rb') as f1:
        testposset1 = pickle.load(f1)

    # with open(r'/mnt/raid5/data3/xli/three_work/task1/dlseq/test474/test474pos8.pkl', 'rb') as f2:
    #     testposset2 = pickle.load(f2)

    with open(r'/home/xli/NABProt/task1ab/dataft/test255/test255neg92.pkl', 'rb') as f3:
        testnegset1 = pickle.load(f3)

    # with open(r'/home/xli/NABProt/task1ab/dataft/test255/test255neg92.pkl', 'rb') as f3:
    #     testnegset = pickle.load(f3)

    RNAnames,RNAseqs=getRNAlist()

    testposset={}
    testposset.update(testposset1)
    # testposset.update(testposset2)
    # labelpos=[1 for i in range(183)]

    testposlist=list(testposset.keys())

    namelist=[]
    seqlist=[]
    labellist=[]
    for i in testposlist:
        namelist.append(i)
        seqlist.append(testposset[i])
        labellist.append(1)
    
    testnegset={}
    testnegset.update(testnegset1)
    # testnegset.update(testnegset4)

    testneglist=list(testnegset.keys())
    for j in testneglist:
        namelist.append(j)
        seqlist.append(testnegset[j])
        labellist.append(0)

    for i in range(len(RNAnames)):
        namelist.append(RNAnames[i])
        seqlist.append(RNAseqs[i])
        labellist.append(0)

    testdict={'name':namelist,'seq':seqlist,'label':labellist}
    test_dataframe = pd.DataFrame(testdict)
    # print(train_dataframe)
    print(test_dataframe)
    return test_dataframe

from sklearn.utils import shuffle
# trainset=gentraindataframe()
# trainset=shuffle(trainset)#[:500]

# sequence_names = trainset['name'].values

model=simplemodel()

if torch.cuda.is_available():
    model.cuda()



#train_loader = DataLoader(dataset=ProDataset(trainset), batch_size=64, shuffle=True, num_workers=2,drop_last=True)#,drop_last=True
testset=gentest255DNAdataframe()

test_loader = DataLoader(dataset=ProDataTestset(testset), batch_size=1, shuffle=False, num_workers=2)

model.load_state_dict(torch.load('/home/xli/NABProt/task1ab/a_retrainmodel/T5/DNA/test/DNAmodel/epoch4DNA.pkl'))
model.eval()    
epoch_loss_avg, valid_true, valid_pred, _  = evaluate(model, test_loader)
print(valid_true, valid_pred)
print('######################################################')
result_valid = analysis(valid_true, valid_pred, 0.5)

print("Valid loss: ", epoch_loss_avg)
print("Valid binary acc: ", result_valid['binary_acc'])
print("Valid precision: ", result_valid['precision'])
print("Valid recall: ", result_valid['recall'])
print("Valid f1: ", result_valid['f1'])
print("Valid AUC: ", result_valid['AUC'])
print("Valid AUPRC: ", result_valid['AUPRC'])
print("Valid mcc: ", result_valid['mcc'])


