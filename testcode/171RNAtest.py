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
        
        label = self.labels[index]

        
        graph = load_graph(sequence_name)
        
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
        
        label = self.labels[index]

        # sequence_embedding = #embedding(sequence_name, sequence, EMBEDDING)
        # #structural_features = #get_dssp_features(sequence_name)
        graph = load_graph_test(sequence_name)
        
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

           
            x = x+x0

        x = self.linear_final(x)
        return x



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
        
        self.optimizer = torch.optim.Adam(self.parameters(),lr=0.001) 
           
    def forward(self,x,adj):
        
        outputlstm,(h,c)=self.lstm(x)
        x=self.fc1(x)
        outputgat=self.gat(x,adj)
        
        output=torch.cat([outputlstm,outputgat],dim=2)

        output=torch.mean(output,dim=1)
        x=self.fc(output)
        return x




def norm_dis(mx): 
    return 2 / (1 + (np.maximum(mx, 4) / 4))

def normalize(mx):
    rowsum = np.array(mx.sum(1))
    r_inv = (rowsum ** -0.5).flatten()
    r_inv[np.isinf(r_inv)] = 0
    r_mat_inv = np.diag(r_inv)
    result = r_mat_inv @ mx @ r_mat_inv
    return result



def padded_adj(original_matrix):
    
    dim=original_matrix.shape[0]
    
    if dim >=512:
        return original_matrix[:512,:512]
    else:
        padded_matrix = np.zeros((512, 512))

    
        return padded_matrix


def load_graph(sequence_name):

    

    dismap = np.load( "/mnt/raid5/data3/xli/three_work/task1/distrain_DNA/" + sequence_name + ".npy")
    mask = ((dismap >= 0) * (dismap <= 14))
    
    adjacency_matrix = mask.astype(np.int)
    

    norm_matrix = normalize(adjacency_matrix.astype(np.float32))
    norm_matrix=padded_adj(norm_matrix).astype(np.float32)
    return norm_matrix

def load_graph_test(sequence_name):

    

    dismap = np.load( "/home/xli/NABProt/task1ab/tomato_prot/dist/RNAdist/" + sequence_name + ".npy")
    mask = ((dismap >= 0) * (dismap <= 14))
    mask = ((dismap >= 0) * (dismap <= 14))
    
    adjacency_matrix = mask.astype(np.int)
   
    norm_matrix = normalize(adjacency_matrix.astype(np.float32))
    norm_matrix=padded_adj(norm_matrix).astype(np.float32)
    return norm_matrix

def to_var(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return x


def loadembedding(name):
    fea = np.load( "/mnt/raid5/data3/xli/three_work/task1/task1fea/" + name + ".npy")
    fea=fea[1:-1]
    return fea


def loadtestembedding(name):
    fea = np.load( "/home/xli/NABProt/task1ab/T5model/RNA/RNAtestfile/tomatofea/" + name + ".npy")
    fea=fea[1:-1]
    if fea.shape[0]>=512:
        fea=fea[:512,:]
    else:
        padded_matrix = np.zeros((512, 1024))
        padded_matrix[:fea.shape[0], :] = fea
        fea=padded_matrix
    return fea


def train_one_epoch(model, data_loader):
    n=0
    epoch_loss_train=0
    for data in data_loader:
        model.optimizer.zero_grad()
       
        
        name,_,label, node_features, graph=data

        
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
        loss.backward()
        # print(loss,n)
        model.optimizer.step()
        epoch_loss_train += loss.item()
        n += 1

    epoch_loss_train_avg = epoch_loss_train / n
    
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





def getRNA171list():
    namelist=[]
    seqlist=[]
    dir='/home/xli/NABProt/task1ab/tomato_prot/data/RNAfasta171.fasta'
    with open(dir,'r') as f:
        for line in f:
            if line[0]=='>':
                namelist.append(line[1:-1])
            else:
                seqlist.append(line[:-1])
 
    return namelist,seqlist

def genRNA171dataframe():
    RNAnames,RNAseqs=getRNA171list()

    

    namelist=[]
    seqlist=[]
    labellist=[]

    for i in range(len(RNAnames)):
        namelist.append(RNAnames[i])
        seqlist.append(RNAseqs[i])
        labellist.append(1)
    
    


    testdict={'name':namelist,'seq':seqlist,'label':labellist}
    test_dataframe = pd.DataFrame(testdict)
    
    return test_dataframe
#######################################################################

model=simplemodel()

if torch.cuda.is_available():
    model.cuda()


from sklearn.utils import shuffle

testset=genRNA171dataframe()



test_loader = DataLoader(dataset=ProDataTestset(testset), batch_size=1, shuffle=False, num_workers=2)


model.load_state_dict(torch.load('/home/xli/NABProt/task1ab/a_retrainmodel/T5/RNA/test/RNAmodel/epoch7RNA.pkl'))
model.eval()
epoch_loss_avg, valid_true, valid_pred, _  = evaluate(model, test_loader)
    # print('evl_loss',epoch_loss_avg)

print(valid_true, valid_pred)


