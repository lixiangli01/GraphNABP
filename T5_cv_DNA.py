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
        for l in range(4):
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
        
        outputlstm,(h,c)=self.lstm(x)
        x=self.fc1(x)
        outputgat=self.gat(x,adj)
        #print(outputgat.shape)
        output=torch.cat([outputlstm,outputgat],dim=2)

        output=torch.mean(output,dim=1)
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



def padded_adj(original_matrix):
    dim=original_matrix.shape[0]
    
    if dim >=512:
        return original_matrix[:512,:512]
    else:
        padded_matrix = np.zeros((512, 512))

    
        padded_matrix[:dim, :dim] = original_matrix

    
        return padded_matrix


def load_graph(sequence_name):

    
#    #dismap = np.load(Feature_Path + "distance_map/" + sequence_name + ".npy") 

    try:
        dismap = np.load( "/home/xli/NABProt/task1ab/dataft/distrain_DNA/" + sequence_name + ".npy")
    except:
        dismap = np.load( "/home/xli/NABProt/task1ab/dataft/dis_task1RNA_train_pos_3948/" + sequence_name + ".npy")
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
    fea = np.load( "/home/xli/NABProt/task1ab/dataft/abfea/T5fea/T5fea/" + name + ".npy")
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
        # optimizer.zero_grad()
        
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







def gentraindataframe():#防止变量
    with open(r'/home/xli/NABProt/task1ab/dataft/task1_train_pos7594.pkl', 'rb') as f:
        traintposset1 = pickle.load(f)

    with open(r'/home/xli/NABProt/task1ab/dataft/task1_train_neg10188.pkl', 'rb') as f1:
        traintnegset1 = pickle.load(f1)

    with open(r'/home/xli/NABProt/task1ab/dataft/task1RNA_pos3948.pkl', 'rb') as f2:
        traintnegset2 = pickle.load(f2)
    
    
    trainposset={}
    trainposset.update(traintposset1)
    
    # print(trainset['P45771'])
    trainposlist=list(trainposset.keys())

    namelist=[]
    seqlist=[]
    labellist=[]

    for i in trainposlist:
       
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

from sklearn.utils import shuffle
trainset=gentraindataframe()
trainset=shuffle(trainset)#[:500]

sequence_names = trainset['name'].values




fold=0
kfold = KFold(n_splits = 5, shuffle = True)



allfoldlist=[]
for train_index, valid_index in kfold.split(sequence_names):
    # print(train_index, valid_index)
    print("\n\n========== Fold " + str(fold + 1) + " ==========")
    train_dataframe = trainset.iloc[train_index, :]
    valid_dataframe = trainset.iloc[valid_index, :]
    print("Train on", str(train_dataframe.shape[0]), "samples, validate on", str(valid_dataframe.shape[0]),
            "samples")
    #model = GraphPPIS(LAYER, INPUT_DIM, HIDDEN_DIM, NUM_CLASSES, DROPOUT, LAMBDA, ALPHA, VARIANT)
    model=simplemodel()

    if torch.cuda.is_available():
        model.cuda()


    train_loader = DataLoader(dataset=ProDataset(train_dataframe), batch_size=64, shuffle=True, num_workers=2,drop_last=True)
    valid_loader = DataLoader(dataset=ProDataset(valid_dataframe), batch_size=1, shuffle=True, num_workers=2)
    
    foldlist=[]

    for epoch in range(10):

        print("\n========== Train epoch " + str(epoch + 1) + " ==========")
        model.train()
        epoch_loss_train_avg = train_one_epoch(model, train_loader)
        print('train_loss',epoch_loss_train_avg)

        epoch_loss_avg, valid_true, valid_pred, _  = evaluate(model, valid_loader)
        # print('evl_loss',epoch_loss_avg)
        result_valid = analysis(valid_true, valid_pred, 0.5)

        print("Valid loss: ", epoch_loss_avg)
        print("Valid binary acc: ", result_valid['binary_acc'])
        print("Valid precision: ", result_valid['precision'])
        print("Valid recall: ", result_valid['recall'])
        print("Valid f1: ", result_valid['f1'])
        print("Valid AUC: ", result_valid['AUC'])
        print("Valid AUPRC: ", result_valid['AUPRC'])
        print("Valid mcc: ", result_valid['mcc'])
        result_list=[epoch_loss_avg,result_valid['binary_acc'],result_valid['precision'],result_valid['recall'],result_valid['f1'],result_valid['AUC'],result_valid['AUPRC'],result_valid['mcc']]
        print(result_list)
        foldlist.append(result_list)
    fold+=1
    allfoldlist.append(foldlist)

cv_result=np.array(allfoldlist).mean(axis=0)

cv_pd=pd.DataFrame(cv_result,columns=['epoch_loss_avg', 'binary_acc', 'precision', 'recall','f1','AUC','AUPRC','mcc'])
cv_pd.to_csv('cv_dna_all_t5.csv')
