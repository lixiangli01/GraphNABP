import torch
import esm
import numpy as np
import pandas as pd
import pickle
# Load ESM model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model, alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
model = model.to(device)
batch_converter = alphabet.get_batch_converter()
model.eval()  


def genfeature_esm(seq):
    data = [("protein1", seq)]
    batch_labels, batch_strs, batch_tokens = batch_converter(data)
   
    batch_tokens = batch_tokens.to(device)
    
    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[33], return_contacts=True)
    token_representations = results["representations"][33]

  
    esmfea=token_representations[0][1:-1].cpu().numpy()
    return esmfea

def split_sequence(sequence, segment_length=1022):
    num_segments = len(sequence) // segment_length
    segments = [sequence[i * segment_length: (i + 1) * segment_length] for i in range(num_segments)]
    if len(sequence) % segment_length != 0:
        segments.append(sequence[num_segments * segment_length:]) 
    array_list=[]
    for i in segments:
        array_list.append(genfeature_esm(i))

    result_array = np.concatenate(array_list, axis=0)
    return result_array




def gentraindataframe():#防止变量
    with open(r'/home/xli/NABProt/task1ab/dataft/task1_train_pos7594.pkl', 'rb') as f:
        traintposset = pickle.load(f)

    with open(r'/home/xli/NABProt/task1ab/dataft/task1_train_neg10188.pkl', 'rb') as f1:
        traintnegset = pickle.load(f1)
    
    trainset={}
    trainset.update(traintposset)
    trainset.update(traintnegset)
    
    trainlist=list(trainset.keys())
    seqlist=[]
    labellist=[]
    for i in trainlist:
        seqstr=trainset[i][0]
        label=trainset[i][1]
        seqlist.append(seqstr)
        labellist.append(label)

    traindict={'name':trainlist,'seq':seqlist,'label':labellist}
    train_dataframe = pd.DataFrame(traindict)
    # print(train_dataframe)
    return train_dataframe


def genRNAtrainalldataframe():
    with open(r'/home/xli/NABProt/task1ab/dataft/task1RNA_pos3948.pkl', 'rb') as f1:
        testposset1 = pickle.load(f1)

    

    testposset={}
    testposset.update(testposset1)
    

    testposlist=list(testposset.keys())

    namelist=[]
    seqlist=[]
    labellist=[]
    for i in testposlist:
        namelist.append(i)
        seqlist.append(testposset[i][0])
        labellist.append(1)
    
    


    testdict={'name':namelist,'seq':seqlist,'label':labellist}
    test_dataframe = pd.DataFrame(testdict)
    # print(train_dataframe)
    return test_dataframe

trainset=genRNAtrainalldataframe()


namelist=list(trainset['name'])
seqlist=list(trainset['seq'])




for i in range(len(namelist)):
    if len(seqlist[i])<=1022:
        tmpfea=genfeature_esm(seqlist[i])
    else:
        tmpfea=split_sequence(seqlist[i], segment_length=1022)
        
    
    dir='/home/xli/NABProt/task1ab/dataft/abfea/ESMfea/esmfea/'+namelist[i]+'.npy'
    np.save(dir,tmpfea)
