

from transformers import T5Tokenizer, T5EncoderModel
import torch
import re
import numpy as np
import time
import pandas as pd
import pickle

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


tokenizer = T5Tokenizer.from_pretrained('/home/xli/NABProt/task2/diff_fea/t5_fea/set_file', do_lower_case=False)
model = T5EncoderModel.from_pretrained("/home/xli/NABProt/task2/diff_fea/t5_fea/set_file")

model.to(device)

def extract_fea(mystr):
    sequence_examples=[]
    sequence_examples.append(mystr)
    #sequence_examples = ["PRTEINO", "SEQWENCE"]
    sequence_examples = [" ".join(list(re.sub(r"[UZOB]", "X", sequence))) for sequence in sequence_examples]
    ids = tokenizer.batch_encode_plus(sequence_examples, add_special_tokens=True, padding="longest")
    input_ids = torch.tensor(ids['input_ids']).to(device)
    attention_mask = torch.tensor(ids['attention_mask']).to(device)
    with torch.no_grad():
        embedding_rpr = model(input_ids=input_ids, attention_mask=attention_mask)
    emb_0 = embedding_rpr.last_hidden_state[0, :] 
    
    return emb_0.detach().cpu().numpy()



def gentraindataframe():
    with open(r'/home/xli/NABProt/task1ab/dataft/task1_train_pos7594.pkl', 'rb') as f:
        traintposset = pickle.load(f)

    with open(r'/home/xli/NABProt/task1ab/dataft/task1_train_neg10188.pkl', 'rb') as f1:
        traintnegset = pickle.load(f1)
    
    trainset={}
    trainset.update(traintposset)
    trainset.update(traintnegset)
    # print(trainset['P45771'])
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


trainset=gentraindataframe()
print(trainset)

namelist=list(trainset['name'])
seqlist=list(trainset['seq'])
a=time.time()
for i in range(len(namelist)):   
    # print(tmpfea)
    output = extract_fea(seqlist[i])
    dir='/home/xli/NABProt/task1ab/dataft/abfea/T5fea/T5fea/'+namelist[i]+'.npy'
    np.save(dir,output)

b=time.time()
print(b-a)






