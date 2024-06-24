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
model.eval()  # disables dropout for deterministic results

# Prepare data (first 2 sequences from ESMStructuralSplitDataset superfamily / 4)
def genfeature_esm(seq):
    data = [("protein1", seq)]
    batch_labels, batch_strs, batch_tokens = batch_converter(data)
    # batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)

    # Extract per-residue representations (on CPU)
    batch_tokens = batch_tokens.to(device)
    
    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[33], return_contacts=True)
    token_representations = results["representations"][33]

    # print(token_representations.shape)   #torch.Size([1, 67, 1280])
    # print(token_representations[0][1:-1].shape)
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


# #a=genfeature_esm("MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG")
# trainset=pd.read_csv('/home/xli/NABProt/task2/data/data/trainDNA719.csv')
# #trainset=pd.read_csv('/home/xli/NABProt/task2/data/data/testDNA179.csv')
# print(trainset)

def gentraindataframe():#防止变量
    with open(r'/home/xli/NABProt/task1ab/dataft/task1_train_pos7594.pkl', 'rb') as f:
        traintposset = pickle.load(f)

    with open(r'/home/xli/NABProt/task1ab/dataft/task1_train_neg10188.pkl', 'rb') as f1:
        traintnegset = pickle.load(f1)
    # print(traintest.keys())
    # train_dataframe = pd.DataFrame(traintest)
    # print(train_dataframe,)
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


def genRNAtrainalldataframe():#防止变量
    with open(r'/home/xli/NABProt/task1ab/dataft/task1RNA_pos3948.pkl', 'rb') as f1:
        testposset1 = pickle.load(f1)

    # with open(r'/mnt/raid5/data3/xli/three_work/task1/dlseq/test474/test474pos8.pkl', 'rb') as f2:
    #     testposset2 = pickle.load(f2)

    # with open(r'/mnt/raid5/data3/xli/three_work/task1/dlseq/task1_train_neg10188.pkl', 'rb') as f3:
    #     testnegset = pickle.load(f3)

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
        seqlist.append(testposset[i][0])
        labellist.append(1)
    
    # testneglist=list(testnegset.keys())
    # for j in testneglist:
    #     namelist.append(j)
    #     seqlist.append(testnegset[j][0])
    #     labellist.append(0)


    testdict={'name':namelist,'seq':seqlist,'label':labellist}
    test_dataframe = pd.DataFrame(testdict)
    # print(train_dataframe)
    return test_dataframe

trainset=genRNAtrainalldataframe()
print(trainset)

namelist=list(trainset['name'])
seqlist=list(trainset['seq'])




for i in range(len(namelist)):
    if len(seqlist[i])<=1022:#之前为1024
        tmpfea=genfeature_esm(seqlist[i])
    else:
        tmpfea=split_sequence(seqlist[i], segment_length=1022)#最大长度为1024，但是有两个特殊token
        
    # print(tmpfea)
    dir='/home/xli/NABProt/task1ab/dataft/abfea/ESMfea/esmfea/'+namelist[i]+'.npy'
    np.save(dir,tmpfea)