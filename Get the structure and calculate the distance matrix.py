import urllib.request
import time
import threading
import random
import urllib.request
import time
import os
import logging
import pickle
# 配置日志记录
logging.basicConfig(filename='task1trainnegdownload.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

errorlist_lock = threading.Lock()
dict_complex_lock=threading.Lock()

from Bio import PDB
import numpy as np
import warnings
import pandas as pd
import pickle

warnings.filterwarnings("ignore")


dictaa = {'GLY': 'G', 'ALA': 'A', 'VAL': 'V', 'LEU': 'L',
          'ILE': 'I', 'PHE': 'F', 'TRP': 'W', 'TYR': 'Y',
          'ASP': 'D', 'ASN': 'N', 'GLU': 'E', 'LYS': 'K',
          'GLN': 'Q', 'MET': 'M', 'SER': 'S', 'THR': 'T',
          'CYS': 'C', 'PRO': 'P', 'HIS': 'H', 'ARG': 'R','MSE':'M'}

aalist = ['GLY', 'ALA', 'VAL', 'LEU', 'ILE',
          'PHE', 'TRP', 'TYR', 'ASP', 'ASN',
          'GLU', 'LYS', 'GLN', 'MET', 'SER',
          'THR', 'CYS', 'PRO', 'HIS', 'ARG','MSE']

ls=['G','A','V','L','I',
    'P','F','Y','W','S',
    'T','C','M','N','Q',
    'D','E','K','R','H']


def ReadPDB(dirfile,n):
    dirstr = dirfile #+ '.pdb'
    with  open(dirstr, "r") as myfile:
        num = 0
        sum = 0
        ls_sort = []
        ls_sort_new = []
        ls_seq_pdb = []
        for line in myfile:
            if line[0:14] == 'MODEL        2':
                break
            elif line[0:4] == 'ATOM' and line[17:20] in aalist and num != int(line[22:26].replace(' ', ''))and line[21]==n or (line[0:4] == 'ANIS' and line[17:20]=='MSE' and num != int(line[22:26].replace(' ', ''))and line[21]==n):
                # if line.split()[0] == 'ATOM' and len(line.split()[3]) >= 3 and num != float(line.split()[5]) and line.split()[5].isdigit():
                # outfile.write(line.split()[3] + ',' + line.split()[5] + '\n')
                sum += 1
                ls_sort.append(line[22:26].replace(' ', ''))
                ls_sort_new.append(str(sum))
                ls_seq_pdb.append(line[17:20])
                # strout = line.split()[3] + ',' + line.split()[5]+','+dictaa[line.split()[3][-3:]]+','+str(sum)
                # print(strout)
                num = int(line[22:26].replace(' ', ''))
        
    myfile.close()
    strAA = ''
    for i in ls_seq_pdb:
        strAA += dictaa[i]
    return ls_sort, ls_sort_new, strAA




def mat_cat(pdbfile):
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure('protein', pdbfile)

    
    chain_id = 'A'
    ca_atom_coordinates = []

    for model in structure:
        for chain in model:
            if chain.id == chain_id:
                for residue in chain:
                    if 'CA' in residue:
                        ca_atom_coordinates.append(residue['CA'].get_coord())



    num_ca_atoms = len(ca_atom_coordinates)

    adjacency_matrix = np.zeros((num_ca_atoms, num_ca_atoms), dtype=float)

    for i in range(num_ca_atoms):
        for j in range(i + 1, num_ca_atoms):
            distance = np.linalg.norm(ca_atom_coordinates[i] - ca_atom_coordinates[j])
            # if distance < threshold:
            #     adjacency_matrix[i][j] = 1
            #     adjacency_matrix[j][i] = 1
            adjacency_matrix[i][j] = distance
            adjacency_matrix[j][i] = distance


    
    return adjacency_matrix


dict_complex={}

def Wdatadict(pdbnamedir):
    global dict_complex
    a=mat_cat(pdbnamedir)#map
    b = (ReadPDB(pdbnamedir, 'A')[2])#seq
    dir=r'D:\work_three\dis_task1_train_neg_10188\\'+pdbnamedir[:-4]+'.npy'
    np.save(dir,a)
    with dict_complex_lock:
        dict_complex[pdbnamedir[:-4]] = [b, '0','task1','10188_neg_train']#, 'task1', 'train'  #,


def downOnePdb(filename):
    try:
        url='https://alphafold.ebi.ac.uk/files/AF-'+filename+'-F1-model_v4.pdb'
        html = urllib.request.urlopen(url, timeout=5).read()
        with open(filename.replace('/', '_') + ".pdb", "wb") as f:
            f.write(html)
        # saveHtml(filename, html)
        time.sleep(0.5)
        

        """调用处理和写入函数"""
        pdbnamedir=filename.replace('/', '_') + ".pdb"
        Wdatadict(pdbnamedir)
        os.remove(pdbnamedir)
        logging.info("下载成功并处理成功" + ':' + filename)
    except Exception as e:
        print(f"下载失败:{filename}, 错误: {str(e)}")
        time.sleep(random.randint(0, 2))
        with errorlist_lock:
            errorlist.append(filename)
        logging.error(f"下载失败:{filename}, 错误: {str(e)}")



def download_in_parallel(file_list):

    
    threads = []

    for filename in file_list:
        thread = threading.Thread(target=downOnePdb, args=(filename,))
        threads.append(thread)
        thread.start()
    
    for thread in threads:
        thread.join()

def retry_failed_downloads(errorlist):
    errorlist_copy = list(errorlist)  

    
    for filename in errorlist_copy:
        downOnePdb(filename)
        
        with errorlist_lock:
            if filename in errorlist:
                errorlist.remove(filename)




if __name__ == '__main__':
    start_time = time.time()
    #file_list=['P0DOK1', 'D0E8I5', 'Q9GRW0', 'Q6EN42', 'Q72EF3', 'Q72EF4', 'Q92UV7', 'Q2RMD6', 'Q2FV99', 'F1QXD3', 'F1QGH7', 'I2DBY1', 'O81862', 'H3BCW1', 'P0DPK1', 'A7GBG3', 'Q3YPH4', 'G3I8R9', 'P0DPI1', 'W0T9X4', 'Q1GNW5', 'F1QCV2', 'Q5S3I3', 'Q9VIH7', 'A0A0H3JPC6', 'A0A0H2WWV6', 'P0CU66', 'Q6NU25', 'A0A0M3KKW3', 'B8Y6I0', 'Q9HWH9', 'P0DM12', 'Q31KC7', 'B3FK34', 'Q9LLQ3', 'Q9LLQ2', 'L8EUQ6', 'F7J0N2', 'D5SL78', 'G2IQQ5', 'Q7N561', 'B5UBC1', 'A0A1S3THR8', 'P0DPQ5', 'Q8Y588', 'Q9XA14', 'Q2FZE3', 'Q8Y8H5', 'Q15JG1', 'I4CHP3', 'H2K885', 'B6A880', 'B6A882', 'A0JD37', 'Q8XBI9', 'O59284', 'Q92ZM6', 'Q8L794', 'Q53W17', 'Q6CJY0', 'Q8TSG7', 'O22317', 'Q2RSU5', 'A0A0R4I952', 'Q6DQW3', 'E2R4X3', 'I3DZK9', 'W0TH64', 'Q85365', 'B6A876', 'Q8ISL8', 'B6A879', 'A0A0R4IKJ1', 'C5IY46', 'Q9GPE9', 'Q6CFT7', 'B5GMG2', 'P0DM15', 'C0HK58', 'C0HK53', 'C0HK56', 'Q9LBR2', 'W0TA43', 'A0A161CFW5', 'Q6C338', 'C0HK57', 'Q9GS23', 'Q6C326', 'C0HK52', 'Q6C877', 'P0DQD2', 'A0A0H3CDY2', 'P0DPD7', 'L7N6F8', 'P0DPQ8', 'A1KXI0', 'Q9F3C7', 'D1A7C3', 'A0A0K0MJN3', 'E4MYY0', 'Q8A7C8', 'Q9LBR4', 'Q97YD2', 'F2R776', 'Q5YTV5', 'K4REZ6', 'A0A0G3F8Z3', 'O29867', 'O74088', 'D2PPM7', 'A7IZE9', 'E1UYT9', 'Q8U4D2', 'B9JK80', 'P89884', 'A1IKL2', 'E1UYU0', 'O29918', 'O30195', 'A0A067SLB9', 'F8GV06', 'Q6D5T7', 'P0DPR0', 'P0DPR1', 'P0DQD3', 'A0NLY7', 'A0ZT93', 'Q9VB68', 'Q7AP87', 'Q76CE8', 'Q76CE6', 'A5XB26', 'H7FWB6', 'C7F6X3', 'Q81L65', 'Q5SK88', 'Q838S1', 'P0DM22', 'Q5SJ58', 'B3TMR8', 'Q8A712', 'B3IXK1', 'Q08FX8', 'O52793', 'A0A0K0MJ13', 'P0DPQ7', 'F1QR43', 'Q89YS5', 'Q4WF29', 'C0HK51', 'Q9X721', 'Q46085', 'Q42369', 'A0A222NNM9', 'P0DPG3', 'C0HLB1', 'C0HK60', 'C0HK59', 'Q6VZT9', 'W0T4V8', 'P0DPG4', 'A0A1L1QK34', 'W4VSA0', 'Q899Y1', 'Q8KGE0', 'P0DPI0', 'C1IWT2', 'Q45914', 'B6YWB8', 'P0DM21', 'Q589Y0', 'L8EBJ9', 'A5HC98', 'C4N147', 'Q7AP54', 'W4VSF6', 'Q96Y66', 'Q96XT2', 'Q96Y68', 'Q86MA1', 'Q59DX8', 'Q96XT4', 'P0DM24', 'D4Z2G1', 'Q988D0', 'P0DPF1', 'Q086E4', 'Q40960', 'Q9ATH2', 'E9P8D2', 'Q8PHA1', 'A0A384E129', 'A0A0H3C7V4', 'A0A0B4J274', 'A0A087WT01', 'A0A0B4J271', 'A0A075B6N1', 'A0A5B6', 'A0A0N7CSQ4', 'A0A0B4J279', 'A0A0B4J277', 'A0A0B4J272', 'A0A075B6T6', 'A0A578', 'A0A0K0K1A5', 'A0A0R4I951', 'A0A0B4J2E0', 'A7B1V0', 'E2RKN8', 'A0A0B4J268', 'Q5Z8S0', 'O97366', 'K4ZRC1', 'D3ZMK9', 'H2E7Q8', 'Q6EZC3', 'A0A0R4IMY7', 'A0A2L0ART2', 'Q6EZC2', 'A0A1D5PXA5', 'B5HDJ6', 'K7NCP5', 'Q4WF56']
    listfile = open('neglist.pickle', 'rb')  
    file_list = pickle.load(listfile)
    # num_threads = 12
    errorlist = []
    download_in_parallel(file_list)

    for retry in range(3):
        retry_failed_downloads(errorlist)
        print(errorlist)
        print('第',retry,'次')
        time.sleep(3)


    with open('task1_train_neg10188.pkl', 'wb') as f1:
        pickle.dump(dict_complex, f1)
    with open('task1_trainneg10188_errorlist.pkl', 'wb') as f2:
        pickle.dump(errorlist, f2)
    end_time = time.time()
    print(end_time - start_time, '秒')
    # np.save('test.npy',dict_complex)







