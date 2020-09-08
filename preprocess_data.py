from pymongo import MongoClient
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
import pickle
import numpy as np
import h5py
from scipy import sparse
###open database 
import queue as VanillaQueue
from multiprocessing import Pool, cpu_count, Process, Manager, Queue, JoinableQueue
import time
import random
import itertools, operator, random
import pandas as pd
import datetime
from tqdm import tqdm
from rdkit import RDLogger
import sys
lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)

input_file = sys.argv[1]

def create_rxn_Morgan2FP(rsmi, psmi, rxnfpsize = 2048, pfpsize=2048, useFeatures = False,calculate_rfp = True):
    """Create a rxn Morgan (r=2) fingerprint as bit vector from SMILES string lists of reactants and products"""
    # Modified from Schneider's code (2014)
    if calculate_rfp is True:
	    rsmi = rsmi.encode('utf-8')
	    try:
	    	mol = Chem.MolFromSmiles(rsmi)
	    except Exception as e:
	    	return

	    try:
	        fp_bit = AllChem.GetMorganFingerprintAsBitVect(mol=mol, radius=2, nBits = rxnfpsize, useFeatures=False, useChirality=True)
	        fp = np.empty(rxnfpsize,dtype = np.bool)
	        DataStructs.ConvertToNumpyArray(fp_bit,fp)

	    except Exception as e:
	        print("Cannot build reactant fp due to {}".format(e))

	        return
	        
	    rfp = fp
    else:
	    rfp = None

    psmi = psmi.encode('utf-8')
    try:
    	mol = Chem.MolFromSmiles(psmi)
    except Exception as e:
    	print(psmi)
    	return

    try:
        fp_bit = AllChem.GetMorganFingerprintAsBitVect(mol=mol, radius=2, nBits = pfpsize, useFeatures=False, useChirality=True)
        fp = np.empty(pfpsize,dtype = np.bool)
        DataStructs.ConvertToNumpyArray(fp_bit,fp)

    except Exception as e:
    	print("Cannot build product fp due to {}".format(e))
    	return
        
    pfp = fp

    return [pfp, rfp]

def process_one_record_reaxys(data):
	while True:
		try:
			(rxn_id, rxn_smiles, outcome) = data
			pfp_batch = []
			rfp_batch = []
			outcomes_batch = []
			rxn_id_batch = []
			rxn_smi_batch = []
			for i in range(len(rxn_id)):
				rct_smiles = rxn_smiles[i].split('>>')[0]
				prd_smiles = rxn_smiles[i].split('>>')[1]
				if '.' in prd_smiles:
					continue
				if rct_smiles is '' or prd_smiles is '':
					continue
				try:
					rct_mol = Chem.MolFromSmiles(rct_smiles)
					prd_mol = Chem.MolFromSmiles(prd_smiles)
				except:
					continue

				if rct_mol is None or prd_mol is None:
					continue

				# try:
				# 	[atom.ClearProp('molAtomMapNumber')for \
				# 			atom in rct_mol.GetAtoms() if atom.HasProp('molAtomMapNumber')]
				# 	[atom.ClearProp('molAtomMapNumber')for \
				# 			atom in prd_mol.GetAtoms() if atom.HasProp('molAtomMapNumber')]
				# except:
				# 	continue
				try:
					rct_smiles = Chem.MolToSmiles(rct_mol,isomericSmiles=True)
					prd_smiles = Chem.MolToSmiles(prd_mol,isomericSmiles=True)
					[pfp,rfp] = create_rxn_Morgan2FP(rct_smiles,prd_smiles)
				except:
					continue
				if pfp is None or rfp is None:
					continue
				pfp_batch.append(pfp)
				rfp_batch.append(rfp)
				outcomes_batch.append(outcome[i])
				rxn_id_batch.append(rxn_id[i])
				rxn_smi_batch.append(rxn_smiles[i])	
			pfp_mtx = sparse.csr_matrix(pfp_batch)
			rfp_mtx = sparse.csr_matrix(rfp_batch)

			print('made it')
			return (pfp_mtx, rfp_mtx, outcomes_batch, rxn_id_batch, rxn_smi_batch)

		except Exception as e:
			print('didnt make it because:\n')
			print(e)
			return (None,None,None,None,None)

print(datetime.datetime.now())

pfp_mtx_list = []
rfp_mtx_list = []
outcomes_list = []
rxn_id_list = []
rxn_smi_list = []

MINIMUM_MAXPUB_YEAR = 1940

rxn_id_batch = []
rxn_smiles_batch = []
outcome_batch = []
counter = 0
batch_size = 1000
df = pd.read_csv(input_file)
rxn_id_list_raw = list(df['rxn_id'])
rxn_smiles_list_raw = list(df['unmapped_smiles'])
outcomes_list_raw = list(df['label'])
def data_generator():
	global rxn_id_batch
	global rxn_smiles_batch
	global outcome_batch
	global counter
	global batch_size
	for _id in tqdm(range(len(rxn_id_list_raw))):
		counter += 1
		rxn_id = rxn_id_list_raw[_id]
		rxn_smiles = rxn_smiles_list_raw[_id]
		outcome = outcomes_list_raw[_id]
		rxn_id_batch.append(rxn_id)
		rxn_smiles_batch.append(rxn_smiles)
		outcome_batch.append(outcome)

		if counter%batch_size == 0:
			yield (rxn_id_batch, rxn_smiles_batch, outcome_batch)
			rxn_id_batch = []
			rxn_smiles_batch = []
			outcome_batch = []
	yield (rxn_id_batch, rxn_smiles_batch, outcome_batch)

generator = data_generator()
from joblib import Parallel, delayed
res = Parallel(n_jobs=16, verbose=5, pre_dispatch=500)(delayed(process_one_record_reaxys)(data) for data in generator)
print(len(res))

for pfp_mtx, rfp_mtx, outcomes_batch, rxn_id_batch, rxn_smi_batch in res:
	if(pfp_mtx is None): continue
	if(pfp_mtx.shape[1] == 0): continue
	pfp_mtx_list.append(pfp_mtx)
	rfp_mtx_list.append(rfp_mtx)
	outcomes_list.append(outcomes_batch)
	rxn_id_list.append(rxn_id_batch)			
	rxn_smi_list.append(rxn_smi_batch)

print(len(pfp_mtx_list),len(outcomes_list),len(rfp_mtx_list), len(rxn_id_list), len(rxn_smi_list))

index_list = list(range(len(pfp_mtx_list)))
# random.seed(70)
# random.shuffle(index_list)
pfp_mtx_list = [pfp_mtx_list[i] for i in index_list]
outcomes_list = [outcomes_list[i] for i in index_list]
rfp_mtx_list = [rfp_mtx_list[i] for i in index_list]
rxn_id_list = [rxn_id_list[i] for i in index_list]
rxm_smi_list = [rxn_smi_list[i] for i in index_list]

sparse_pfp_matrix = sparse.vstack(pfp_mtx_list)
pfp_mtx_list = []
sparse_rfp_matrix = sparse.vstack(rfp_mtx_list)
rfp_mtx_list = []
outcomes_list = list(itertools.chain.from_iterable(outcomes_list))
rxn_id_list = list(itertools.chain.from_iterable(rxn_id_list))
rxn_smi_list = list(itertools.chain.from_iterable(rxn_smi_list))

print(len(outcomes_list))
print(len(rxn_id_list))
print(len(rxn_smi_list))

print(index_list[0:100])
print(rxn_id_list[0:100])

print("SHAPE:", sparse_pfp_matrix.shape)
print("LENGTH OF OLIST:", len(outcomes_list))

PFP_FPATH = "./data/preprocessed/pfp_dataset_shuffle_reaxys.npz"
RFP_FPATH = "./data/preprocessed/rfp_dataset_shuffle_reaxys.npz"
BIN_FPATH = "./data/preprocessed/outcome_dataset_shuffle_reaxys.pickle"
RID_FPATH = "./data/preprocessed/rid_shuffle_reaxys.pickle"
RSMI_FPATH = "./data/preprocessed/rsmi_shuffle_reaxys.pickle"

sparse.save_npz(PFP_FPATH,sparse_pfp_matrix)
sparse.save_npz(RFP_FPATH,sparse_rfp_matrix)

with open(BIN_FPATH,"wb") as BIN:
	pickle.dump(outcomes_list,BIN)
with open(RID_FPATH,"wb") as RID:
	pickle.dump(rxn_id_list,RID)
with open(RSMI_FPATH,"wb") as RSMI:
	pickle.dump(rxn_smi_list,RSMI)

print(datetime.datetime.now())