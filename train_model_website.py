from __future__ import division

import keras
# from theano.tensor import lt,le,eq,gt,ge

import numpy as np#something
import datetime
import time
import os
import sys
import argparse

import h5py # needed for save_weights, fails otherwise

from keras import backend as K 
from keras.models import Sequential, Model, model_from_json
from keras.layers import Dense, Activation, Input
from keras.layers.core import Flatten, Permute, Reshape, Dropout, Lambda, RepeatVector
from keras.layers.wrappers import TimeDistributed
from keras.layers import merge
from keras import activations
from keras.layers.merge import Dot, Add
from keras.optimizers import SGD, Adam, Adadelta
from keras.layers.convolutional import Convolution1D, Convolution2D
from keras.regularizers import l2
from keras.utils.np_utils import to_categorical
from keras.utils.generic_utils import func_dump
from keras.utils.generic_utils import func_load
from keras.engine.topology import Layer
import pickle
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
import os
from sklearn.metrics import roc_auc_score
from rdkit import RDLogger
from scipy import sparse
import pandas as pd
from tqdm import tqdm
import random

lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)
####utilities
def set_keras_backend(backend):

    if K.backend() != backend:
        os.environ['KERAS_BACKEND'] = backend
        # reload(K)
        # assert K.backend() == backend

from sklearn.metrics import roc_auc_score
from keras.callbacks import Callback

class RocCallback(Callback):
    def __init__(self,training_data,validation_data):
        self.x = training_data[0]
        self.y = training_data[1]
        self.x_val = validation_data[0]
        self.y_val = validation_data[1]


    def on_train_begin(self, logs={}):
        return

    def on_train_end(self, logs={}):
        return

    def on_epoch_begin(self, epoch, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):
        y_pred_train = self.model.predict_proba(self.x)
        roc_train = roc_auc_score(self.y, y_pred_train)
        y_pred_val = self.model.predict_proba(self.x_val)
        roc_val = roc_auc_score(self.y_val, y_pred_val)
        print('\rroc-auc_train: %s - roc-auc_val: %s' % (str(round(roc_train,4)),str(round(roc_val,4))),end=100*' '+'\n')
        return

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        return



class Highway_self(Layer):

	def __init__(self, activation = 'elu',**kwargs):
		super(Highway_self, self).__init__(**kwargs)
		self.activation = activations.get(activation)
		self.transform_actv = activations.get('sigmoid')
		

	def build(self, input_shape):
		#weights of the dense layer
		self.kernel = self.add_weight(name = 'kernel',
								 shape = (input_shape[1],input_shape[1]),
								 initializer ='glorot_uniform',
								 trainable = True)
		self.bias = self.add_weight(name = 'bias',
								 shape = (input_shape[1],),
								 initializer ='zeros',
								 trainable = True)
		self.kernel_T = self.add_weight(name = 'kernel_T',
								 shape = (input_shape[1],input_shape[1]),
								 initializer ='glorot_uniform',
								 trainable = True)
		self.bias_T = self.add_weight(name = 'bias_T',
								 shape = (input_shape[1],),
								 initializer ='zeros',
								 trainable = True)
		self.input_dim = input_shape[1]
		super(Highway_self, self).build(input_shape)
	
	def call(self, x):
		transform_fun = self.activation(K.bias_add(K.dot(x,self.kernel), self.bias))
		transform_gate = self.transform_actv(K.bias_add(K.dot(x,self.kernel_T), self.bias_T))
		carry_gate = K.ones(self.input_dim,) - transform_gate
		output = transform_fun*transform_gate + x*carry_gate
		return output

	def compute_output_shape(self, input_shape):
		return (input_shape[0],input_shape[1])

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
	        fp_bit = AllChem.GetMorganFingerprintAsBitVect(mol=mol, radius=2, nBits = rxnfpsize, useFeatures=False)
	        fp = np.empty(rxnfpsize,dtype = 'int8')
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
        fp_bit = AllChem.GetMorganFingerprintAsBitVect(mol=mol, radius=2, nBits = pfpsize, useFeatures=False)
        fp = np.empty(pfpsize,dtype = 'int8')
        DataStructs.ConvertToNumpyArray(fp_bit,fp)

    except Exception as e:
    	print("Cannot build product fp due to {}".format(e))
    	return
        
    pfp = fp
    return [pfp, rfp]


#function for loading and partitioning data
def load_and_partition_data(pfp_csr_matrix, rfp_csr_matrix, outcomes_list, pfp_label_list, rxn_smi_list, split_ratio, batch_size):
	N_samples = len(outcomes_list)
	N_train = int(N_samples * split_ratio[0])
	N_val	= int(N_samples * split_ratio[1])
	N_test  = N_samples - N_train - N_val
	print('Total number of samples: {}'.format(N_samples))
	print('Training   on {}% - {}'.format(split_ratio[0]*100, N_train))
	print('Validating on {}% - {}'.format(split_ratio[1]*100, N_val))
	print('Testing    on {}% - {}'.format((1-split_ratio[1]-split_ratio[0])*100, N_test))

	return {
		'N_samples': N_samples,
		'N_train': N_train,
		#
		'train_generator': batch_data_generator(pfp_csr_matrix, rfp_csr_matrix, outcomes_list, 0, N_train, batch_size),
		'train_label_generator': batch_label_generator(pfp_label_list, outcomes_list, rxn_smi_list, 0, N_train, batch_size),
		'train_nb_samples': N_train,
		#
		'val_generator': batch_data_generator(pfp_csr_matrix, rfp_csr_matrix, outcomes_list, N_train, N_train + N_val, batch_size),
		'val_label_generator': batch_label_generator(pfp_label_list, outcomes_list, rxn_smi_list, N_train, N_train + N_val, batch_size),
		'val_nb_samples': N_val,
		#
		'test_generator': batch_data_generator(pfp_csr_matrix, rfp_csr_matrix, outcomes_list, N_train + N_val, N_samples, batch_size),
		'test_label_generator': batch_label_generator(pfp_label_list, outcomes_list, rxn_smi_list, N_train + N_val, N_samples, batch_size),
		'test_nb_samples': N_test,
		#
		#
		'batch_size': batch_size,
	}


#batch data generator
def batch_data_generator( pfp_csr_matrix, rfp_csr_matrix, outcomes_list, start_at, end_at, batch_size):
	while True:
		for start_index in range(start_at, end_at,batch_size):
			end_index = min(start_index + batch_size, end_at)

			y_train_batch = outcomes_list[start_index:end_index]
			pfp_matrix = pfp_csr_matrix[start_index:end_index,:].todense()
			rfp_matrix = rfp_csr_matrix[start_index:end_index,:].todense()
			
			pfp_train_batch = np.asarray(pfp_matrix,dtype = 'float32')
			rfp_train_batch = np.asarray(rfp_matrix,dtype = 'float32')

			rxnfp_train_batch = pfp_train_batch - rfp_train_batch
			y_train_batch = np.asarray(y_train_batch,dtype = 'float32')
			
			yield ([pfp_train_batch, rxnfp_train_batch],y_train_batch)

def batch_label_generator(pfp_label_list, outcomes_list, rxn_smi_list, start_at, end_at, batch_size):
	while True:
		for start_index in range(start_at, end_at,batch_size):
			end_index = min(start_index + batch_size, end_at)

			rxn_id = []
			rxn_true = []
			rxn_smi = []
			print(len(pfp_label_list),len(outcomes_list), len(rxn_smi_list))
			for i in range(start_index,end_index):
				rxn_id.append(pfp_label_list[i])
				rxn_true.append(outcomes_list[i])
				rxn_smi.append(rxn_smi_list[i])
				
			yield (rxn_id, rxn_true, rxn_smi)

def multiple_batch_data_generator(data_generator_list):
	while True:
		pfp_train_batch = []
		rxnfp_train_batch = []
		y_train_batch = []
		for i in range(len(data_generator_list)):
			(x,y) = next(data_generator_list[i])
			if pfp_train_batch == []:
				pfp_train_batch = x[0]
			else:
				pfp_train_batch = np.append(pfp_train_batch,x[0],0)
			if rxnfp_train_batch == []:
				rxnfp_train_batch = x[1]
			else:
				rxnfp_train_batch = np.append(rxnfp_train_batch,x[1],0)
			if y_train_batch == []:
				y_train_batch = y
			else:
				y_train_batch = np.append(y_train_batch,y)
		yield ([pfp_train_batch,rxnfp_train_batch],y_train_batch)

def multiple_batch_label_generator(label_generator_list):
	while True:
		batch_label = [[],[],[]]
		for i in range(len(label_generator_list)):
			(x,y,z) = next(label_generator_list[i])
			batch_label[0] = np.append(batch_label[0],x,0)
			batch_label[1] = np.append(batch_label[1],y,0)
			batch_label[2] = np.append(batch_label[2],z,0)
		yield batch_label
#build model structure

# def pos_ct(y_true, y_pred):
# 	pos_pred = K.sum(gt((K.clip(y_pred, 0, 1)),0.5))
# 	return pos_pred
# def true_pos(y_true, y_pred):
# 	true_pos_ct = K.sum(gt((K.clip(y_pred*y_true, 0, 1)),0.5))
# 	return true_pos_ct
# def real_pos(y_true, y_pred):
# 	real_pos_ct = K.sum(gt((K.clip(y_true, 0, 1)),0.5))
# 	return real_pos_ct

def build(pfp_len = 2048, rxnfp_len = 2048,l2v = 0.01):
	input_pfp = Input(shape = (pfp_len,))
	input_rxnfp = Input(shape = (rxnfp_len,))
	
	input_pfp_h1 = Dense(1024, activation = 'elu')(input_pfp)
	input_pfp_h2 = Dropout(0.3)(input_pfp_h1)
	input_pfp_h3 = Highway_self(activation = 'elu')(input_pfp_h2)
	input_pfp_h4 = Highway_self(activation = 'elu')(input_pfp_h3)
	input_pfp_h5 = Highway_self(activation = 'elu')(input_pfp_h4)
	input_pfp_h6 = Highway_self(activation = 'elu')(input_pfp_h5)
	input_pfp_h7 = Highway_self(activation = 'elu')(input_pfp_h6)

	input_rxnfp_h1 = Dense(1024, activation = 'elu')(input_rxnfp)
	merged_h1 = Dot(axes = 1, normalize=False)([input_pfp_h7,input_rxnfp_h1])
	
	output= Dense(1, activation = 'sigmoid')(merged_h1)
	model = Model([input_pfp,input_rxnfp],output)

	model.count_params()
	model.summary()
	
	adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

	model.compile(loss='binary_crossentropy', optimizer='adam', metrics = ['acc',keras.metrics.Precision(name='precision'),keras.metrics.Recall(name='recall')])
	return model

#train model
def train_bin_class(model, pfp_csr_matrices, rfp_csr_matrices, outcomes_lists, pfp_label_lists, rxn_smi_lists, split_ratio, class_weight, batch_size):
	train_generator_list = []
	val_generator_list = []
	train_nb_samples = 0
	val_nb_samples = 0
	data = [0]*len(outcomes_lists)
	print(len(data))
	batch_size_total = 0
	for i in range(len(outcomes_lists)):
		data[i] = load_and_partition_data(pfp_csr_matrices[i], rfp_csr_matrices[i], outcomes_lists[i], pfp_label_lists[i], rxn_smi_lists[i], split_ratio, batch_size[i])
		train_generator_list.append(data[i]['train_generator'])
		val_generator_list.append(data[i]['val_generator'])
		train_nb_samples += data[i]['train_nb_samples']
		val_nb_samples += data[i]['val_nb_samples']
		batch_size_total += data[i]['batch_size']
	train_generator = multiple_batch_data_generator(train_generator_list)
	val_generator = multiple_batch_data_generator(val_generator_list)

	##this need to be updated later
	# roc = RocCallback(training_data=(X_train, y_train),
 #                  validation_data=(X_test, y_test))


	from keras.callbacks import ModelCheckpoint, CSVLogger, EarlyStopping, ReduceLROnPlateau
	reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.2,
              patience=2, min_lr=0.0001)
	callbacks = [
		EarlyStopping(patience = 5),
		CSVLogger('training.log'),
		reduce_lr,
		]

	try:
		hist = model.fit_generator(train_generator, 
			verbose = 1,
			validation_data = val_generator,
			steps_per_epoch = np.ceil(train_nb_samples/batch_size_total),
			epochs = nb_epoch, 
			callbacks = callbacks,
			validation_steps = np.ceil(val_nb_samples/batch_size_total),
			class_weight = class_weight,
		)

	except KeyboardInterrupt:
		print('Stopped training early!')

if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('--nb_epoch', type = int, default = 100,
						help = 'Number of epochs to train for, default 100')
	parser.add_argument('--batch_size', type = int, default = 20,
						help = 'Batch size, default 20')
	
	args = parser.parse_args()

	nb_epoch           = int(args.nb_epoch)
	batch_size_input   = int(args.batch_size)

	SPLIT_RATIO = (0.8, 0.1)
	# set_keras_backend("theano")
	model = build(pfp_len = 2048, rxnfp_len = 2048, l2v = 1e-9)
	
	print("model built")

	rid_lists = []
	rxn_smi_lists = []
	outcomes_lists = []
	pfp_csr_matrices = []
	rfp_csr_matrices = []

	PFP1_FPATH = "./data/preprocessed/pfp_dataset_shuffle_reaxys.npz"
	RFP1_FPATH = "./data/preprocessed/rfp_dataset_shuffle_reaxys.npz"
	RID1_FPATH = "./data/preprocessed/rid_shuffle_reaxys.pickle"
	BIN1_FPATH = "./data/preprocessed/outcome_dataset_shuffle_reaxys.pickle"
	RXN1_FPATH = "./data/preprocessed/rsmi_shuffle_reaxys.pickle"

	with open(RID1_FPATH,"rb") as RLB:
		rid_lists.append(pickle.load(RLB))
	with open(RXN1_FPATH,"rb") as RXN:
		rxn_smi_lists.append(pickle.load(RXN))
	with open(BIN1_FPATH,"rb") as BIN:
		outcomes_lists.append(pickle.load(BIN))
	print(outcomes_lists[0][:100])
	pfp_csr_matrices.append(sparse.load_npz(PFP1_FPATH))
	rfp_csr_matrices.append(sparse.load_npz(RFP1_FPATH))

	# PFP2_FPATH = "/data/chuang7/preprocessed/pfp_dataset_shuffle.npz"
	# RFP2_FPATH = "/data/chuang7/preprocessed/rfp_dataset_shuffle.npz"
	# RID2_FPATH = "/data/chuang7/preprocessed/rid_shuffle.pickle"
	# BIN2_FPATH = "/data/chuang7/preprocessed/outcome_dataset_shuffle.pickle"
	# RXN2_FPATH = "/data/chuang7/preprocessed/rsmi_shuffle.pickle"

	# with open(RID2_FPATH,"rb") as RLB:
	# 	rid_lists.append(pickle.load(RLB))
	# with open(RXN2_FPATH,"rb") as RXN:
	# 	rxn_smi_lists.append(pickle.load(RXN))
	# with open(BIN2_FPATH,"rb") as BIN:
	# 	outcomes_lists.append(pickle.load(BIN)) 
	# pfp_csr_matrices.append(sparse.load_npz(PFP2_FPATH))
	# rfp_csr_matrices.append(sparse.load_npz(RFP2_FPATH))

	# PFP3_FPATH = "/data/chuang7/preprocessed/pfp_dataset_shuffle_2.npz"
	# RFP3_FPATH = "/data/chuang7/preprocessed/rfp_dataset_shuffle_2.npz"
	# RID3_FPATH = "/data/chuang7/preprocessed/rid_shuffle_2.pickle"
	# BIN3_FPATH = "/data/chuang7/preprocessed/outcome_dataset_shuffle_2.pickle"
	# RXN3_FPATH = "/data/chuang7/preprocessed/rsmi_shuffle_2.pickle"

	# with open(RID3_FPATH,"rb") as RLB:
	# 	rid_lists.append(pickle.load(RLB))
	# with open(RXN3_FPATH,"rb") as RXN:
	# 	rxn_smi_lists.append(pickle.load(RXN))
	# with open(BIN3_FPATH,"rb") as BIN:
	# 	outcomes_lists.append(pickle.load(BIN)) 
	# pfp_csr_matrices.append(sparse.load_npz(PFP3_FPATH))
	# rfp_csr_matrices.append(sparse.load_npz(RFP3_FPATH))

	print("raw data loaded, converting rfp matrix to dense...\n")

	nb_sample_1 = len(outcomes_lists[0])
	# nb_sample_2 = len(outcomes_lists[1])
	# nb_sample_3 = len(outcomes_lists[2])

	batch_size = [0]
	batch_size[0] = batch_size_input

	# print(batch_size)
	print(len(outcomes_lists),len(pfp_csr_matrices),len(rfp_csr_matrices),len(rid_lists),len(rxn_smi_lists))
	print(len(outcomes_lists[0]),len(rid_lists[0]),len(rxn_smi_lists[0]))
	# print(len(outcomes_lists[1]),len(rid_lists[1]),len(rxn_smi_lists[1]))
	# print(len(outcomes_lists[2]),len(rid_lists[2]),len(rxn_smi_lists[2]))
	print("raw data loaded, converting rfp matrix to dense...\n")
	print(datetime.datetime.now())
	class_weight = None

	train_bin_class(model, pfp_csr_matrices, rfp_csr_matrices, outcomes_lists, rid_lists, rxn_smi_lists, SPLIT_RATIO, class_weight, batch_size)
	model.save('./model/my_model_shuffle.h5')
