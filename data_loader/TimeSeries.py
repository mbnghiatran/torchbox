import os
import torch
import random
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from sklearn.model_selection import train_test_split

# define data class for 1D CNN reading from csv
class TimeSeriesDataset(Dataset):
	
	def __init__(self,data,data_path,padding=True,training=True,normalize=True):
		self.data_path = data_path
		self.data = data
		self.training = training
		self.padding = padding
	
	def __getitem__(self,idx):
		# read data information
		data_path,abnormal,lead1,lead3,hr,severity = self.data.iloc[idx,:].values.tolist()
		# data = np.array(pd.read_csv(open(data_path,'r')), dtype = np.float).reshape(-1)
		data_3 = self.read_lead(lead3,"III")
		data_2 = self.read_lead(data_path,"II")
		# normalize
		# data = self.std_normalize(data)
		data_3 = self.std_normalize(data_3)
		data_2 = self.std_normalize(data_2)
		# padding if on
		# if (self.padding):
		# 	left = int(data.shape[0] * random.uniform(0,0.15))
		# 	right = int(data.shape[0] * random.uniform(0,0.15))
		# 	# data = np.pad(data,(left,right),'constant',constant_values=(0,0))
		
		# concat severity
		if(int(severity)==3):
				severity=2
		hr = int(hr)
		abnormal = int(abnormal)

		hr_tensor=torch.FloatTensor([hr])
		abnormal_tensor= torch.LongTensor([int(abnormal)])
		severity_tensor = torch.LongTensor([int(severity)])
		# choose which label to use
		label_tensor = abnormal_tensor
		data_tensor = torch.FloatTensor([data_2])
		data_3_tensor = torch.FloatTensor([data_3])

		return data_tensor,data_3_tensor,label_tensor

	def __len__(self):
		return len(self.data)
	
	def std_normalize(self, data):
		mean = np.mean(data)
		std = np.math.sqrt(np.mean(np.power((data - mean), 2)))
		return (data - mean) / std

	def read_lead(self,data_path,lead_num="III"):
		if ("VT" in data_path):
			data_path = os.path.join( "/data/Vien_Tim/D123_Short/"+"D"+lead_num+"_Short",data_path)
		else:
			data_path = os.path.join( "/data/Viet_Gia_Clinic/D123/"+"D"+lead_num,data_path)
		data = np.array(pd.read_csv(open(data_path,'r')), dtype = np.float).reshape(-1)
		return data
