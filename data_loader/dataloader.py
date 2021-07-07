import os
import torch
import numpy as np
import pandas as pd
from data_loader import transforms
import utils
from sklearn.model_selection import train_test_split

def data_split(data, test_size):
	x_train, x_test, y_train, y_test = train_test_split(data, data["label"], test_size=test_size)
	return x_train, x_test, y_train, y_test

def get_data_loader(cfg):
	collocation = None
	data = cfg["data"]["data_csv_name"]
	valid_data = cfg["data"]["validation_csv_name"]
	test_data = cfg["data"]["test_csv_name"]
	
	train_set = pd.read_csv(data)
	test_set = pd.read_csv(test_data)

	if (valid_data == ""):
		print("No validation set available, auto split the training into validation")
		print("Splitting dataset into train and valid....")
		split_ratio = float(cfg["data"]["validation_ratio"])
		train_set, valid_set, _ , _ = data_split(train_set, split_ratio)
		print("Done Splitting !!!")
	else:
		print("Creating validation set from file")
		print("Reading validation data from file: ", valid_data)
		valid_set = pd.read_csv(valid_data)
	
	# Get Custom Dataset inherit from torch.utils.data.Dataset
	dataset, module, _ = utils.general.get_attr_by_name(cfg["data"]["data.class"])
	# Create Dataset
	batch_size = int(cfg["data"]["batch_size"])
	train_set = dataset(train_set, transform = transforms.train_transform)
	valid_set = dataset(valid_set, transform = transforms.val_transform)
	test_set = dataset(test_set, transform = transforms.val_transform)

	# DataLoader
	train_loader = torch.utils.data.DataLoader(
		train_set, batch_size=batch_size, collate_fn=collocation, shuffle=True
    )
	valid_loader = torch.utils.data.DataLoader(
        valid_set, batch_size=batch_size, collate_fn=collocation, shuffle=False
    )
	test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=32, collate_fn=collocation, shuffle=False
    )
	
	## check Dataloader successfully
    # dataiter = iter(train_loader)
    # images, labels = dataiter.next()
    # print(images.shape, labels.shape)
	return train_loader, valid_loader, test_loader