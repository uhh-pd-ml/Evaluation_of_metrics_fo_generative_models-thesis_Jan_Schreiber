import wandb
import os
import numpy as np
import matplotlib.pyplot as plt
import sys
import sklearn
import sklearn.metrics as metrics
import scipy.stats as statas
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from datetime import date,datetime,time
import shutil
from data_loader import HDF5Dataset, BatchLoader, BatchLoaderTwoFiles, BatchLoaderTwoFilesFixSplitTrain

from itertools import cycle
from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import roc_auc_score

total_start = datetime.now()


n_split1 = 255000
n_split2 = 255000
n_gesamt = n_split1 + n_split2
training_percentage = 80
validation_percentage = 10
test_percentage = 10


# Define Parameters - Convolutional NN

params_convolutional = {
        
        "model" : "convolutional 1,0Sigma",

        # IO
        "input_path"  : '/beegfs/desy/user/diefenbs/shower_data/pion_uniform_510k_PunchThroughCut70.hdf5',
        "file_path1" : '/beegfs/desy/user/schreibj/Hadron_classifier/shower_data/pion_510k_orig_merged.hdf5',
        "file_path2" : '/beegfs/desy/user/schreibj/Hadron_classifier/shower_data/pion_510k_trafo_1,0Sigma_merged.hdf5',
        "output_path" : '/beegfs/desy/user/schreibj/Hadron_classifier/results/',

        "batch_size" : 128, #256 baseline
        "batch_size2" : 128,
        "epochs" : 30,
        "train_size" : n_split1,
        'shuffle': True,
        'lr_convolutional' : 0.001,
        "wandb_dir" : '/beegfs/desy/user/schreibj/wandb/Hadron_classifier/',
    }

# Define a project name for weights and biases - online tracking:
script_save_path = "/home/schreibj/Hadron_classifier/v2_convolutional_error_run_1,0Sigma_Classifier_Hadron_showers.py"
project_name_wandb = "v2_error_run_convolutional_Classifier_Hadron_showers"
folder_name = "v2_error_run"
date = "02.20.23"

# Define if data should be logged via weights & biases
wandb_logging = True

# Create a directory for all plots, scripts and model data:
path = f"/beegfs/desy/user/schreibj/Hadron_classifier/Savings_complete/{folder_name}/{project_name_wandb}_{date}_savings_complete"

os.makedirs(path, exist_ok = True)



# Make sure we have a GPU allocated
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
use_cuda = torch.cuda.is_available()
print(device) # The output should be 'cuda'


# Now we start with the Neural Network

# Defining the Models

class NeuralNet_convolutional(nn.Module):
    def __init__(self):
        super(NeuralNet_convolutional, self).__init__()
        
        self.conv1 = nn.Conv3d(1, 32, 5, 1)
        self.conv2 = nn.Conv3d(32, 64, 3, 1)
        self.conv3 = nn.Conv3d(64, 64, 3, 1)
               
        self.fc1 = nn.Linear(2048, 512) # adjust input size like this x.shape[1]*x.shape[2]*x.shape[3]*x.shape[4]
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 1)
        
        
        
    def forward(self, x):
        
        leakyRelu_slope = 0.1
        print(x.shape)
        x = x.view(x.shape[0],1,48,25,25)
        
        x = self.conv1(x)
        x = F.leaky_relu(x,leakyRelu_slope)
        x = F.max_pool3d(x, 2)
        x = self.conv2(x)
        x = F.leaky_relu(x,leakyRelu_slope)
        x = F.max_pool3d(x, 2)
        x = self.conv3(x)
        x = F.leaky_relu(x,leakyRelu_slope)
        
        print(x.shape)
        x = x.view(-1,x.shape[1]*x.shape[2]*x.shape[3]*x.shape[4])
        x = self.fc1(x)
        x = F.leaky_relu(x,leakyRelu_slope)
        x = self.fc2(x)
        x = F.leaky_relu(x,leakyRelu_slope)
        x = self.fc3(x)
        x = F.leaky_relu(x,leakyRelu_slope)
        x = self.fc4(x)
        
        return x
    

# Define Transformations

def threshold(x, threshmin, newval=0.0):
    x[x < threshmin] = newval
    return x

def tf_core25_thresh(x):
    if len(x.shape) == 4:
        return threshold(x[:, :, 13:38, 11:36], threshmin=0.25, newval=0.0)
    else: 
        return threshold(x[:, 13:38, 11:36], threshmin=0.25, newval=0.0)


# SWITCH BETWEEN MODELS - Convolutional Network    
    
# Loss function
criterion1 = nn.BCEWithLogitsLoss()  # when no sigmoid in last layer - sigmoid is integrated
criterion2 = nn.BCEWithLogitsLoss()
criterion3 = nn.BCEWithLogitsLoss()
criterion4 = nn.BCEWithLogitsLoss()
criterion5 = nn.BCEWithLogitsLoss()

# Create network object
model1 = NeuralNet_convolutional().to(device)
model2 = NeuralNet_convolutional().to(device)
model3 = NeuralNet_convolutional().to(device)
model4 = NeuralNet_convolutional().to(device)
model5 = NeuralNet_convolutional().to(device)

# Create optimizer object - Adam optimiser
optimizer1 = torch.optim.Adam(model1.parameters(),lr=params_convolutional["lr_convolutional"])
optimizer2 = torch.optim.Adam(model2.parameters(),lr=params_convolutional["lr_convolutional"])
optimizer3 = torch.optim.Adam(model3.parameters(),lr=params_convolutional["lr_convolutional"])
optimizer4 = torch.optim.Adam(model4.parameters(),lr=params_convolutional["lr_convolutional"])
optimizer5 = torch.optim.Adam(model5.parameters(),lr=params_convolutional["lr_convolutional"])

# Loading a pretrained Convolutional Network


print(f'train_size: {params_convolutional["train_size"]}')
print(f'train_size2: {params_convolutional["train_size2"]}')
print(f'validation_size: {params_convolutional["validation_size"]}')
print(f'validation_size2: {params_convolutional["validation_size2"]}')
print(f'test_size: {params_convolutional["test_size"]}')
print(f'test_size2: {params_convolutional["test_size2"]}')




train_loader = BatchLoaderTwoFilesFixSplitTrain(file_path1 = params_convolutional["file_path1"],
                                   file_path2 = params_convolutional["file_path2"],
                                   train_size = params_convolutional["train_size"],
                                   batch_size = params_convolutional["batch_size"], 
                                   shuffle=True, 
                                   transform=tf_core25_thresh,
                                   mode = 'train')

validation_loader = BatchLoaderTwoFilesFixSplitTrain(file_path1 = params_convolutional["file_path1"],
                                   file_path2 = params_convolutional["file_path2"],
                                   train_size = params_convolutional["train_size"], 
                                   batch_size = params_convolutional["batch_size"], 
                                   shuffle=True, 
                                   transform=tf_core25_thresh,
                                   mode = 'val')

test_loader = BatchLoaderTwoFilesFixSplitTrain(file_path1 = params_convolutional["file_path1"],
                                   file_path2 = params_convolutional["file_path2"],
                                   train_size = params_convolutional["train_size"], 
                                   batch_size = params_convolutional["batch_size"], 
                                   shuffle=True, 
                                   transform=tf_core25_thresh,
                                   mode = 'test' )


# Check whether actually loaded number of batches fits the expected amount of batches

print(f'number of train batches: {int(params_convolutional["train_size"]/params_convolutional["batch_size"])}')
print(f'number of validation batches: {int(params_convolutional["validation_size"]/params_convolutional["batch_size"])}')
print(f'number of test batches: {int(params_convolutional["test_size"]/params_convolutional["batch_size"])}')


# Train Convolutional network

start_convolutional = datetime.now()

# Keep track of the accuracies
train_accs_convolutional_1 = np.array([])
validation_accs_convolutional_1 = np.array([])
train_losses_convolutional_batchwise_1 = np.array([])
train_losses_convolutional_1 = np.array([])
validation_losses_convolutional_1 = np.array([])

train_accs_convolutional_2 = np.array([])
validation_accs_convolutional_2 = np.array([])
train_losses_convolutional_batchwise_2 = np.array([])
train_losses_convolutional_2 = np.array([])
validation_losses_convolutional_2 = np.array([])

train_accs_convolutional_3 = np.array([])
validation_accs_convolutional_3 = np.array([])
train_losses_convolutional_batchwise_3 = np.array([])
train_losses_convolutional_3 = np.array([])
validation_losses_convolutional_3 = np.array([])

train_accs_convolutional_4 = np.array([])
validation_accs_convolutional_4 = np.array([])
train_losses_convolutional_batchwise_4 = np.array([])
train_losses_convolutional_4 = np.array([])
validation_losses_convolutional_4 = np.array([])

train_accs_convolutional_5 = np.array([])
validation_accs_convolutional_5 = np.array([])
train_losses_convolutional_batchwise_5 = np.array([])
train_losses_convolutional_5 = np.array([])
validation_losses_convolutional_5 = np.array([])


# Initializing a counter for setting a limit to the training procedure
batch_counter = 0
epoch_counter = 0

n_batches_train_total = int(params_convolutional["train_size"]/params_convolutional["batch_size"])
n_batches_validation_total = int(params_convolutional["validation_size"]/params_convolutional["batch_size"]) 
n_batches_test_total = int(params_convolutional["test_size"]/params_convolutional["batch_size"]) 


# Inicialise weights and biases
if wandb_logging == True:
    wandb.init(
        project = project_name_wandb,
        dir=params_convolutional['wandb_dir'],
        config = params_convolutional,
        name = params_convolutional["model"])


# Training/Evaluation Loop

for epoch_idx in range(params_convolutional["epochs"]):
    
    start_epoch_time = datetime.now()
    
    # Counting empty batches 
    num_empty_batches_train_model1 = 0 
    num_empty_batches_train_model2 = 0
    num_empty_batches_train_model3 = 0
    num_empty_batches_train_model4 = 0
    num_empty_batches_train_model5 = 0

    num_empty_batches_validation_model1 = 0 
    num_empty_batches_validation_model2 = 0
    num_empty_batches_validation_model3 = 0
    num_empty_batches_validation_model4 = 0
    num_empty_batches_validation_model5 = 0
    
    # Calculate predictions
    train_acc_convolutional_1 = 0
    validation_acc_convolutional_1 = 0
    train_loss_convolutional_batchwise_1 = 0
    train_loss_convolutional_1 = 0
    validation_loss_convolutional_1 = 0 
    
    train_acc_convolutional_2 = 0
    validation_acc_convolutional_2 = 0
    train_loss_convolutional_batchwise_2 = 0
    train_loss_convolutional_2 = 0
    validation_loss_convolutional_2 = 0
    
    train_acc_convolutional_3 = 0
    validation_acc_convolutional_3 = 0
    train_loss_convolutional_batchwise_3 = 0
    train_loss_convolutional_3 = 0
    validation_loss_convolutional_3 = 0
    
    train_acc_convolutional_4 = 0
    validation_acc_convolutional_4 = 0
    train_loss_convolutional_batchwise_4 = 0
    train_loss_convolutional_4 = 0
    validation_loss_convolutional_4 = 0
    
    train_acc_convolutional_5 = 0
    validation_acc_convolutional_5 = 0
    train_loss_convolutional_batchwise_5 = 0
    train_loss_convolutional_5 = 0
    validation_loss_convolutional_5 = 0
    
    
    #LOADING THE TRAININGDATA
    for batch_idx, (data, energy, label, ind_outs) in enumerate(train_loader):
        
        # Check for empty batches. Empty batches will be skipped
        skip_model1 = 0
        skip_model2 = 0
        skip_model3 = 0
        skip_model4 = 0
        skip_model5 = 0
        
        if data[0].shape[0] == 0: skip_model1 = 1
        if data[1].shape[0] == 0: skip_model2 = 1
        if data[2].shape[0] == 0: skip_model3 = 1
        if data[3].shape[0] == 0: skip_model4 = 1
        if data[4].shape[0] == 0: skip_model5 = 1
        
        #Count empty batches
        if skip_model1 == 1: num_empty_batches_train_model1 += 1 
        if skip_model2 == 1: num_empty_batches_train_model2 += 1
        if skip_model3 == 1: num_empty_batches_train_model3 += 1
        if skip_model4 == 1: num_empty_batches_train_model4 += 1
        if skip_model5 == 1: num_empty_batches_train_model5 += 1
        
        # Testing validity of data:
        assert data[0].shape[0] == label[0].shape[0]
        
        #TRAINING THE MODEL: 
        
        if skip_model1 == 0:model1.train()
        if skip_model2 == 0:model2.train()
        if skip_model3 == 0:model3.train()
        if skip_model4 == 0:model4.train()
        if skip_model5 == 0:model5.train()
        
        # Reset gradient
        optimizer1.zero_grad()
        optimizer2.zero_grad()
        optimizer3.zero_grad()
        optimizer4.zero_grad()
        optimizer5.zero_grad()
        
        # Apply the network
        net_out1 = model1(data[0].to(device))
        net_out2 = model2(data[1].to(device))
        net_out3 = model3(data[2].to(device))
        net_out4 = model4(data[3].to(device))
        net_out5 = model5(data[4].to(device))
        
        # Calculate the loss function
        label1 = label[0].clone().detach().to(device)
        label2 = label[1].clone().detach().to(device)
        label3 = label[2].clone().detach().to(device)
        label4 = label[3].clone().detach().to(device)
        label5 = label[4].clone().detach().to(device)
        
        loss1 = criterion1(net_out1,label1)
        loss2 = criterion2(net_out2,label2)
        loss3 = criterion3(net_out3,label3)
        loss4 = criterion4(net_out4,label4)
        loss5 = criterion5(net_out5,label5)
        
        
        # Evaluating the training loss batchwise for the first epoch - Troubleshooting
        if epoch_counter < 3:
            train_loss_convolutional_batchwise_1 = loss1.item()
            if skip_model1 == 0:train_losses_convolutional_batchwise_1 = np.append(train_losses_convolutional_batchwise_1,train_loss_convolutional_batchwise_1)
            train_loss_convolutional_batchwise_2 = loss2.item()
            if skip_model2 == 0:train_losses_convolutional_batchwise_2 = np.append(train_losses_convolutional_batchwise_2,train_loss_convolutional_batchwise_2)
            train_loss_convolutional_batchwise_3 = loss3.item()
            if skip_model3 == 0:train_losses_convolutional_batchwise_3 = np.append(train_losses_convolutional_batchwise_3,train_loss_convolutional_batchwise_3)
            train_loss_convolutional_batchwise_4 = loss4.item()
            if skip_model4 == 0:train_losses_convolutional_batchwise_4 = np.append(train_losses_convolutional_batchwise_4,train_loss_convolutional_batchwise_4)
            train_loss_convolutional_batchwise_5 = loss5.item()
            if skip_model5 == 0:train_losses_convolutional_batchwise_5 = np.append(train_losses_convolutional_batchwise_5,train_loss_convolutional_batchwise_5)
            
        # Keeping track of the training loss
        if skip_model1 == 0:train_loss_convolutional_1 += loss1.item()
        if skip_model2 == 0:train_loss_convolutional_2 += loss2.item()
        if skip_model3 == 0:train_loss_convolutional_3 += loss3.item()
        if skip_model4 == 0:train_loss_convolutional_4 += loss4.item()
        if skip_model5 == 0:train_loss_convolutional_5 += loss5.item()
        
        
        # Calculate the gradients
        if skip_model1 == 0:loss1.backward()
        if skip_model2 == 0:loss2.backward()
        if skip_model3 == 0:loss3.backward()
        if skip_model4 == 0:loss4.backward()
        if skip_model5 == 0:loss5.backward()
        
        
        # Update the weights
        if skip_model1 == 0:optimizer1.step()
        if skip_model2 == 0:optimizer2.step()
        if skip_model3 == 0:optimizer3.step()
        if skip_model4 == 0:optimizer4.step()
        if skip_model5 == 0:optimizer5.step()
        
        
        # Keeping track of number of batches in one epoch:
        batch_counter += 1
    
        print(f"Convolutional Network Epoch: batch number: {batch_counter},epoch number: {epoch_counter}")
        
        
        # 1. Evaluation on Trainingdata during Trainingloop:
        model1.eval()
        model2.eval()
        model3.eval()
        model4.eval()
        model5.eval()
        
        print(f'Evaluation on Trainingdata batch_idx,{batch_idx}')
        
        with torch.no_grad(): 
            if skip_model1 == 0:label_pred_train_1 = torch.round(torch.sigmoid(net_out1.detach().cpu()))
            if skip_model2 == 0:label_pred_train_2 = torch.round(torch.sigmoid(net_out2.detach().cpu()))
            if skip_model3 == 0:label_pred_train_3 = torch.round(torch.sigmoid(net_out3.detach().cpu()))
            if skip_model4 == 0:label_pred_train_4 = torch.round(torch.sigmoid(net_out4.detach().cpu()))
            if skip_model5 == 0:label_pred_train_5 = torch.round(torch.sigmoid(net_out5.detach().cpu()))
    
        if skip_model1 == 0:train_acc_convolutional_1 += sum(label1.detach().cpu() == label_pred_train_1) / label1.shape[0]
        if skip_model2 == 0:train_acc_convolutional_2 += sum(label2.detach().cpu() == label_pred_train_2) / label2.shape[0]
        if skip_model3 == 0:train_acc_convolutional_3 += sum(label3.detach().cpu() == label_pred_train_3) / label3.shape[0]
        if skip_model4 == 0:train_acc_convolutional_4 += sum(label4.detach().cpu() == label_pred_train_4) / label4.shape[0]
        if skip_model5 == 0:train_acc_convolutional_5 += sum(label5.detach().cpu() == label_pred_train_5) / label5.shape[0]
    
        
    # Calculating the medium accuracy and medium loss on the training data after each epoch and logging into weights and biases
    train_acc_convolutional_1 /= (n_batches_train_total - num_empty_batches_train_model1)
    if wandb_logging == True:
        wandb.log({'epoch': epoch_counter, 'train_acc_convolutional_1': train_acc_convolutional_1})
    train_loss_convolutional_1 /= (n_batches_train_total - num_empty_batches_train_model1)
    if wandb_logging == True:
        wandb.log({'epoch': epoch_counter, 'train_loss_convolutional_1': train_loss_convolutional_1})
    train_acc_convolutional_2 /= (n_batches_train_total - num_empty_batches_train_model2)
    if wandb_logging == True:
        wandb.log({'epoch': epoch_counter, 'train_acc_convolutional_2': train_acc_convolutional_2})
    train_loss_convolutional_2 /= (n_batches_train_total - num_empty_batches_train_model2)
    if wandb_logging == True:
        wandb.log({'epoch': epoch_counter, 'train_loss_convolutional_2': train_loss_convolutional_2})
    train_acc_convolutional_3 /= (n_batches_train_total - num_empty_batches_train_model3)
    if wandb_logging == True:
        wandb.log({'epoch': epoch_counter, 'train_acc_convolutional_3': train_acc_convolutional_3})
    train_loss_convolutional_3 /= (n_batches_train_total - num_empty_batches_train_model3)
    if wandb_logging == True:
        wandb.log({'epoch': epoch_counter, 'train_loss_convolutional_3': train_loss_convolutional_3})
    train_acc_convolutional_4 /= (n_batches_train_total - num_empty_batches_train_model4)
    if wandb_logging == True:
        wandb.log({'epoch': epoch_counter, 'train_acc_convolutional_4': train_acc_convolutional_4})
    train_loss_convolutional_4 /= (n_batches_train_total - num_empty_batches_train_model4)
    if wandb_logging == True:
        wandb.log({'epoch': epoch_counter, 'train_loss_convolutional_4': train_loss_convolutional_4})
    train_acc_convolutional_5 /= (n_batches_train_total - num_empty_batches_train_model5)
    if wandb_logging == True:
        wandb.log({'epoch': epoch_counter, 'train_acc_convolutional_5': train_acc_convolutional_5})
    train_loss_convolutional_5 /= (n_batches_train_total - num_empty_batches_train_model5)
    if wandb_logging == True:
        wandb.log({'epoch': epoch_counter, 'train_loss_convolutional_5': train_loss_convolutional_5})
    
    # print some information
    print("Model 1 - Convolutional Network Epoch:",epoch_counter, "Train Accuracy 1 :", train_acc_convolutional_1, "Train Loss 1 :", train_loss_convolutional_1, "# Empty batches :", num_empty_batches_train_model1)
    print("Model 2 - Convolutional Network Epoch:",epoch_counter, "Train Accuracy 2 :", train_acc_convolutional_2, "Train Loss 2 :", train_loss_convolutional_2, "# Empty batches :", num_empty_batches_train_model2)
    print("Model 3 - Convolutional Network Epoch:",epoch_counter, "Train Accuracy 3 :", train_acc_convolutional_3, "Train Loss 3 :", train_loss_convolutional_3, "# Empty batches :", num_empty_batches_train_model3)
    print("Model 4 - Convolutional Network Epoch:",epoch_counter, "Train Accuracy 4 :", train_acc_convolutional_4, "Train Loss 4 :", train_loss_convolutional_4, "# Empty batches :", num_empty_batches_train_model4)
    print("Model 5 - Convolutional Network Epoch:",epoch_counter, "Train Accuracy 5 :", train_acc_convolutional_5, "Train Loss 5 :", train_loss_convolutional_5, "# Empty batches :", num_empty_batches_train_model5)
    
              
    # and store the accuracy and loss for later use
    train_accs_convolutional_1 = np.append(train_accs_convolutional_1,train_acc_convolutional_1)
    train_losses_convolutional_1 = np.append(train_losses_convolutional_1,train_loss_convolutional_1)
    train_accs_convolutional_2 = np.append(train_accs_convolutional_2,train_acc_convolutional_2)
    train_losses_convolutional_2 = np.append(train_losses_convolutional_2,train_loss_convolutional_2)
    train_accs_convolutional_3 = np.append(train_accs_convolutional_3,train_acc_convolutional_3)
    train_losses_convolutional_3 = np.append(train_losses_convolutional_3,train_loss_convolutional_3)
    train_accs_convolutional_4 = np.append(train_accs_convolutional_4,train_acc_convolutional_4)
    train_losses_convolutional_4 = np.append(train_losses_convolutional_4,train_loss_convolutional_4)
    train_accs_convolutional_5 = np.append(train_accs_convolutional_5,train_acc_convolutional_5)
    train_losses_convolutional_5 = np.append(train_losses_convolutional_5,train_loss_convolutional_5)
    np.savez_compressed(f"/beegfs/desy/user/schreibj/Hadron_classifier/Savings_complete/{folder_name}/{os.path.basename(path)}/train_accs_convolutional_1_{date}.npz",train_accs_convolutional_1=train_accs_convolutional_1)
    np.savez_compressed(f"/beegfs/desy/user/schreibj/Hadron_classifier/Savings_complete/{folder_name}/{os.path.basename(path)}/train_losses_convolutional_1_{date}.npz",train_losses_convolutional_1=train_losses_convolutional_1)
    np.savez_compressed(f"/beegfs/desy/user/schreibj/Hadron_classifier/Savings_complete/{folder_name}/{os.path.basename(path)}/train_accs_convolutional_2_{date}.npz",train_accs_convolutional_1=train_accs_convolutional_2)
    np.savez_compressed(f"/beegfs/desy/user/schreibj/Hadron_classifier/Savings_complete/{folder_name}/{os.path.basename(path)}/train_losses_convolutional_2_{date}.npz",train_losses_convolutional_1=train_losses_convolutional_2)
    np.savez_compressed(f"/beegfs/desy/user/schreibj/Hadron_classifier/Savings_complete/{folder_name}/{os.path.basename(path)}/train_accs_convolutional_3_{date}.npz",train_accs_convolutional_1=train_accs_convolutional_3)
    np.savez_compressed(f"/beegfs/desy/user/schreibj/Hadron_classifier/Savings_complete/{folder_name}/{os.path.basename(path)}/train_losses_convolutional_3_{date}.npz",train_losses_convolutional_1=train_losses_convolutional_3)
    np.savez_compressed(f"/beegfs/desy/user/schreibj/Hadron_classifier/Savings_complete/{folder_name}/{os.path.basename(path)}/train_accs_convolutional_4_{date}.npz",train_accs_convolutional_1=train_accs_convolutional_4)
    np.savez_compressed(f"/beegfs/desy/user/schreibj/Hadron_classifier/Savings_complete/{folder_name}/{os.path.basename(path)}/train_losses_convolutional_4_{date}.npz",train_losses_convolutional_1=train_losses_convolutional_4)
    np.savez_compressed(f"/beegfs/desy/user/schreibj/Hadron_classifier/Savings_complete/{folder_name}/{os.path.basename(path)}/train_accs_convolutional_5_{date}.npz",train_accs_convolutional_1=train_accs_convolutional_5)
    np.savez_compressed(f"/beegfs/desy/user/schreibj/Hadron_classifier/Savings_complete/{folder_name}/{os.path.basename(path)}/train_losses_convolutional_5_{date}.npz",train_losses_convolutional_1=train_losses_convolutional_5)
    
    # end of loop over batches
    
    epoch_counter += 1
    print(f'Number of epochs trained: {epoch_counter}')
    

    # EVALUATING THE MODEL after each epoch: 
    model1.eval() 
    model2.eval()
    model3.eval()
    model4.eval()
    model5.eval()
    
    
    
    # 2. Evaluation on evaluationdata:   
    
    for batch_idx, (data, energy, label, ind_outs) in enumerate(validation_loader):
        
        # Check for empty batches. Empty batches will be skipped
        skip_model1 = 0
        skip_model2 = 0
        skip_model3 = 0
        skip_model4 = 0
        skip_model5 = 0
        
        if data[0].shape[0] == 0: skip_model1 = 1
        if data[1].shape[0] == 0: skip_model2 = 1
        if data[2].shape[0] == 0: skip_model3 = 1
        if data[3].shape[0] == 0: skip_model4 = 1
        if data[4].shape[0] == 0: skip_model5 = 1
        
        #Count empty batches
        if skip_model1 == 1: num_empty_batches_validation_model1 += 1 
        if skip_model2 == 1: num_empty_batches_validation_model2 += 1
        if skip_model3 == 1: num_empty_batches_validation_model3 += 1
        if skip_model4 == 1: num_empty_batches_validation_model4 += 1
        if skip_model5 == 1: num_empty_batches_validation_model5 += 1
        
        # Testing validity of data:
        assert data[0].shape[0] == label[0].shape[0]
        
        print(f'Evaluation on Validationdata batch_idx,{batch_idx}')
        
        net_out_validation_1 = model1(data[0].to(device)).detach().cpu()
        net_out_validation_2 = model2(data[1].to(device)).detach().cpu()
        net_out_validation_3 = model3(data[2].to(device)).detach().cpu()
        net_out_validation_4 = model4(data[3].to(device)).detach().cpu()
        net_out_validation_5 = model5(data[4].to(device)).detach().cpu()
        
        label_pred_validation_1 = torch.round(torch.sigmoid(net_out_validation_1))
        label_pred_validation_2 = torch.round(torch.sigmoid(net_out_validation_2))
        label_pred_validation_3 = torch.round(torch.sigmoid(net_out_validation_3))
        label_pred_validation_4 = torch.round(torch.sigmoid(net_out_validation_4))
        label_pred_validation_5 = torch.round(torch.sigmoid(net_out_validation_5))
        
            
        validation_loss_1 = criterion1(net_out_validation_1,label_pred_validation_1)
        validation_loss_2 = criterion2(net_out_validation_2,label_pred_validation_2)
        validation_loss_3 = criterion3(net_out_validation_3,label_pred_validation_3)
        validation_loss_4 = criterion4(net_out_validation_4,label_pred_validation_4)
        validation_loss_5 = criterion5(net_out_validation_5,label_pred_validation_5)
        
        if skip_model1 == 0: validation_acc_convolutional_1 += sum(label[0] == label_pred_validation_1) / label[0].shape[0]
        if skip_model2 == 0: validation_acc_convolutional_2 += sum(label[1] == label_pred_validation_2) / label[1].shape[0]
        if skip_model3 == 0: validation_acc_convolutional_3 += sum(label[2] == label_pred_validation_3) / label[2].shape[0]
        if skip_model4 == 0: validation_acc_convolutional_4 += sum(label[3] == label_pred_validation_4) / label[3].shape[0]
        if skip_model5 == 0: validation_acc_convolutional_5 += sum(label[4] == label_pred_validation_5) / label[4].shape[0]
        
        # Keeping track of the validation loss
        if skip_model1 == 0: validation_loss_convolutional_1 += validation_loss_1.item()
        if skip_model2 == 0: validation_loss_convolutional_2 += validation_loss_2.item()
        if skip_model3 == 0: validation_loss_convolutional_3 += validation_loss_3.item()
        if skip_model4 == 0: validation_loss_convolutional_4 += validation_loss_4.item()
        if skip_model5 == 0: validation_loss_convolutional_5 += validation_loss_5.item()
        
    # Calculating the medium accuracy and medium loss on the validation data after each epoch and logging into weights and biases                               
    validation_acc_convolutional_1 /= (n_batches_validation_total - num_empty_batches_validation_model1) 
    if wandb_logging == True:
        wandb.log({'epoch': epoch_counter, 'validation_acc_convolutional_1': validation_acc_convolutional_1})
    validation_loss_convolutional_1 /= (n_batches_validation_total - num_empty_batches_validation_model1)        
    if wandb_logging == True:
        wandb.log({'epoch': epoch_counter, 'validation_loss_convolutional_1': validation_loss_convolutional_1})
    validation_acc_convolutional_2 /= (n_batches_validation_total - num_empty_batches_validation_model2) 
    if wandb_logging == True:
        wandb.log({'epoch': epoch_counter, 'validation_acc_convolutional_2': validation_acc_convolutional_2})
    validation_loss_convolutional_2 /= (n_batches_validation_total - num_empty_batches_validation_model2)         
    if wandb_logging == True:
        wandb.log({'epoch': epoch_counter, 'validation_loss_convolutional_2': validation_loss_convolutional_2}) 
    validation_acc_convolutional_3 /= (n_batches_validation_total - num_empty_batches_validation_model3) 
    if wandb_logging == True:
        wandb.log({'epoch': epoch_counter, 'validation_acc_convolutional_3': validation_acc_convolutional_3})
    validation_loss_convolutional_3 /= (n_batches_validation_total - num_empty_batches_validation_model3)         
    if wandb_logging == True:
        wandb.log({'epoch': epoch_counter, 'validation_loss_convolutional_3': validation_loss_convolutional_3}) 
    validation_acc_convolutional_4 /= (n_batches_validation_total - num_empty_batches_validation_model4)  
    if wandb_logging == True:
        wandb.log({'epoch': epoch_counter, 'validation_acc_convolutional_4': validation_acc_convolutional_4})
    validation_loss_convolutional_4 /= (n_batches_validation_total - num_empty_batches_validation_model4)         
    if wandb_logging == True:
        wandb.log({'epoch': epoch_counter, 'validation_loss_convolutional_4': validation_loss_convolutional_4})
    validation_acc_convolutional_5 /= (n_batches_validation_total - num_empty_batches_validation_model5)  
    if wandb_logging == True:
        wandb.log({'epoch': epoch_counter, 'validation_acc_convolutional_5': validation_acc_convolutional_5})
    validation_loss_convolutional_5 /= (n_batches_validation_total - num_empty_batches_validation_model5)         
    if wandb_logging == True:
        wandb.log({'epoch': epoch_counter, 'validation_loss_convolutional_5': validation_loss_convolutional_5}) 



    # print some information
    print("Model 1 - convolutional Network Epoch:",epoch_counter, "Validation Accuracy 1:", validation_acc_convolutional_1, "Validation Loss 1:", validation_loss_convolutional_1, "# Empty batches :", num_empty_batches_validation_model1)
    print("Model 2 - convolutional Network Epoch:",epoch_counter, "Validation Accuracy 2:", validation_acc_convolutional_2, "Validation Loss 2:", validation_loss_convolutional_2, "# Empty batches :", num_empty_batches_validation_model2)
    print("Model 3 - convolutional Network Epoch:",epoch_counter, "Validation Accuracy 3:", validation_acc_convolutional_3, "Validation Loss 3:", validation_loss_convolutional_3, "# Empty batches :", num_empty_batches_validation_model3)
    print("Model 4 - convolutional Network Epoch:",epoch_counter, "Validation Accuracy 4:", validation_acc_convolutional_4, "Validation Loss 4:", validation_loss_convolutional_4, "# Empty batches :", num_empty_batches_validation_model4)
    print("Model 5 - convolutional Network Epoch:",epoch_counter, "Validation Accuracy 5:", validation_acc_convolutional_5, "Validation Loss 5:", validation_loss_convolutional_5, "# Empty batches :", num_empty_batches_validation_model5)
    
    # and store the accuracy and loss for later use
    validation_accs_convolutional_1 = np.append(validation_accs_convolutional_1,validation_acc_convolutional_1)
    validation_losses_convolutional_1 = np.append(validation_losses_convolutional_1,validation_loss_convolutional_1)
    validation_accs_convolutional_2 = np.append(validation_accs_convolutional_2,validation_acc_convolutional_2)
    validation_losses_convolutional_2 = np.append(validation_losses_convolutional_2,validation_loss_convolutional_2)
    validation_accs_convolutional_3 = np.append(validation_accs_convolutional_3,validation_acc_convolutional_3)
    validation_losses_convolutional_3 = np.append(validation_losses_convolutional_3,validation_loss_convolutional_3)
    validation_accs_convolutional_4 = np.append(validation_accs_convolutional_4,validation_acc_convolutional_4)
    validation_losses_convolutional_4 = np.append(validation_losses_convolutional_4,validation_loss_convolutional_4)
    validation_accs_convolutional_5 = np.append(validation_accs_convolutional_5,validation_acc_convolutional_5)
    validation_losses_convolutional_5 = np.append(validation_losses_convolutional_5,validation_loss_convolutional_5)

    np.savez_compressed(f"/beegfs/desy/user/schreibj/Hadron_classifier/Savings_complete/{folder_name}/{os.path.basename(path)}/validation_accs_convolutional_1_{date}.npz",validation_accs_convolutional_1=validation_accs_convolutional_1)
    np.savez_compressed(f"/beegfs/desy/user/schreibj/Hadron_classifier/Savings_complete/{folder_name}/{os.path.basename(path)}/validation_losses_convolutional_1_{date}.npz",validation_losses_convolutional_1=validation_losses_convolutional_1)
    np.savez_compressed(f"/beegfs/desy/user/schreibj/Hadron_classifier/Savings_complete/{folder_name}/{os.path.basename(path)}/validation_accs_convolutional_2_{date}.npz",validation_accs_convolutional_2=validation_accs_convolutional_2)
    np.savez_compressed(f"/beegfs/desy/user/schreibj/Hadron_classifier/Savings_complete/{folder_name}/{os.path.basename(path)}/validation_losses_convolutional_2_{date}.npz",validation_losses_convolutional_2=validation_losses_convolutional_2)
    np.savez_compressed(f"/beegfs/desy/user/schreibj/Hadron_classifier/Savings_complete/{folder_name}/{os.path.basename(path)}/validation_accs_convolutional_3_{date}.npz",validation_accs_convolutional_3=validation_accs_convolutional_3)
    np.savez_compressed(f"/beegfs/desy/user/schreibj/Hadron_classifier/Savings_complete/{folder_name}/{os.path.basename(path)}/validation_losses_convolutional_3_{date}.npz",validation_losses_convolutional_3=validation_losses_convolutional_3)
    np.savez_compressed(f"/beegfs/desy/user/schreibj/Hadron_classifier/Savings_complete/{folder_name}/{os.path.basename(path)}/validation_accs_convolutional_4_{date}.npz",validation_accs_convolutional_4=validation_accs_convolutional_4)
    np.savez_compressed(f"/beegfs/desy/user/schreibj/Hadron_classifier/Savings_complete/{folder_name}/{os.path.basename(path)}/validation_losses_convolutional_4_{date}.npz",validation_losses_convolutional_4=validation_losses_convolutional_4)
    np.savez_compressed(f"/beegfs/desy/user/schreibj/Hadron_classifier/Savings_complete/{folder_name}/{os.path.basename(path)}/validation_accs_convolutional_5_{date}.npz",validation_accs_convolutional_5=validation_accs_convolutional_5)
    np.savez_compressed(f"/beegfs/desy/user/schreibj/Hadron_classifier/Savings_complete/{folder_name}/{os.path.basename(path)}/validation_losses_convolutional_5_{date}.npz",validation_losses_convolutional_5=validation_losses_convolutional_5)
    
    # saving the model if the validation loss is lower than before. In the end the model with the lowest validation lost is being stored               
    
    #  best model                
    if epoch_counter == 1:
        lowest_validation_loss_convolutional_1 = validation_loss_convolutional_1
        lowest_validation_loss_convolutional_2 = validation_loss_convolutional_2
        lowest_validation_loss_convolutional_3 = validation_loss_convolutional_3
        lowest_validation_loss_convolutional_4 = validation_loss_convolutional_4
        lowest_validation_loss_convolutional_5 = validation_loss_convolutional_5
        torch.save(model1.state_dict(), f"/beegfs/desy/user/schreibj/Hadron_classifier/Savings_complete/{folder_name}/{os.path.basename(path)}/best_Hadron_classifier_convolutional_1,0Sigma_model_1.pt")   
        torch.save(model2.state_dict(), f"/beegfs/desy/user/schreibj/Hadron_classifier/Savings_complete/{folder_name}/{os.path.basename(path)}/best_Hadron_classifier_convolutional_1,0Sigma_model_2.pt")    
        torch.save(model3.state_dict(), f"/beegfs/desy/user/schreibj/Hadron_classifier/Savings_complete/{folder_name}/{os.path.basename(path)}/best_Hadron_classifier_convolutional_1,0Sigma_model_3.pt")    
        torch.save(model4.state_dict(), f"/beegfs/desy/user/schreibj/Hadron_classifier/Savings_complete/{folder_name}/{os.path.basename(path)}/best_Hadron_classifier_convolutional_1,0Sigma_model_4.pt") 
        torch.save(model5.state_dict(), f"/beegfs/desy/user/schreibj/Hadron_classifier/Savings_complete/{folder_name}/{os.path.basename(path)}/best_Hadron_classifier_convolutional_1,0Sigma_model_5.pt")    
            
            
    if validation_loss_convolutional_1 < lowest_validation_loss_convolutional_1:
        lowest_validation_loss_convolutional_1 = validation_loss_convolutional_1                           
        torch.save(model1.state_dict(), f"/beegfs/desy/user/schreibj/Hadron_classifier/Savings_complete/{folder_name}/{os.path.basename(path)}/best_Hadron_classifier_convolutional_1,0Sigma_model_1.pt")                               
    if validation_loss_convolutional_2 < lowest_validation_loss_convolutional_2:
        lowest_validation_loss_convolutional_2 = validation_loss_convolutional_2                           
        torch.save(model2.state_dict(), f"/beegfs/desy/user/schreibj/Hadron_classifier/Savings_complete/{folder_name}/{os.path.basename(path)}/best_Hadron_classifier_convolutional_1,0Sigma_model_2.pt")
    if validation_loss_convolutional_3 < lowest_validation_loss_convolutional_3:
        lowest_validation_loss_convolutional_3 = validation_loss_convolutional_3                           
        torch.save(model3.state_dict(), f"/beegfs/desy/user/schreibj/Hadron_classifier/Savings_complete/{folder_name}/{os.path.basename(path)}/best_Hadron_classifier_convolutional_1,0Sigma_model_3.pt")
    if validation_loss_convolutional_4 < lowest_validation_loss_convolutional_4:
        lowest_validation_loss_convolutional_4 = validation_loss_convolutional_4                           
        torch.save(model4.state_dict(), f"/beegfs/desy/user/schreibj/Hadron_classifier/Savings_complete/{folder_name}/{os.path.basename(path)}/best_Hadron_classifier_convolutional_1,0Sigma_model_4.pt")
    if validation_loss_convolutional_5 < lowest_validation_loss_convolutional_5:
        lowest_validation_loss_convolutional_5 = validation_loss_convolutional_5                           
        torch.save(model5.state_dict(), f"/beegfs/desy/user/schreibj/Hadron_classifier/Savings_complete/{folder_name}/{os.path.basename(path)}/best_Hadron_classifier_convolutional_1,0Sigma_model_5.pt")
    
    print(f'lowest_validation_loss_convolutional_1: {lowest_validation_loss_convolutional_1}')
    print(f'lowest_validation_loss_convolutional_2: {lowest_validation_loss_convolutional_2}')
    print(f'lowest_validation_loss_convolutional_3: {lowest_validation_loss_convolutional_3}')
    print(f'lowest_validation_loss_convolutional_4: {lowest_validation_loss_convolutional_4}')
    print(f'lowest_validation_loss_convolutional_5: {lowest_validation_loss_convolutional_5}')
    
    # last model
    torch.save(model1.state_dict(), f"/beegfs/desy/user/schreibj/Hadron_classifier/Savings_complete/{folder_name}/{os.path.basename(path)}/last_Hadron_classifier_convolutional_1,0Sigma_model_1.pt")
    torch.save(model2.state_dict(), f"/beegfs/desy/user/schreibj/Hadron_classifier/Savings_complete/{folder_name}/{os.path.basename(path)}/last_Hadron_classifier_convolutional_1,0Sigma_model_2.pt")
    torch.save(model3.state_dict(), f"/beegfs/desy/user/schreibj/Hadron_classifier/Savings_complete/{folder_name}/{os.path.basename(path)}/last_Hadron_classifier_convolutional_1,0Sigma_model_3.pt")
    torch.save(model4.state_dict(), f"/beegfs/desy/user/schreibj/Hadron_classifier/Savings_complete/{folder_name}/{os.path.basename(path)}/last_Hadron_classifier_convolutional_1,0Sigma_model_4.pt")
    torch.save(model5.state_dict(), f"/beegfs/desy/user/schreibj/Hadron_classifier/Savings_complete/{folder_name}/{os.path.basename(path)}/last_Hadron_classifier_convolutional_1,0Sigma_model_5.pt")
    
    endtime_epoch = datetime.now()

    print(f'time for one epoch: {endtime_epoch - start_epoch_time}')
    
    # end of loop over epochs     
          
# Mark the run as finished
if wandb_logging == True:
    wandb.finish()

end_convolutional = datetime.now()
print(f'total time training Convolutional: {end_convolutional - start_convolutional}')


# Plotting the Loss - after each batch (for the first epochs) 

print(f'len(train_losses_convolutional_batchwise_1): {len(train_losses_convolutional_batchwise_1)}')
print(f'len(train_losses_convolutional_batchwise_2): {len(train_losses_convolutional_batchwise_2)}')
print(f'len(train_losses_convolutional_batchwise_3): {len(train_losses_convolutional_batchwise_3)}')
print(f'len(train_losses_convolutional_batchwise_4): {len(train_losses_convolutional_batchwise_4)}')#
print(f'len(train_losses_convolutional_batchwise_5): {len(train_losses_convolutional_batchwise_5)}')
 
fig, ax = plt.subplots(figsize=(15,8))
plt.axis('on')
ax.plot(train_losses_convolutional_batchwise_1,label="train_loss_convolutional_batchwise_1",color='blue')
ax.plot(train_losses_convolutional_batchwise_2,label="train_loss_convolutional_batchwise_2",color='red')
ax.plot(train_losses_convolutional_batchwise_3,label="train_loss_convolutional_batchwise_3",color='green')
ax.plot(train_losses_convolutional_batchwise_4,label="train_loss_convolutional_batchwise_4",color='darkorange')
ax.plot(train_losses_convolutional_batchwise_5,label="train_loss_convolutional_batchwise_5",color='purple')
plt.legend(fontsize = 20)
ax.set_xlabel('Batch', fontsize = 25)
ax.set_ylabel('Loss', fontsize = 25)
fig.tight_layout()
plt.savefig(f"/beegfs/desy/user/schreibj/Hadron_classifier/Savings_complete/{folder_name}/{os.path.basename(path)}/Loss_plot_Hadron_showers_convolutional_batchwise_1,0Sigma_model_1_{str(datetime.now())}.pdf")


#Saving the script

# Destination path
destination = f"/beegfs/desy/user/schreibj/Hadron_classifier/Savings_complete/{folder_name}/{os.path.basename(path)}/Script_saves_{script_save_path}_{str(datetime.now())}.txt"
 
try:
    shutil.copy(source, destination)
    print("File copied successfully.")
 
    # If source and destination are same
except shutil.SameFileError:
    print("Source and destination represents the same file.")
 
    # If there is any permission issue
except PermissionError:
    print("Permission denied.")
 
    # For other errors
except:
    print("Error occurred while copying file.")
    
print(f'train_accs_convolutional_1: {train_accs_convolutional_1}')
print(f'validation_accs_convolutional_1: {validation_accs_convolutional_1}')
print(f'train_accs_convolutional_2: {train_accs_convolutional_2}')
print(f'validation_accs_convolutional_2: {validation_accs_convolutional_2}')
print(f'train_accs_convolutional_3: {train_accs_convolutional_3}')
print(f'validation_accs_convolutional_3: {validation_accs_convolutional_3}')
print(f'train_accs_convolutional_4: {train_accs_convolutional_4}')
print(f'validation_accs_convolutional_4: {validation_accs_convolutional_4}')
print(f'train_accs_convolutional_5: {train_accs_convolutional_5}')
print(f'validation_accs_convolutional_5: {validation_accs_convolutional_5}')


# Plotting the accuracy of all models together
print(f'len(train_accs_convolutional_1): {len(train_accs_convolutional_1)}')
print(f'len(validation_accs_convolutional_1): {len(validation_accs_convolutional_1)}')
print(f'len(train_accs_convolutional_1): {len(train_accs_convolutional_2)}')
print(f'len(validation_accs_convolutional_1): {len(validation_accs_convolutional_2)}')
print(f'len(train_accs_convolutional_1): {len(train_accs_convolutional_3)}')
print(f'len(validation_accs_convolutional_1): {len(validation_accs_convolutional_3)}')
print(f'len(train_accs_convolutional_1): {len(train_accs_convolutional_4)}')
print(f'len(validation_accs_convolutional_1): {len(validation_accs_convolutional_4)}')
print(f'len(train_accs_convolutional_1): {len(train_accs_convolutional_5)}')
print(f'len(validation_accs_convolutional_1): {len(validation_accs_convolutional_5)}')
fig, ax = plt.subplots(figsize=(15,8))
plt.axis('on')
ax.plot(train_accs_convolutional_1,label="train_convolutional_1",color='blue')
ax.plot(validation_accs_convolutional_1,label="validation_convolutional_2",color='blue',linestyle='dashed')
ax.plot(train_accs_convolutional_2,label="train_convolutional_1",color='red')
ax.plot(validation_accs_convolutional_2,label="validation_convolutional_2",color='red',linestyle='dashed')
ax.plot(train_accs_convolutional_3,label="train_convolutional_3",color='green')
ax.plot(validation_accs_convolutional_3,label="validation_convolutional_3",color='green',linestyle='dashed')
ax.plot(train_accs_convolutional_4,label="train_convolutional_4",color='darkorange')
ax.plot(validation_accs_convolutional_4,label="validation_convolutional_4",color='darkorange',linestyle='dashed')
ax.plot(train_accs_convolutional_5,label="train_convolutional_5",color='purple')
ax.plot(validation_accs_convolutional_5,label="validation_convolutional_5",color='purple',linestyle='dashed')
plt.legend(fontsize = 20)
ax.set_xlabel("Epoch", fontsize = 25)
ax.set_ylabel("Accuracy", fontsize = 25)
fig.tight_layout()
plt.savefig(f"/beegfs/desy/user/schreibj/Hadron_classifier/Savings_complete/{folder_name}/{os.path.basename(path)}/Accuracy_plot_Hadron_showers_convolutional_1,0Sigma_error_run_{str(datetime.now())}.pdf")

total_end = datetime.now()

print(f'total time for script: {total_end - total_start}')
print('Script finisched')


# Plotting the loss - with axhline
fig, ax = plt.subplots(figsize=(15,8))
plt.axis('on')
ax.plot(train_losses_convolutional_1,label="train_convolutional_1",color='blue')
ax.plot(validation_losses_convolutional_1,label="validation_convolutional_2",color='blue',linestyle='dashed')
ax.plot(train_losses_convolutional_2,label="train_convolutional_1",color='red')
ax.plot(validation_losses_convolutional_2,label="validation_convolutional_2",color='red',linestyle='dashed')
ax.plot(train_losses_convolutional_3,label="train_convolutional_3",color='green')
ax.plot(validation_losses_convolutional_3,label="validation_convolutional_3",color='green',linestyle='dashed')
ax.plot(train_losses_convolutional_4,label="train_convolutional_4",color='darkorange')
ax.plot(validation_losses_convolutional_4,label="validation_convolutional_4",color='darkorange',linestyle='dashed')
ax.plot(train_losses_convolutional_5,label="train_convolutional_5",color='purple')
ax.plot(validation_losses_convolutional_5,label="validation_convolutional_5",color='purple',linestyle='dashed')
plt.axhline(lowest_validation_loss_convolutional_1,color='blue',ls='dashed', label='lowest_validation_loss_convolutional_1')
plt.axhline(lowest_validation_loss_convolutional_2,color='red',ls='dashed', label='lowest_validation_loss_convolutional_2')
plt.axhline(lowest_validation_loss_convolutional_3,color='green',ls='dashed', label='lowest_validation_loss_convolutional_3')
plt.axhline(lowest_validation_loss_convolutional_4,color='darkorange',ls='dashed', label='lowest_validation_loss_convolutional_4')
plt.axhline(lowest_validation_loss_convolutional_5,color='purple',ls='dashed', label='lowest_validation_loss_convolutional_5')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
ax.set_xlabel("Epoch", fontsize = 25)
ax.set_ylabel("Loss", fontsize = 25)
fig.tight_layout()
plt.savefig(f"/beegfs/desy/user/schreibj/Hadron_classifier/Savings_complete/{folder_name}/{os.path.basename(path)}/Loss_plot_Hadron_showers_convolutional_axhline_error_run_{str(datetime.now())}.pdf")

# Plotting the loss 
print(f'train_losses_convolutional_1: {train_losses_convolutional_1}')
print(f'validation_losses_convolutional_1: {validation_losses_convolutional_1}')
print(f'train_losses_convolutional_2: {train_losses_convolutional_2}')
print(f'validation_losses_convolutional_2: {validation_losses_convolutional_2}')
print(f'train_losses_convolutional_3: {train_losses_convolutional_3}')
print(f'validation_losses_convolutional_3: {validation_losses_convolutional_3}')
print(f'train_losses_convolutional_4: {train_losses_convolutional_4}')
print(f'validation_losses_convolutional_4: {validation_losses_convolutional_4}')
print(f'train_losses_convolutional_5: {train_losses_convolutional_5}')
print(f'validation_losses_convolutional_5: {validation_losses_convolutional_5}')

print(f'len(train_losses_convolutional_1): {len(train_losses_convolutional_1)}')
print(f'len(validation_losses_convolutional_1): {len(validation_losses_convolutional_1)}')
print(f'len(train_losses_convolutional_1): {len(train_losses_convolutional_2)}')
print(f'len(validation_losses_convolutional_1): {len(validation_losses_convolutional_2)}')
print(f'len(train_losses_convolutional_1): {len(train_losses_convolutional_3)}')
print(f'len(validation_losses_convolutional_1): {len(validation_losses_convolutional_3)}')
print(f'len(train_losses_convolutional_1): {len(train_losses_convolutional_4)}')
print(f'len(validation_losses_convolutional_1): {len(validation_losses_convolutional_4)}')
print(f'len(train_losses_convolutional_1): {len(train_losses_convolutional_5)}')
print(f'len(validation_losses_convolutional_1): {len(validation_losses_convolutional_5)}')

fig, ax = plt.subplots(figsize=(15,8))
plt.axis('on')
ax.plot(train_losses_convolutional_1,label="train_convolutional_1",color='blue')
ax.plot(validation_losses_convolutional_1,label="validation_convolutional_2",color='blue',linestyle='dashed')
ax.plot(train_losses_convolutional_2,label="train_convolutional_1",color='red')
ax.plot(validation_losses_convolutional_2,label="validation_convolutional_2",color='red',linestyle='dashed')
ax.plot(train_losses_convolutional_3,label="train_convolutional_3",color='green')
ax.plot(validation_losses_convolutional_3,label="validation_convolutional_3",color='green',linestyle='dashed')
ax.plot(train_losses_convolutional_4,label="train_convolutional_4",color='darkorange')
ax.plot(validation_losses_convolutional_4,label="validation_convolutional_4",color='darkorange',linestyle='dashed')
ax.plot(train_losses_convolutional_5,label="train_convolutional_5",color='purple')
ax.plot(validation_losses_convolutional_5,label="validation_convolutional_5",color='purple',linestyle='dashed')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
ax.set_xlabel("Epoch", fontsize = 25)
ax.set_ylabel("Loss", fontsize = 25)
fig.tight_layout()
plt.savefig(f"/beegfs/desy/user/schreibj/Hadron_classifier/Savings_complete/{folder_name}/{os.path.basename(path)}/Loss_plot_Hadron_showers_convolutional_error_run_{str(datetime.now())}.pdf")







#Plotting the ROC-Curves and saving AUC_SCORE

# Create a directory Plots afterwards
path_plots_afterwards = f"/beegfs/desy/user/schreibj/Hadron_classifier/Savings_complete/{folder_name}/{project_name_wandb}_{date}_savings_complete/Plots afterwards"

try:
    os.makedirs(path_plots_afterwards, exist_ok = True)
except OSError:
    pass



roc_auc_scores_model1 = np.array([])
roc_auc_scores_model2 = np.array([])
roc_auc_scores_model3 = np.array([])
roc_auc_scores_model4 = np.array([])
roc_auc_scores_model5 = np.array([])


# Loading a pretrained convolutional Network
model1 = NeuralNet_convolutional().to(device)
model2 = NeuralNet_convolutional().to(device)
model3 = NeuralNet_convolutional().to(device)
model4 = NeuralNet_convolutional().to(device)
model5 = NeuralNet_convolutional().to(device)
model1.load_state_dict(torch.load(f"/beegfs/desy/user/schreibj/Hadron_classifier/Savings_complete/{folder_name}/{os.path.basename(path)}/best_Hadron_classifier_convolutional_1,0Sigma_model_1.pt", map_location=torch.device('cpu')))
model2.load_state_dict(torch.load(f"/beegfs/desy/user/schreibj/Hadron_classifier/Savings_complete/{folder_name}/{os.path.basename(path)}/best_Hadron_classifier_convolutional_1,0Sigma_model_2.pt", map_location=torch.device('cpu')))
model3.load_state_dict(torch.load(f"/beegfs/desy/user/schreibj/Hadron_classifier/Savings_complete/{folder_name}/{os.path.basename(path)}/best_Hadron_classifier_convolutional_1,0Sigma_model_3.pt", map_location=torch.device('cpu')))
model4.load_state_dict(torch.load(f"/beegfs/desy/user/schreibj/Hadron_classifier/Savings_complete/{folder_name}/{os.path.basename(path)}/best_Hadron_classifier_convolutional_1,0Sigma_model_4.pt", map_location=torch.device('cpu')))
model5.load_state_dict(torch.load(f"/beegfs/desy/user/schreibj/Hadron_classifier/Savings_complete/{folder_name}/{os.path.basename(path)}/best_Hadron_classifier_convolutional_1,0Sigma_model_5.pt", map_location=torch.device('cpu')))
model1.eval()
model2.eval()
model3.eval()
model4.eval()
model5.eval()
model1.to(device)
model2.to(device)
model3.to(device)
model4.to(device)
model5.to(device)


print(f'number of total test batches available: {int(params_convolutional["test_size"]/params_convolutional["batch_size"])}')


# Predict on the test-set
list_labels_pred_model1 = np.array([])
list_labels_true_model1 = np.array([])
list_labels_pred_model2 = np.array([])
list_labels_true_model2 = np.array([])
list_labels_pred_model3 = np.array([])
list_labels_true_model3 = np.array([])
list_labels_pred_model4 = np.array([])
list_labels_true_model4 = np.array([])
list_labels_pred_model5 = np.array([])
list_labels_true_model5 = np.array([])

for batch_idx, (data, energy, label, ind_outs) in enumerate(test_loader):
        print(f'Evaluation on Testdata batch_idx,{batch_idx}')
        
        # Check for empty batches. Empty batches will be skipped
        skip_model1 = 0
        skip_model2 = 0
        skip_model3 = 0
        skip_model4 = 0
        skip_model5 = 0
        
        if data[0].shape[0] == 0: skip_model1 = 1
        if data[1].shape[0] == 0: skip_model2 = 1
        if data[2].shape[0] == 0: skip_model3 = 1
        if data[3].shape[0] == 0: skip_model4 = 1
        if data[4].shape[0] == 0: skip_model5 = 1
          
        label1 = label[0].clone().detach().to(device)
        label2 = label[1].clone().detach().to(device)
        label3 = label[2].clone().detach().to(device)
        label4 = label[3].clone().detach().to(device)
        label5 = label[4].clone().detach().to(device)
        
        #net_out_test = model(data.to(device)).detach().cpu()
        net_out_test1 = model1(data[0].to(device)).detach().cpu()
        net_out_test2 = model2(data[1].to(device)).detach().cpu()
        net_out_test3 = model3(data[2].to(device)).detach().cpu()
        net_out_test4 = model4(data[3].to(device)).detach().cpu()
        net_out_test5 = model5(data[4].to(device)).detach().cpu()
        
        
        label_pred_test1 = torch.sigmoid(net_out_test1)
        label_pred_test2 = torch.sigmoid(net_out_test2)
        label_pred_test3 = torch.sigmoid(net_out_test3)
        label_pred_test4 = torch.sigmoid(net_out_test4)
        label_pred_test5 = torch.sigmoid(net_out_test5)
        #print(f'label_pred_test.shape,{label_pred_test.shape}')
        
        if skip_model1 == 0:list_labels_pred_model1 = np.append(list_labels_pred_model1,label_pred_test1)
        if skip_model1 == 0:list_labels_true_model1 = np.append(list_labels_true_model1,label1.cpu())
        if skip_model2 == 0:list_labels_pred_model2 = np.append(list_labels_pred_model2,label_pred_test2)
        if skip_model2 == 0:list_labels_true_model2 = np.append(list_labels_true_model2,label2.cpu())
        if skip_model3 == 0:list_labels_pred_model3 = np.append(list_labels_pred_model3,label_pred_test3)
        if skip_model3 == 0:list_labels_true_model3 = np.append(list_labels_true_model3,label3.cpu())
        if skip_model4 == 0:list_labels_pred_model4 = np.append(list_labels_pred_model4,label_pred_test4)
        if skip_model4 == 0:list_labels_true_model4 = np.append(list_labels_true_model4,label4.cpu())
        if skip_model5 == 0:list_labels_pred_model5 = np.append(list_labels_pred_model5,label_pred_test5)
        if skip_model5 == 0:list_labels_true_model5 = np.append(list_labels_true_model5,label5.cpu())

print(f'End - list_labels_pred_model1.shape,{list_labels_pred_model1.shape}')
print(f'End - list_labels_true_model1.shape,{list_labels_true_model1.shape}')

#AUC
fpr_model1, tpr_model1, thresholds_model1 = sklearn.metrics.roc_curve(list_labels_true_model1, list_labels_pred_model1, pos_label=1,drop_intermediate=False) 
fpr_model2, tpr_model2, thresholds_model2 = sklearn.metrics.roc_curve(list_labels_true_model2, list_labels_pred_model2, pos_label=1,drop_intermediate=False) 
fpr_model3, tpr_model3, thresholds_model3 = sklearn.metrics.roc_curve(list_labels_true_model3, list_labels_pred_model3, pos_label=1,drop_intermediate=False) 
fpr_model4, tpr_model4, thresholds_model4 = sklearn.metrics.roc_curve(list_labels_true_model4, list_labels_pred_model4, pos_label=1,drop_intermediate=False) 
fpr_model5, tpr_model5, thresholds_model5 = sklearn.metrics.roc_curve(list_labels_true_model5, list_labels_pred_model5, pos_label=1,drop_intermediate=False) 
roc_auc_model1 = auc(fpr_model1,tpr_model1)
roc_auc_model2 = auc(fpr_model2,tpr_model2)
roc_auc_model3 = auc(fpr_model3,tpr_model3)
roc_auc_model4 = auc(fpr_model4,tpr_model4)
roc_auc_model5 = auc(fpr_model5,tpr_model5)

print(f'roc_auc_model1: ,{roc_auc_model1}')
print(f'roc_auc_model2: ,{roc_auc_model2}')
print(f'roc_auc_model3: ,{roc_auc_model3}')
print(f'roc_auc_model4: ,{roc_auc_model4}')
print(f'roc_auc_model5: ,{roc_auc_model5}')

roc_auc_scores_model1 = np.append(roc_auc_scores_model1,roc_auc_model1)
roc_auc_scores_model2 = np.append(roc_auc_scores_model2,roc_auc_model2)
roc_auc_scores_model3 = np.append(roc_auc_scores_model3,roc_auc_model3)
roc_auc_scores_model4 = np.append(roc_auc_scores_model4,roc_auc_model4)
roc_auc_scores_model5 = np.append(roc_auc_scores_model5,roc_auc_model5)


#Plot roc-auc scores 

plt.figure()
lw = 2
plt.plot(
    fpr_model1,
    tpr_model1,
    color="darkorange",
    lw=lw,
    label="ROC curve (area = %0.2f)" % roc_auc_scores_model1,
)
plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc="lower right")
plt.title(" 1,0Sigma _model1")
#plt.show()
plt.savefig(f"{path_plots_afterwards}/roc_auc_convolutional_1,0Sigma_{str(datetime.now())}.pdf")

plt.figure()
lw = 2
plt.plot(
    fpr_model2,
    tpr_model2,
    color="darkorange",
    lw=lw,
    label="ROC curve (area = %0.2f)" % roc_auc_scores_model2,
)
plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc="lower right")
plt.title(" 1,0Sigma _model2")
#plt.show()
plt.savefig(f"{path_plots_afterwards}/roc_auc_convolutional_1,0Sigma_{str(datetime.now())}.pdf")

plt.figure()
lw = 2
plt.plot(
    fpr_model3,
    tpr_model3,
    color="darkorange",
    lw=lw,
    label="ROC curve (area = %0.2f)" % roc_auc_scores_model3,
)
plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc="lower right")
plt.title(" 1,0Sigma _model3")
#plt.show()
plt.savefig(f"{path_plots_afterwards}/roc_auc_convolutional_1,0Sigma_{str(datetime.now())}.pdf")

plt.figure()
lw = 2
plt.plot(
    fpr_model4,
    tpr_model4,
    color="darkorange",
    lw=lw,
    label="ROC curve (area = %0.2f)" % roc_auc_scores_model4,
)
plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc="lower right")
plt.title(" 1,0Sigma _model4")
#plt.show()
plt.savefig(f"{path_plots_afterwards}/roc_auc_convolutional_1,0Sigma_{str(datetime.now())}.pdf")

plt.figure()
lw = 2
plt.plot(
    fpr_model5,
    tpr_model5,
    color="darkorange",
    lw=lw,
    label="ROC curve (area = %0.2f)" % roc_auc_scores_model5,
)
plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc="lower right")
plt.title(" 1,0Sigma _model5")
#plt.show()
plt.savefig(f"{path_plots_afterwards}/roc_auc_convolutional_1,0Sigma_{str(datetime.now())}.pdf")


#Plot roc-auc scores all together in one plot

plt.figure()
lw = 2

plt.plot(
    fpr_model1,
    tpr_model1,
    color="blue",
    lw=lw,
    label="ROC curve model1 (area = %0.2f)" % roc_auc_scores_model1,
)

plt.plot(
    fpr_model2,
    tpr_model2,
    color="red",
    lw=lw,
    label="ROC curve model2 (area = %0.2f)" % roc_auc_scores_model2,
)

plt.plot(
    fpr_model3,
    tpr_model3,
    color="green",
    lw=lw,
    label="ROC curve model3 (area = %0.2f)" % roc_auc_scores_model3,
)

plt.plot(
    fpr_model4,
    tpr_model4,
    color="darkorange",
    lw=lw,
    label="ROC curve model4 (area = %0.2f)" % roc_auc_scores_model4,
)

plt.plot(
    fpr_model5,
    tpr_model5,
    color="purple",
    lw=lw,
    label="ROC curve model5 (area = %0.2f)" % roc_auc_scores_model5,
)
plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc="lower right")
plt.title(" 1,0Sigma")
#plt.show()
plt.savefig(f"{path_plots_afterwards}/roc_auc_convolutional_1,0Sigma_{str(datetime.now())}.pdf")


#Save roc-auc scores as .npz

np.savez_compressed(f"/beegfs/desy/user/schreibj/Hadron_classifier/Savings_complete/{folder_name}/{os.path.basename(path)}/roc_auc_scores_convolutional_1,0Sigma_error_run_{date}.npz"
                    ,roc_auc_scores_model1=roc_auc_scores_model1
                    ,roc_auc_scores_model2=roc_auc_scores_model2
                    ,roc_auc_scores_model3=roc_auc_scores_model3
                    ,roc_auc_scores_model4=roc_auc_scores_model4
                    ,roc_auc_scores_model5=roc_auc_scores_model5
                    )

total_end = datetime.now()

print(f'total time for script: {total_end - total_start}')
print('Script finished')