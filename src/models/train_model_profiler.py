# Loading packages
import os
import sys
import warnings

# Ignore future warnings that arises from BertModel
warnings.filterwarnings("ignore", category=FutureWarning)

# Importing base modules
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import DataLoader
from transformers import BertModel

# Inserting path to AmazonData class object
sys.path.insert(0, os.getcwd() + "/src/data/")
from AmazonData import AmazonData

# Setting device to cpu or cuda
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Training on {device}')

# Defining parameters
batch_size = 75
opt = "sgd"
drop_out = 0.35
lr = 0.01

# Define loss function
loss_fn = nn.CrossEntropyLoss().to(device)

# Load class model
class SentimentClassifier(nn.Module):
    def __init__(self, n_classes, p):
        super(SentimentClassifier, self).__init__()
        # load BERT-model
        self.bert = BertModel.from_pretrained("bert-base-cased")
        self.drop = nn.Dropout(p=p)
        self.out = nn.Linear(self.bert.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        _, pooled_output = self.bert(
            input_ids=input_ids, attention_mask=attention_mask, return_dict=False
        )
        output = self.drop(pooled_output)
        return self.out(output)

# Build dataloader
def build_dataLoader(data, batch_size):
    return DataLoader(data, batch_size=batch_size, shuffle=True)

# Build optimizer
def build_optimizer(opt, model, lr):
    if opt == 'adam':
        return optim.Adam(model.parameters(), lr=lr, betas=(0.85,0.89), weight_decay=1e-3)
    else:
        return optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-3)

# Define train-loop
def train_epoch(model, trainloader, optimizer):
    running_loss = 0
    running_acc = 0
    num_batches = len(trainloader)
    i = 0
    for batch_idx, data in enumerate(trainloader):
        if i > 5:
            break
        print(f"Batch {batch_idx+1} of {num_batches}")
        input_ids = data["input_ids"].to(device)
        attention_mask = data["attention_mask"].to(device)
        labels = data["targets"].to(device)
        # Zero the gradients
        optimizer.zero_grad()
        # make predictions
        output = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = loss_fn(output, labels)
        running_loss += loss.item()
        # backpropagate loss
        loss.backward()
        # take step
        optimizer.step()
        # Softmax predictions and get indices
        y_pred = F.softmax(output, dim=1).argmax(dim=1)
        running_acc += ((y_pred == labels).sum()/ labels.shape[0]).item() / num_batches
        i += 1
        if ((batch_idx + 1) % 2) == 0:
            print(
                f"Loss: {running_loss} \tAccuracy: {round(running_acc,4) * 100}%"
            )
    return running_loss / num_batches, running_acc

# build model
def build_model(dropout):
    return SentimentClassifier(n_classes=3, p=dropout)

# Train
def train():
    """
    Function to train model and store training results
    Requirements:
        - Data must have been generated before executing this script

    Outputs:
        - models/final_model.pth

    """
    # Build model
    model = build_model(drop_out).to(device)
    # load train_set
    train_set = torch.load('data/processed/train.pth')
    # create dataloader
    trainloader = build_dataLoader(train_set, batch_size)
    # create optimizer
    optimizer = build_optimizer(opt, model, lr)
    # define epochs
    num_epochs = 3
    model.train()
    for i in range(num_epochs):
        print(f"Epoch {i+1} of {num_epochs}")
        
        avg_loss, avg_acc = train_epoch(model, trainloader, optimizer)
        print(
            f"Epoch {i+1} loss: {avg_loss} \tEpoch acc: {avg_acc}"
        )
    

# Python
if __name__ == "__main__":
    # Train
    train()
