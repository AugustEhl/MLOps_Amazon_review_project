import os
import pprint
import sys
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import torch
import torch.nn.functional as F
import wandb
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import BertModel

sys.path.insert(0, os.getcwd() + "/src/data/")
from AmazonData import AmazonData

# Setting seed
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

sweep_config = {"method": "random"}

parameters_dict = {
    "optimizer": {"values": ["sgd", "adam"]},
    "batch_size": {"values": [100,150,200,250]},
    "drop_out": {"values": [0.15,0.25,0.35]},
    "lr": {"values": [0.01,0.001]},
}
sweep_config["parameters"] = parameters_dict
pprint.pprint(sweep_config)

sweep_id = wandb.sweep(sweep_config, project="Amazon-Reviews-v2", entity="amazonproject")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
loss_fn = nn.CrossEntropyLoss().to(device)


class SentimentClassifier(nn.Module):
    def __init__(self, n_classes, p):
        super(SentimentClassifier, self).__init__()
        self.bert = BertModel.from_pretrained("bert-base-cased")
        self.drop = nn.Dropout(p=p)
        self.out = nn.Linear(self.bert.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        _, pooled_output = self.bert(
            input_ids=input_ids, attention_mask=attention_mask, return_dict=False
        )
        output = self.drop(pooled_output)
        return self.out(output)

loss_fn = nn.CrossEntropyLoss().to(device)

def build_dataLoader(data, batch_size):
    return DataLoader(data, batch_size=batch_size, shuffle=True)

def build_optimizer(opt, model, lr):
    if opt == 'adam':
        return optim.Adam(model.parameters(), lr=lr, betas=(0.85,0.89), weight_decay=1e-3)
    else:
        return optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-3)

def train_epoch(model, trainloader, optimizer):
    running_loss = 0
    running_acc = 0
    num_batches = len(trainloader)
    for batch_idx, data in enumerate(trainloader):
        print(f"Batch {batch_idx+1} of {num_batches}")
        input_ids = data["input_ids"].to(device)
        attention_mask = data["attention_mask"].to(device)
        labels = data["targets"].to(device)
        optimizer.zero_grad()
        output = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = loss_fn(output, labels)
        running_loss += loss.item()

        loss.backward()
        optimizer.step()

        y_pred = F.softmax(output, dim=1).argmax(dim=1)
        running_acc += ((y_pred == labels).sum()/ labels.shape[0]).item() / num_batches
        wandb.log(
            {
                " (Batch loss)": running_loss,
                " (Batch accuracy)": running_acc,
            }
        )
        if ((batch_idx + 1) % 2) == 0:
            print(
                f"Loss: {running_loss} \tAccuracy: {round(running_acc,4) * 100}%"
            )
    return running_loss / num_batches, running_acc

def build_model(dropout):
    return SentimentClassifier(n_classes=3, p=dropout)

def train(config=None):
    """
    Function to train model and store training results
    Requirements:
        - Data must have been generated before executing this script

    Parameters:
        (OPTIONAL)
        --lr: learning rate
        --epochs: Number of training loops
        --batch_size: Batch size

    Outputs:
        - models/final_model.pth

    """
    with wandb.init(config=config):
        config = wandb.config
        print("Initializing training")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f'Training on {device}')
        model = build_model(config.drop_out).to(device)
        train_set = torch.load('data/processed/train.pth')
        trainloader = build_dataLoader(train_set, config.batch_size)
        optimizer = build_optimizer(config.optimizer, model, config.lr)
        num_epochs = 10
        model.train()
        for i in range(num_epochs):
            print(f"Epoch {i+1} of {num_epochs}")
            
            avg_loss, avg_acc = train_epoch(model, trainloader, optimizer)
            print(
                f"Epoch {i+1} loss: {avg_loss} \tEpoch acc: {avg_acc}"
            )
            wandb.log(
                {
                    "Loss": avg_loss,
                    "Accuracy": avg_acc
                }
            )
        torch.save(
            model,
            f"models/run_v2/model_opt{config.optimizer}_bs{config.batch_size}_do{config.drop_out}_lr{config.lr}.pth",
        )


if __name__ == "__main__":
    wandb.agent(sweep_id, train, count=15)
