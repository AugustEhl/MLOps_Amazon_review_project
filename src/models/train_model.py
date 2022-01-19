import os
import pprint
import sys
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
from sklearn.metrics import confusion_matrix
import torch
import torch.nn.functional as F
import wandb
from torch import nn, optim
from torch.utils.data import DataLoader
from transformers import BertModel

sys.path.insert(0, os.getcwd() + "/src/data/")
from make_dataset import AmazonData

# Setting seed
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

sweep_config = {"method": "random"}

parameters_dict = {
    "optimizer": {"value": "sgd"},
    "batch_size": {"value": 20},
    "drop_out": {"value": 0.15},
    "lr": {"value": 0.01},
}
sweep_config["parameters"] = parameters_dict
pprint.pprint(sweep_config)

sweep_id = wandb.sweep(sweep_config, project="Amazon-Reviews", entity="amazonproject")

table = wandb.Table(columns=["ReviewRating", "PredictedRating"])


RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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
    return DataLoader(data, batch_size=batch_size, shuffle=True, num_workers=8)

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
                "Epoch_" + str(i + 1) + " (Batch loss)": running_loss,
                "Epoch_" + str(i + 1) + " (Batch accuracy)": running_acc,
            }
        )
        if ((batch_idx + 1) % 2) == 0:
            print(
                f"Loss: {running_loss} \tAccuracy: {round(running_acc,4) * 100}%"
            )
            random_review = np.random.randint(labels.shape[0])
            table.add_data(labels[random_review], y_pred[random_review])
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
        print(device)
        model = build_model(config.drop_out).to(device)
        train_set = torch.load('data/processed/train.pth')
        trainloader = build_dataLoader(train_set, config.batch_size)
        optimizer = build_optimizer(config.optimizer, model, config.lr)
        num_epochs = 3
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
                    "Accuracy": avg_acc,
                    "Classes": table,
                }
            )
        torch.save(
            model,
            f"models/model_opt{config.optimizer}_bs{config.batch_size}_do{config.drop_out}_lr{config.lr}.pth",
        )


if __name__ == "__main__":
    wandb.agent(sweep_id, train, count=1)
