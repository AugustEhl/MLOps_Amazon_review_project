import os
import pprint
import sys
import warnings
import pickle

warnings.filterwarnings("ignore", category=FutureWarning)

import click
import numpy as np
from sklearn.metrics import confusion_matrix
import torch
import torch.nn.functional as F
import wandb
from torch import nn, optim
from torch.utils.data import DataLoader
from transformers import BertModel
from train_model import SentimentClassifier

sys.path.insert(0, os.getcwd() + "/src/data/")

from make_dataset import AmazonData
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

@click.command()
@click.argument("batch_size", type=click.INT, default=20)
@click.argument("data", type=click.Path(), default="data/processed/test.pth")
@click.argument("load_model_path", type = click.Path(), default="models/model_optsgd_bs20_do0.15_lr0.01.pth")
def evaluate(batch_size, data, load_model_path):

    model = torch.load(load_model_path)
    test_set = torch.load(data)
    testloader = torch.utils.data.DataLoader(
        test_set, batch_size=batch_size, shuffle=False
    )

    test_acc = []
    model.eval()
    test_acc= 0
    for _, data in enumerate(testloader):
        input_ids = data["input_ids"].to(device)
        attention_mask = data["attention_mask"].to(device)
        labels = data["targets"].to(device)
        with torch.no_grad():
            output = model(input_ids=input_ids, attention_mask=attention_mask)
        y_pred = F.softmax(output, dim=1).argmax(dim=1)
        test_acc += ((y_pred == labels).sum()/ labels.shape[0]).item()

    test_acc = test_acc / len(testloader) * 100
    print(f'Accuracy: {round(test_acc,4)}%')

if __name__ == "__main__":
    evaluate()