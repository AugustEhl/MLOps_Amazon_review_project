import os
import pprint
import sys

import click
import numpy as np
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
    "optimizer": {"values": ["adam", "sgd"]},
    "batch_size": {"values": [150, 200, 250]},
    "epochs": {"value": 5},
    "drop_out": {"values": [0.15, 0.25, 0.35]},
    "lr": {"values": [0.01, 0.001]},
}
sweep_config["parameters"] = parameters_dict
pprint.pprint(sweep_config)

sweep_id = wandb.sweep(sweep_config, project="Amazon-Reviews")

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


@click.command()
@click.argument("input_filepath", type=click.Path())
def train(input_filepath, config=None):
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
        model = SentimentClassifier(n_classes=3, p=config.drop_out).to(device)
        train_set = torch.load(input_filepath)
        trainloader = DataLoader(train_set, batch_size=config.batch_size, shuffle=True)
        opt = config.optimizer
        if opt == "adam":
            optimizer = optim.Adam(
                model.parameters(), lr=config.lr, betas=(0.85, 0.89), weight_decay=1e-3
            )
        else:
            optimizer = optim.SGD(
                model.parameters(), lr=config.lr, momentum=0.9, weight_decay=1e-3
            )
        model.train()
        for i in range(config.epochs):
            print(f"Epoch {i+1} of {config.epochs}")
            running_loss = 0
            running_acc = 0
            for batch_idx, data in enumerate(trainloader):
                print(f"Batch {batch_idx+1} of {len(trainloader)}")
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
                running_acc += (
                    ((y_pred == labels).sum() / labels.shape[0]) / len(trainloader)
                ).item()
                wandb.log(
                    {
                        "Epoch_" + str(i + 1) + " (Batch loss)": running_loss,
                        "Epoch_" + str(i + 1) + " (Batch accuracy)": running_acc,
                    }
                )
                if ((batch_idx + 1) % 5) == 0:
                    print(f"Loss: {running_loss} \tAccuracy.: {running_acc * 100}%")
                    random_review = np.random.randint(labels.shape[0])
                    table.add_data(labels[random_review], y_pred[random_review])
            print(
                f"Epoch {i+1} loss: {running_loss / len(trainloader)} \tEpoch acc.: {running_acc}"
            )
            wandb.log(
                {
                    "Loss": running_loss / len(trainloader),
                    "Accuracy (%)": running_acc,
                    "Classes": table,
                }
            )
        torch.save(
            model,
            f"models/model_opt{config.optimizer}_bs{config.batch_size}_do{config.drop_out}_lr{config.lr}.pth",
        )


if __name__ == "__main__":
    wandb.agent(sweep_id, train, count=5)
