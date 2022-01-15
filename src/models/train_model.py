from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup
import torch
import numpy as np
from torch import nn, optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
import wandb
from torch.profiler import profile, record_function, ProfilerActivity, tensorboard_trace_handler
import pprint
import click
import os
import sys
sys.path.insert(0,os.getcwd() + '/src/data/')
from make_dataset import AmazonData

# Setting seed
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

loss_fn = nn.CrossEntropyLoss().to(device)

class SentimentClassifier(nn.Module):

  def __init__(self, n_classes, dropout):
    super(SentimentClassifier, self).__init__()
    self.bert = BertModel.from_pretrained('bert-base-cased')
    self.drop = nn.Dropout(p=dropout)
    self.out = nn.Linear(self.bert.config.hidden_size, n_classes)
  
  def forward(self, input_ids, attention_mask):
    _, pooled_output = self.bert(
      input_ids=input_ids,
      attention_mask=attention_mask
    )
    output = self.drop(pooled_output)
    return self.out(output)

sweep_config = {
    'method': 'random'
}

parameters_dict = {
    'optimizer': {
        'values': ['adam','sgd']
        },
    'batch_size': {
          'values': [35, 45, 65]
        },
    'epochs': {
        'value': 50
        },
    'dropout': {
        'values': [0.15, 0.25, 0.35]
    },
    'lr': {
        'values': [0.01,0.001]
    }
}
sweep_config['parameters'] = parameters_dict
pprint.pprint(sweep_config)

sweep_id = wandb.sweep(sweep_config, project="Amazon-Reviews")

table = wandb.Table(columns=["ReviewRating", "PredictedRating"])

@click.command()
@click.argument('input_filepath', type=click.Path())
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
    wandb.init(project="Amazon-Reviews", config=config)
    config = wandb.config
    table = wandb.Table(columns=["ReviewRating", "PredictedRating"])
    print("Initializing training")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SentimentClassifier(n_classes=3, dropout=config.dropout).to(device)
    train_set = torch.load(input_filepath)
    trainloader = DataLoader(train_set, batch_size = config.batch_size, shuffle=True)
    opt = config.optimizer
    if opt == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=config.lr, betas=(0.85,0.89), weight_decay=1e-3)
    else:
        optimizer = optim.SGD(model.parameters(), lr=config.lr, momentum=0.9, weight_decay=1e-3)
    model.train()
    for i in range(config.epochs):
        print(f'Epoch {i+1} of {config.epochs}')
        running_loss = 0
        running_acc = 0
        with profile(
        activities=[ProfilerActivity.CPU],
        schedule=torch.profiler.schedule(
            wait=2,
            warmup=2,
            active=6,
            repeat=1),
        on_trace_ready=tensorboard_trace_handler('tb_trace_vae.pt.trace.json'),
        with_stack=True
    ) as profiler:
            for batch_idx, data in enumerate(trainloader):
                data = data.to(device)
                input_ids = data['input_ids'].to(device)
                attention_mask = data["attention_mask"].to(device)
                labels = data["targets"].to(device)
                optimizer.zero_grad()
                with record_function("model_inference"):
                    output = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = loss_fn(output, labels)
                running_loss += loss.item()
                
                loss.backward()
                optimizer.step()

                y_pred = nn.Softmax(output, dim=1).argmax(dim=1)
                running_acc += ((y_pred == labels).sum() / labels.shape[0]).item()
                if (batch_idx % 5) == 0:
                    print(f'Batch {batch_idx} loss: {running_loss / len(trainloader)} \tBatch acc.: {running_acc / len(trainloader)}')
                    random_review = np.random.randint(labels.shape[0])
                    table.add_data(labels[random_review], y_pred[random_review])
            print(f'Epoch {i+1} loss: {running_loss / len(trainloader)} \tEpoch acc.: {running_acc / len(trainloader)}')
            wandb.log({'Loss': running_loss / len(trainloader),
                'Accuracy (%)': running_acc / len(trainloader),
                'Clases': table})



if __name__ == '__main__':
    wandb.agent(sweep_id, train)
