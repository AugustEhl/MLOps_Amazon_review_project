import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import os
import sys
#import wandb
import numpy as np
from torchvision import transforms
from transformers import BertModel
import pytorch_lightning as pl
from pytorch_lightning import LightningModule, Trainer, AdamW
from pytorch_lightning.loggers import WandbLogger
wandb_logger = WandbLogger()
#from sklearn import metrics

sys.path.insert(0, os.getcwd() + "/src/data/")
from make_dataset import AmazonData

#Defining a model
class SentimentClassifier(LightningModule):
    def __init__(self):

        super(SentimentClassifier, self).__init__()

        self.bert = BertModel.from_pretrained("bert-base-cased", return_dict=True)
        self.classifier = nn.Linear(self.bert.config.hidden_size, n_classes)
        self.n_training_steps = n_training_steps
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, input_ids, attention_mask, labels=None):
        output = self.bert(input_ids, attention_mask=attention_mask)
        output = self.classifier(output.pooler_output)
        output = torch.sigmoid(output)
        loss = 0
        if labels is not None:
            loss = self.criterion(output, labels)
        return loss, output

    def training_step(self, batch, batch_idx):

        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        loss, outputs = self(input_ids, attention_mask, labels)
        self.log("train_loss", loss, prog_bar=True, logger=True)
        
        return {"loss": loss, "predictions": outputs, "labels": labels}

    def validation_step(self, batch, batch_idx):
      
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        loss, outputs = self(input_ids, attention_mask, labels)
        self.log("val_loss", loss, prog_bar=True, logger=True)
        
        return loss

    def training_epoch_end(self, outputs):
        labels = []
        predictions = []
        for output in outputs:
            for out_labels in output["labels"].to(device):
                labels.append(out_labels)
            for out_predictions in output["predictions"].to(device):
                predictions.append(out_predictions)

            labels = torch.stack(labels).int()
            predictions = torch.stack(predictions)
            self.logger.experiment.log({'logits': wandb.Histrogram(preds)})
        

    def configure_optimizers(self):
        #not sure yet how to finish this part
        optimizer = AdamW(self.parameters(), lr=0.01)


    train_set = torch.load('data/processed/train.pth')
    trainloader = DataLoader(train_set, batch_size=10, shuffle=True, num_workers=4)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    sweep_config = {"method": "random"}
    parameters_dict = {
    "optimizer": {"values": ["adam", "sgd"]},
    "batch_size": {"values": [150, 200, 250]},
    "epochs": {"value": 5},
    "num_workers": {"value": 4},
    "drop_out": {"values": [0.15, 0.25, 0.35]},
    "lr": {"values": [0.01, 0.001]},
}
    sweep_config["parameters"] = parameters_dict
    sweep_id = wandb.sweep(sweep_config, project="Amazon-Reviews", entity="amazonproject")
    table = wandb.Table(columns=["ReviewRating", "PredictedRating"])
    RANDOM_SEED = 42
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    config = wandb.config
    model = SentimentClassifier(n_classes=3).to(device)
    checkpoint_callback = ModelCheckpoint(dirpath="./models", monitor="train_loss", mode="min")
    trainer = Trainer(accelerator='cpu',max_epochs = 3, default_root_dir=os.getcwd(), limit_train_batches=0.2,
        callbacks=[checkpoint_callback],logger=WandbLogger(project="Amazon-project") )
    
    trainer.fit(model,train_dataloader=trainloader)
