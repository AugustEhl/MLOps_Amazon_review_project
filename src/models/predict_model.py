# Importing modules
import os
import sys
import warnings

# Remove warnings from Bert
warnings.filterwarnings("ignore", category=FutureWarning)

# Loading modules
import click
import torch
import torch.nn.functional as F

# Append data path
sys.path.insert(0, os.getcwd() + "/src/data/")

# Load class object from make_data
from make_dataset import AmazonData

# Set device to cuda or cpu
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Add arguments
@click.command()
@click.argument("batch_size", type=click.INT, default=20)
@click.argument("data", type=click.Path(), default="data/processed/test.pth")
@click.argument("load_model_path", type = click.Path(), default="models/model_optsgd_bs20_do0.15_lr0.01.pth")
def evaluate(batch_size: int, data: str, load_model_path: str):
    # load model
    model = torch.load(load_model_path)
    # load test set
    test_set = torch.load(data)
    # create testloader
    testloader = torch.utils.data.DataLoader(
        test_set, batch_size=batch_size, shuffle=False
    )
    # Create empty list to store data
    test_acc = []
    # Deactivate model traininger
    model.eval()
    test_acc= 0
    for _, data in enumerate(testloader):
        input_ids = data["input_ids"].to(device)
        attention_mask = data["attention_mask"].to(device)
        labels = data["targets"].to(device)
        # turn off gradients and make predictions
        with torch.no_grad():
            output = model(input_ids=input_ids, attention_mask=attention_mask)
        # softmax predictions
        y_pred = F.softmax(output, dim=1).argmax(dim=1)
        test_acc += ((y_pred == labels).sum()/ labels.shape[0]).item()

    test_acc = test_acc / len(testloader) * 100
    # Print accuracies
    print(f'Accuracy: {round(test_acc,4)}%')

# Python
if __name__ == "__main__":
    # Evaluate
    evaluate()