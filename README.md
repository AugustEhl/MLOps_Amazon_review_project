# MLOps_Amazon_review_project

## Predicting Amazon ratings of video games based on review comments

Overall goal: Predict Amazon rating of video games from review comment using PyTorch transformers

PyTorch ecosystem: Transformers

Data: Amazon Reviews (https://nijianmo.github.io/amazon/index.html)

### Project description:
In this project we work with Amazon Reviews data. We will predict ratings of various products in the dataset based on the review comment. To do this we will use the PyTorch ecosystem transformers from Huggingface git repo (https://github.com/huggingface/transformers). For this project, we will utilize various tools to make organized and efficient training of deep learning models in Python using various tools presented in the course (https://skaftenicki.github.io/dtu_mlops/).

## Running the project:
To run the project one has to make sure to get the data this can be done with the following command:
 - make data
"make data" check if there is any data in data/raw and if not it will automatically download it to the folder before preprocessing.

When the data has been downloaded and processed it can be found in data/processed/ where a train and test set can be found stored as .pth

Then training of the model can start by executing from the root of this repo:
 - python src/models/train_model.py
NOTE: You must be logged in to Wandb and be part of the team to run this script. If you are not a member you have to remove the "entity" argument from line 46 in train_model.py.
The model will make 15 models with different sets of hyperparameters based on batch size, learning rate, dropout rates and whether to use SGD or Adam as an optimizer. The trained models are trained for 10 epochs for each hyperparameter sweep.

Lastly ,the model can be validated on the test set by executing from root of the repo:
 - python src/models/predict_model.py

We have already run all this and experiment logs can be found here:
 - "https://wandb.ai/amazonproject/Amazon-Reviews-v2/reports/Amazon_video_ratings--VmlldzoxNDY2MDEz"
Furthermore, profiling of trainscript can be visualized with snakeviz with following command:
 - snakeviz train.prof  
