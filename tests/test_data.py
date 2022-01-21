import pytest
from tests import _PATH_DATA
from src.data.make_dataset import *


"Testing the size of the dataset"
@pytest.mark.skipif(not os.path.exists(_PATH_DATA), reason="Data does not exist")
def test_data_length():
    data= load_dataset('data/raw/reviews_Amazon_Instant_Video_5.json.gz')
    assert len(data) == 37126, "Dataset did not have the correct number of samples"

"Testing the size of the dataset before and after preprocessing"
@pytest.mark.skipif(not os.path.exists(_PATH_DATA), reason="Data does not exist")
def test_size():
    #before preprocessing
    data= load_dataset('data/raw/reviews_Amazon_Instant_Video_5.json.gz')
    size_data= len(data)
    #after preprocessing
    processed_data_train, processed_data_test = data_preprocessing(_PATH_DATA)
    size_test = len(processed_data_train)
    size_train= len(processed_data_test)
    size_processed_data=size_test+size_train

    assert size_processed_data == size_data, 'size of datasets after preprocessing is incorrect'

"Testing if the labels get only 3 different values"
@pytest.mark.skipif(not os.path.exists(_PATH_DATA), reason="Data does not exist")
def test_testtargets():
    processed_data_train, processed_data_test = data_preprocessing(_PATH_DATA)

    output_train = []
    for x in processed_data_train.targets:
        if x not in output_train:
            output_train.append(x)

    output_test = []
    for x in processed_data_test.targets:
        if x not in output_test:
            output_test.append(x)

    assert len(output_train) == 3, "Train data did not have the correct number of labels"
    assert len(output_test) == 3, "Test data did not have the correct number of labels"

"Testing the tokenizer"
def test_tokenizer():
    from transformers import BertTokenizer
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    tokens=tokenizer.tokenize("Welcome to the exam session")
    assert tokens==['welcome', 'to', 'the', 'exam', 'session'], 'Wrong output from the tokanizer'
    
