from trainMVShapeClassifier import trainMVShapeClassifier
from testMVImageClassifier import testMVImageClassifier
import torch
import pickle as p
import os

# Get absolute path to dataset folder
script_dir = os.path.dirname(os.path.abspath(__file__))
dataset_dir = os.path.join(script_dir, '..', 'dataset')
train_path = os.path.join(dataset_dir, 'train')
test_path = os.path.join(dataset_dir, 'test')

# TRAIN
model, info = trainMVShapeClassifier(train_path, cuda=torch.cuda.is_available(), verbose=True)

# # TO SAVE TIME for just testing code, uncomment the following 2 lines to load your pre-trained model
model = torch.load('model/model_epoch_29.pth', map_location=lambda storage, location: storage, weights_only=False)["model"]
info = p.load( open( "info.p", "rb" ) )

# TEST
testMVImageClassifier(test_path, model, info, pooling='mean', cuda=torch.cuda.is_available(), verbose=False)
testMVImageClassifier(test_path, model, info, pooling='max', cuda=torch.cuda.is_available(), verbose=False)