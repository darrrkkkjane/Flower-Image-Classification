from load_checkpoint import load_checkpoint
from process_image import process_image
from get_input_args_predict import get_input_args
from predict_function import predict
import json
import numpy as np
import torch
from torch import nn
from torch import optim 
from PIL import Image
import random
from torchvision import models, datasets, transforms

# Example call:
#     python predict.py flowers/test/100/image_07939.jpg checkpoint.pth --gpu cpu --json_file cat_to_name.json --num_top_prob 10

def main():
    
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''

    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
    
    in_arg = get_input_args()
    
    model = load_checkpoint(in_arg.checkpoint)      
    image_path = in_arg.image_path 
    

    prob, predicted_class = predict(image_path, model, in_arg.num_top_prob)
    pred = predicted_class
    pred = pred.to('cpu')
    pred = pred.numpy().squeeze()

    prob = prob.to('cpu')
    prob = prob.numpy().squeeze()
    
    print(f'Flower name: {cat_to_name[str(pred[0])]}')
    print(f'Class probability: {max(prob)}')
    
    pred = list(pred)[::-1]
    prob = list(prob)[::-1]
    
    return cat_to_name[str(pred[len(pred)-1])], prob[len(prob)-1]


   
    
# Call to main function to run the program
if __name__ == "__main__":
    main()    
    