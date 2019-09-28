import torch
import numpy as np
from torchvision import datasets, transforms, models
from torch import nn
from collections import OrderedDict
from torch import optim
from PIL import Image
import sys
import argparse
import json
import utils

def main():
    parser = argparse.ArgumentParser(description='Predict an image using neural network')
    parser.add_argument('--image_path', type=str,
                        default = './flowers/test/15/image_06374',help='path to image.')
    parser.add_argument('--save_dir', type=str, 
                        default = './alexnet.pth',help='path to checkpoint in.')
    parser.add_argument('--top_k', type=int,
                        default=5,help='number of top most likely classes.')
    parser.add_argument('--cat_to_name', type=str,
                    default = 'cat_to_name.json',help='path to cattegories.')
    parser.add_argument('--gpu', type=bool, default=False,
                    help='Turn GPU mode on or off.')
    parser.add_argument('--arch', type=str, default='alexnet', help='architecture')
    parser.add_argument('--hidden_units', type=str, default='1024,512', help='hidden units')
    
    args = parser.parse_args()
    image_path = args.image_path
    save_dir = args.save_dir
    top_k = args.top_k
    cat_to_name = args.cat_to_name
    gpu = args.gpu
    arch = args.arch
    hidden_units = args.hidden_units
    hidden_units = hidden_units.split(',')
    hidden_units = [int(layer) for layer in hidden_units]
    with open(cat_to_name, 'r') as f:
        flower_to_name = json.load(f)
    
    model = utils.load_model(save_dir,hidden_units)
    
    top_k_prob, top_k_classes = utils.predict(image_path,model,gpu,top_k)

    print(top_k_prob)
    print(top_k_classes)
    
    i=0
    classes_label = []
    for i in top_k_classes:
        classes_label.append(flower_to_name.get(str(i)))
    i=0
    while i < top_k:
        print("{} with a probability of {}".format(classes_label[i], top_k_prob[i]))
        i += 1
    
if __name__ == '__main__':
    main()