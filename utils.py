from torchvision import datasets, transforms, models
import torch
from torch import nn
from collections import OrderedDict
from PIL import Image
import numpy as np
import torch


def load_model(path, hidden_units):
    checkpoint = torch.load(path)
    model = models.alexnet()
    number_features = checkpoint['in_features']
    model.classifier = nn.Sequential(OrderedDict([
                                      ('dropout1', nn.Dropout(p=0.5)),
                                      ('fc1', nn.Linear(number_features, hidden_units[0])),
                                      ('relu1', nn.ReLU()),
                                      ('dropout2', nn.Dropout(p=0.5)),
                                      ('fc2', nn.Linear(hidden_units[0], hidden_units[1])),  
                                      ('relu2', nn.ReLU()),
                                      ('dropout3', nn.Dropout(p=0.5)),
                                      ('fc3', nn.Linear(hidden_units[1], 102)),
                                      ('output', nn.LogSoftmax(dim=1)),
                                  ]))
    
    model.epoch=checkpoint['epoch']
    model.class_to_idx=checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['state_dict'])
    return model
    
    
def process_image(image):


    
    pil_image = Image.open(f'{image}' + '.jpg')
    
    transform = transforms.Compose([transforms.Resize(256),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], 
                                                         [0.229, 0.224, 0.225])]) 
    
    
    pil_tfd = transform(pil_image)
    
    
    array_im_tfd = np.array(pil_tfd)
    
    return array_im_tfd
    
    
    

def predict(image_path, model, gpu, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # TODO: Implement the code to predict the class from an image file   
    
    
    predictor = model.cpu()
    imgage = process_image(image_path)
    
    image_tensor = torch.from_numpy(imgage).type(torch.FloatTensor)
    
    final_image = image_tensor.unsqueeze_(0)
    predictor.eval()
    with torch.no_grad():
        output = predictor.forward(final_image)

    probabilities = torch.exp(output)
    probabilities_top = probabilities.topk(topk)[0]
    index_top = probabilities.topk(topk)[1]
    
    
    probabilities_top_list = np.array(probabilities_top)[0]
    index_top_list = np.array(index_top[0])
    
    
    class_to_index = predictor.class_to_idx
    
    index_to_class = {x: y for y, x in class_to_index.items()}

    
    classes_top_list = []
    for index in index_top_list:
        classes_top_list += [index_to_class[index]]
        
    return probabilities_top_list, classes_top_list