import sys
import argparse
import torch
import numpy as np
from torchvision import datasets, transforms, models
from torch import nn
from collections import OrderedDict
from torch import optim
from PIL import Image
import json
import utils

def validate(model, test_loader, gpu, criterion):
    accuracy = 0
    test_loss = 0
    with torch.no_grad():
        model.eval()
        for images, labels in test_loader:
            if gpu:
                images, labels = images.to('cuda'), labels.to('cuda')
            else:
                pass

            output = model.forward(images)
            test_loss += criterion(output, labels).data[0]
            ps = torch.exp(output).data 
            equality = (labels.data == ps.max(1)[1])
            accuracy += equality.type_as(torch.FloatTensor()).mean()

    return test_loss/len(test_loader), accuracy/len(test_loader)


def main():
    parser = argparse.ArgumentParser(description='Train.py')
    parser.add_argument('--gpu', type=bool, default=False, help='Enable or Disable CUDA')
    parser.add_argument('--arch', type=str, default='alexnet', help='architecture')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--hidden_units', type=str, default='1024,512', help='hidden units')
    parser.add_argument('--epochs', type=int, default=15, help='number of epochs')
    parser.add_argument('--data_dir', type=str, default='flowers', help='dataset directory',required=True)
    parser.add_argument('--save_dir' , type=str, default='./', help='checkpoint directory path')
    args = parser.parse_args()
#     print(args)
    
    gpu = args.gpu
    arch = args.arch
    lr = args.lr
    hidden_units = args.hidden_units
    hidden_units = hidden_units.split(',')
    hidden_units = [int(layer) for layer in hidden_units]
    epochs = args.epochs
    data_dir = args.data_dir
    save_dir = args.save_dir
    
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], 
                                                           [0.229, 0.224, 0.225])]) 

    train_dataset =datasets.ImageFolder(train_dir,transform=train_transforms) 
    val_dataset =datasets.ImageFolder(valid_dir,transform=test_transforms) 
    test_dataset =datasets.ImageFolder(test_dir,transform=test_transforms) 

    trainloader = torch.utils.data.DataLoader(train_dataset,batch_size=64,shuffle=True)
    valloader= torch.utils.data.DataLoader(val_dataset,batch_size=64)
    testloader= torch.utils.data.DataLoader(test_dataset,batch_size=64)
    
        
    number_features = 0
    if arch == 'alexnet':
        model=models.alexnet(pretrained=True)
        for parm in model.parameters():
            parm.requires_grid=False
        number_features = 9216
    else:
        model=models.alexnet(pretrained=True)
        for parm in model.parameters():
            parm.requires_grid=False
        number_features = 9216
        
    my_classifier = nn.Sequential(OrderedDict([
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
    model.classifier = my_classifier
    criterion = nn.NLLLoss()
    optimizer= optim.SGD(model.classifier.parameters(),lr=lr)
    
    if gpu:
        model.to('cuda')
    
    
    best_accuracy=0
    for e in range(epochs):
        running_loss=0;
        for images, labels in trainloader:
            if gpu:
                images, labels = images.to('cuda'), labels.to('cuda')
            else:
                pass
            optimizer.zero_grad();
            output = model(images)
            loss = criterion(output,labels)
            running_loss += loss.item()
            loss.backward()
            optimizer.step()
        else:
            validation_loss, accuracy = validate(model, valloader, gpu, criterion)

            print("Epoch: {}/{} ".format(e+1, epochs),
                            "Training Loss: {:.3f} ".format(running_loss/len(trainloader)),
                            "Validation Loss: {:.3f} ".format(validation_loss),
                            "Validation Accuracy: {:.3f}".format(accuracy))
    
    model.class_to_idx = train_dataset.class_to_idx
    test_loss, accuracy = validate(model, testloader, gpu, criterion)
    print("Test. Accuracy: {:.3f}".format(accuracy))
    print("Test. Loss: {:.3f}".format(test_loss))
    
    top_k_prob, top_k_classes = utils.predict('./flowers/test/15/image_06374',model,gpu,5)

    print(top_k_prob)
    print(top_k_classes)
    
   
    
    model.class_to_idx = train_dataset.class_to_idx
    class_to_idx = train_dataset.class_to_idx
    model.epoch=epochs
    checkpoint_state = {
                   'state_dict': model.state_dict(),
                   'epoch': model.epoch,
                   'batch_size': trainloader.batch_size,
                   'optimizer_state':optimizer.state_dict(),
                   'class_to_idx': model.class_to_idx,
                   'learning_rate': 0.01,
                   'output_size': 102,
                   'input_size':(224,224,3),
                   'in_features':number_features,
                 }
    torch.save(checkpoint_state, save_dir+'alexnet.pth')
   
 
    
if __name__ == '__main__':
    main()


