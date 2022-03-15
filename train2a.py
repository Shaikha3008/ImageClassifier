import numpy as np
#manipulate datasets !
import json
##PYTHON IMAGING LIB
import PIL
from PIL import Image
import seaborn as sns
import matplotlib.pyplot as plt
##https://pytorch.org/tutorials/beginner/nn_tutorial.html using this tutorial to understand which libraries 
#using this pytorch tutorial for the libs
import torch
from torch import nn #help us in creating and training of the neural network. 
from torch import tensor
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms, models
#import torchvision.modules as modules
import argparse

#OrderedDict preserves the order in which the keys are inserted. 
from collections import OrderedDict 

parser = argparse.ArgumentParser(
    description='Some arg_pass for my project',
)
parser.add_argument(type=str, dest='data_dir')
parser.add_argument('--save_dir', dest='save_dir', type=str, default='')
parser.add_argument('--arch', dest='arch', type=str, default='densenet161')
parser.add_argument('--learning_rate', dest='learning_rate', type=float, default=0.001)
parser.add_argument('--hidden_units', dest='hidden_units', type=int, default=512)
parser.add_argument('--epochs', dest='epochs', type=int, default=3)
parser.add_argument('--gpu', action='store_const', const='gpu')

args = parser.parse_args()

data_dir = args.data_dir
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'


data_transforms = { 
    'train': transforms.Compose([
        transforms.RandomResizedCrop(size=224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(30),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])]),
    'test': transforms.Compose([
        transforms.Resize(size=256),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])]),
    'valid': transforms.Compose([
        transforms.Resize(size=256),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])    
}



image_datasets = {
    'train': datasets.ImageFolder(train_dir, transform=data_transforms['train']),
    'test': datasets.ImageFolder(test_dir, transform=data_transforms['test']),
    'valid': datasets.ImageFolder(valid_dir, transform=data_transforms['valid'])}

trainloader = torch.utils.data.DataLoader(image_datasets['train'], batch_size=64, shuffle=True)
testloader = torch.utils.data.DataLoader(image_datasets['test'], batch_size=64, shuffle=True)
validationloader = torch.utils.data.DataLoader(image_datasets['valid'], batch_size=64,shuffle=True)

import json

with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)
    
# Use GPU if it's available.. and locate device from user if not
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    if args.gpu!='gpu':
        print('GPU disabled')
        device='cpu'
    if args.gpu=='gpu':
        print('GPU enabled')
        device='cuda'
else:
    print('GPU is not available, using CPU now')
    device='cpu'
    
    
#putting two options for user.. vgg16 is the default and most trusted and worked on
hidden_units=args.hidden_units

if args.arch=='densenet161':    
    module = models.densenet161(pretrained=True)
    # Freeze parameters so we don't backprop through them
    for param in module.parameters():
        param.requires_grad = False
    module.classifier = nn.Sequential(nn.Linear(2208, hidden_units),
                                 nn.ReLU(),
                                 nn.Linear(hidden_units, int(hidden_units/2)),
                                 nn.ReLU(),                                 
                                 nn.Linear(int(hidden_units/2), 102),
                                 nn.Dropout(0.2),
                                 nn.LogSoftmax(dim=1))
    
elif args.arch=="vgg16":
    module =models.vgg16(pretrained=True)
    # Freeze parameters so we don't backprop through them
    for param in module.parameters():
        param.requires_grad = False
    module.classifier = nn.Sequential(nn.Linear(25088, hidden_units),
                                   nn.ReLU(),
                                   nn.Linear(hidden_units, int(hidden_units/2)),
                                   nn.ReLU(),
                                   nn.Dropout(0.2),                                 
                                   nn.Linear(int(hidden_units/2), 102),
                                   nn.Dropout(0.2),
                                   nn.LogSoftmax(dim=1))
                               
criterion = nn.NLLLoss()
learning_rate=args.learning_rate
optimizer = optim.Adam(module.classifier.parameters(), lr=learning_rate)
module.to(device)



#training started...
epochs = args.epochs
steps = 0
runloss = 0
print_every = 5
for epoch in range(epochs):
    for inputs, labels in trainloader:
        steps += 1
        # Move input and label tensors to the default device
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        modlog = module.forward(inputs)
        loss = criterion(modlog, labels)
        loss.backward()
        optimizer.step()

        runloss += loss.item()
        
        if steps % print_every == 0:
            testloss = 0
            accuracy = 0
            module.eval()
            with torch.no_grad():
                for inputs, labels in validationloader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    modlog = module.forward(inputs)
                    batch_loss = criterion(modlog, labels)
                    
                    testloss += batch_loss.item()
                    
                    # Calculate accuracy
                    exps = torch.exp(modlog)
                    top_pre, top_class = exps.topk(1, dim=1)
                    matchs = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(matchs.type(torch.FloatTensor)).item()
                    
            print(f"Epoch {epoch+1}/{epochs}.."
                  f"Loss on Training is: {runloss/print_every:.3f}.. "
                  f"Validation loss: {testloss/len(validationloader):.3f}.. "
                  f"Validation accuracy is.. *100 for % :): {accuracy/len(validationloader):.3f}")
            runloss = 0
            module.train()
#save the checkpoint!!!! very very very important!!!
#to use in predict later on
checkpoint = {'architecture':args.arch,
              'module.classifier':module.classifier,
              'module.class_to_idx':image_datasets['train'].class_to_idx,
              'state_dict': module.state_dict(),
              'epoch': epoch,
              'optimizer_state_dict': optimizer.state_dict()  }

if args.save_dir!='':
   torch.save(checkpoint, args.save_dir + '/some.pth')
print("training is done :)")