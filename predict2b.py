import argparse
from PIL import Image
from matplotlib.pyplot import imshow as ish
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import json

import argparse

#checkpoint = 'some.pth'
device='cpu'
parser = argparse.ArgumentParser(
    description='Some arg_pass for predicting',
)
parser.add_argument(type=str, dest='input')
#parser.add_argument(type=str, dest='checkpoint')
#parser.add_argument('--arch', dest='arch', type=str, default='architecture')
parser.add_argument('checkpoint')
parser.add_argument('--gpu', action='store_const', const='gpu')
parser.add_argument('--top_k', dest='top_k',type=int, default=1)
parser.add_argument('--category_names', dest='category_names',type=str, default=None)
args = parser.parse_args()

if torch.cuda.is_available():
    if args.gpu!='gpu':
        print('GPU disabled')
        device='cpu'
    if args.gpu=='gpu':
        print('GPU enabled')
        device='cuda'
else:
    print('GPU is not available, switching CPU')
    device='cpu'
    

    
#data_dir = 'flowers'
#train_dir = data_dir + '/train'
#train_dataset = datasets.ImageFolder(train_dir)

def load(checkpoint_path):
    status = torch.load(checkpoint_path)
  
    
    if status['architecture']=='densenet161':    
    
        model=models.densenet161(pretrained=True)
        for p in model.parameters():
            p.requires_grad = False  
        model.classifier = status['module.classifier']
        model.class_to_idx = status['module.class_to_idx']
        model.load_state_dict(status['state_dict'])

    elif status['architecture']=='vgg16':    
    
        model=models.vgg16(pretrained=True)
        for p in model.parameters():
            p.requires_grad = False  
        model.classifier = status['module.classifier']
        model.class_to_idx = status['module.class_to_idx']
        model.load_state_dict(status['state_dict'])
    #optimizer.load_status_dict(status['optimizer_state_dict'])   
    return model

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
#using PIL!   '''

    image = Image.open(image).convert('RGB')
    print(image.format, image.size, image.mode)
    width, length = image.size
#https://note.nkmk.me/en/python-pillow-square-circle-thumbnail/#:~:text=source%3A%20pillow_thumbnail.py-,Add%20padding%20to%20make%20a%20rectangle%20square,paste%20it%20with%20paste()%20.
#to help make the image square i used this source 
#https://stackoverflow.com/questions/44231209/resize-rectangular-image-to-square-keeping-ratio-and-fill-background-with-black
 #First, resize the images where the shortest side is 256 pixels, keeping the aspect ratio. This can be done with the thumbnail or resize methods. Then you'll need to crop out the center 224x224 portion of the image.  

    """  
    if they are equal image ration is one, if one is bigger than other then try to equal them out 
    image_ratio = length / width 
    if image_ratio > 1 :
        imgage = image.resize(256,round(256/image_ratio))   
    else:
        imgage = image.resize(round(image_ratio*256),256)
        """

    # Find shorter size and create settings to crop shortest side to 256
    if width < length:
        size=[256, (256*1000000)]
    else: 
        size=[(256*1000000), 256]
        
    #image.resize(size) does not work! dunno why!!!!
    image.thumbnail(size)
    size=224
#get 1 fourth of the image up width and length and then use it for cropping
    middle1 = width/4
    middle2= length/4
 #now use that middle part of each width and length to get for parts of the image   
    down = middle2+(size/2)
    right = middle1+(size/2)
    left = middle1-(size/2)
    up = middle2-(size/2)
#crop each part of the finding 
    image = image.crop((left, up, right, down))
    
#Color channels of images are typically encoded as integers 0-255, but the model expected floats 0-1. You'll need to convert the values. It's easiest with a Numpy array, which you can get from a PIL image like so np_image = np.array(pil_image). 
    numpy_image = np.array(image)/256 

    numpy_image = numpy_image.transpose(2, 0, 1)
    
    return numpy_image
    
def predict(image_path, model, top_k):
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    
    image_path: string. Path to image, directly to image and not to folder.
    model: pytorch neural network.
    top_k: integer. The top K classes to be calculated
    
    returns top_probabilities(k), top_labels
    '''
 #https://genomicsclass.github.io/book/pages/machine_learning.html
#this sourxe suggest d
#how to get top 5?? https://forums.fast.ai/t/get-top-5-predictions-in-image-classifier-solved/76249/8
    model.to(device)
    model.eval();
#https://stackoverflow.com/questions/49827838/keras-top-5-predictions
    torch_image = torch.from_numpy(np.expand_dims(process_image(image_path), 
                                                  axis=0)).type(torch.FloatTensor).to(device)
                                 
    process_1 = model.forward(torch_image)
    process_2 = torch.exp(process_1)
    process_3 = process_2.topk(top_k)
    ##bestfit, namebestfit = ((torch.exp(vgg16.forward(torch_image))).topk(top_k))
    bestfit, namebestfit = process_3  
    
   
   #The Python detach() method is used to separate the underlying raw stream from the buffer and return it. After the raw stream has been detached, the buffer is in an unusable state.
    bestfit = np.array(bestfit.detach())[0] 
    namebestfit = np.array(namebestfit.detach())[0]
                                   
    # You need to convert from these indices to the actual class labels using class_to_idx which hopefully you added to the model or from an ImageFolder you used to load the data
    idx_to_class = {val: key for key, val in model.class_to_idx.items()}   
    namebestfit = [idx_to_class[lab] for lab in namebestfit]
    flower1 = [cat_to_name[lab] for lab in namebestfit]
    #cinverted to class 
   
    return bestfit, namebestfit, flower1
    
model=load(args.checkpoint)
idx_to_class = {v: k for k, v in model.class_to_idx.items()}

probs,classes,_=predict(args.input, model, args.top_k)

print('\nMost matched image class and probability :)')
classes, probs in zip(classes, probs)
print("Image Class: {}, Probability: {}\n".format(classes[0], probs[0]))
#print("Image Class: {}, Probability: {}".format(names, probs))

if args.top_k >1:
    print('TOP IMAGE(s) CLASSES AND PROBABILITIES FOUND: :)')
    for probs_, classes_ in zip(probs, classes):   
        print("The Image Class is: {}, and it's Probability: {}".format(classes_, probs_))
    
if args.category_names !=None:
    print('\nIMAGE NAME(S) AND PROBABILITY')
    with open(args.category_names, 'r') as f:
        cat_to_name = json.load(f)    
    names = [cat_to_name[c] for c in classes]  
    for names_map, probs_map in zip(names,probs):  
        print("{}: {}".format(names_map,probs_map))
