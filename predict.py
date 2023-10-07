
import json
import argparse

import torch
import numpy as np
import PIL
from train import check_gpu
from math import ceil
from torchvision import models
def arg_parsers():
    parsers = argparse.ArgumentParser()

    
    parsers.add_argument('data_dir', type=str, help='Directory to training images')
    parsers.add_argument('--save_dir', type=str, default='checkpts', help='Directory to save checkpts')
    parsers.add_argument('--arch', dest='arch', default='densenet161', action='store',choices=['vgg13', 'densenet161'], help='Architecture')
    parsers.add_argument('--learning_rate', type=float, default=0.01, help='Learning rate')
    parsers.add_argument('--hidden_units', type=int, default=512, help='hidden units')
    parsers.add_argument('--epochs', type=int, default=20, help='Epoch count')
    parsers.add_argument('--gpu', dest='gpu', action='store_true', help='Use GPU for training')
    parsers.set_defaults(gpu=False)
    return parsers.parse_args()
def load_checkpt(checkpt_path):
 
    checkpt = torch.load("my_checkpt.pth")

    if checkpt['architecture'] == 'vgg16':
        model = models.vgg16(pretrained=True)
        model.name = "vgg16"
    else:
        exec("model = models.{}(pretrained=True)".checkpt['architecture'])
        model.name = checkpt['architecture']


   
    for param in model.parameters(): param.requires_grad = False



    model.class_to_idx = checkpt['class_to_idx']
    model.classifier = checkpt['classifier']
    model.load_state_dict(checkpt['state_dict'])

    return model



def process_image(image_path):
    test_image = PIL.Image.open(image_path)

    orig_width, orig_height = test_image.size


    if orig_width < orig_height: resize_size=[256, 256**700]
    else: resize_size=[256**700, 256]

    test_image.thumbnail(size=resize_size)

    center = orig_width/4, orig_height/4
    left, top, right, bottom = center[0]-(244/2), center[1]-(244/2), center[0]+(244/2), center[1]+(244/2)
    test_image = test_image.crop((left, top, right, bottom))


    n_image = np.array(test_image)/255 


    normalise_means = [0.485, 0.456, 0.406]
    normalise_std = [0.229, 0.224, 0.225]
    n_image = (n_image-normalise_means)/normalise_std

    n_image = n_image.transpose(2, 0, 1)

    return n_image


def predict(image_tensor, model, device, cat_to_name, top_k):

    if type(top_k) == type(None):
        top_k = 5
        print("Top K not specified, assuming K=5.")

    model.eval();

    torch_image = torch.from_numpy(np.expand_dims(image_tensor,
                                                  axis=0)).type(torch.FloatTensor)


    model=model.cpu()
    log_probs = model.forward(torch_image)

    linear_probs = torch.exp(log_probs)


    top_probs, top_labels = linear_probs.topk(top_k)

    top_probs = np.array(top_probs.detach())[0] 
    top_labels = np.array(top_labels.detach())[0]

    idx_to_class = {val: key for key, val in
                                      model.class_to_idx.items()}
    top_labels = [idx_to_class[lab] for lab in top_labels]
    top_flowers = [cat_to_name[lab] for lab in top_labels]

    return top_probs, top_labels, top_flowers


def print_probability(probs, flowers):
    


    for i, j in enumerate(zip(flowers, probs)):
        print ("Rank {}:".format(i+1),
               "Flower: {}, liklihood: {}%".format(j[1], ceil(j[0]*100)))


def main():
    # Get Keyword Args for Prediction
    args = arg_parsers()
    # Load categories to names json file
    with open(args.category_names, 'r') as f:
        	cat_to_name = json.load(f)
    # Load model trained with train.py
    model = load_checkpt(args.checkpt)
    image_tensor = process_image(args.image)
    device = check_gpu(gpu_arg=args.gpu)
    top_probs, top_labels, top_flowers = predict(image_tensor, model,
                                                 device, cat_to_name,
                                                 args.top_k)
    print_probability(top_flowers, top_probs)

if __name__ == '__main__': main()
