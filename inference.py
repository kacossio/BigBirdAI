import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
import os
import numpy as np
import cv2
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import pandas as pd
from torchvision import transforms, datasets
from PIL import Image





def pre_processing(image):

	transform = transforms.Compose([
	                transforms.Resize((128, 128)),
	                transforms.ToTensor(),
	                transforms.Normalize(mean=[0.485, 0.456, 0.406],
	                     std=[0.229, 0.224, 0.225])
	            ])
	image = transform(image)
	image = image.unsqueeze(0)
	return image



def inference_result(model,image):
	results = model(image)
	return int(results.argmax())

def conv_block(in_channels, out_channels, pool=False):
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1), 
              nn.BatchNorm2d(out_channels),
              nn.ReLU(inplace=True)]
    if pool: layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)
    

# CNN with residual connections
class BigBirdResNet(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        
        self.conv1 = conv_block(in_channels, 64)
        self.conv2 = conv_block(64, 128, pool=True)
        self.res1 = nn.Sequential(conv_block(128, 128), conv_block(128, 128))
        
        self.conv3 = conv_block(128, 256, pool=True)
        self.conv4 = conv_block(256, 512, pool=True)
        self.res2 = nn.Sequential(conv_block(512, 512), conv_block(512, 512))
        
        self.classifier = nn.Sequential(nn.MaxPool2d(4),
                                        nn.Flatten(),
                                        nn.Dropout(0.2))
        self.final_layer = nn.Sequential(nn.Linear(8192, num_classes))
    def forward(self, xb):
        out = self.conv1(xb)
        #print(f"{out.shape}, 1")
        out = self.conv2(out)
        #print(f"{out.shape}, 2")
        out = self.res1(out) + out # add residual
        #print(f"{out.shape}, 3")
        out = self.conv3(out)
        #print(f"{out.shape}, 4")
        out = self.conv4(out)
        #print(f"{out.shape}, 5")
        out = self.res2(out) + out # add residual
        #print(f"{out.shape}, 6")
        out = self.classifier(out)
        #print(f"{out.shape}, 7")
        out = self.final_layer(out)
        #print(f"{out.shape}, 8")
        return out


def build_model():
	model = torch.load("bigbird_resnet_v1.pth")
	model = BigBirdResNet(3, 200) # 3 color channels and 9 output classes
	model.load_state_dict(torch.load("bigbird_resnet_v1.pth"))
	model.eval()
	return model

def lookup_class(results):
	results = results+1
	with open('classes.txt', 'r') as searchfile:
	    for line in searchfile:
	        if results == int(line.split(" ")[0]):
	            print(line.split(" ")[1])
	            return results

if __name__ == "__main__":
	model = build_model()
	image = Image.open("Horned_Grebe_test.jpg")
	image = pre_processing(image)
	inference = inference_result(model,image)
	result = lookup_class(inference)

