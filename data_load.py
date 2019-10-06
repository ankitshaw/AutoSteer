import cv2
import os
import numpy as np
import scipy
import pickle
import matplotlib.pyplot as plt

def dispImg(image):
    return plt.imshow(image)

def greyscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  
def equalize(image):
    return cv2.equalizeHist(image)

def canny(image, l_threshold, h_threshold):
    return cv2.Canny(image,l_threshold, h_threshold)

def blur(image, kernel_size):
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
  
def resize(image, shape):
    image = cv2.resize(image, shape)
    return image

def scaled_angle(angle):
    return float(angle) * scipy.pi / 180
  
def process_image(image):
    image = greyscale(image)
    image = equalize(image)
    image = resize(image, (100,100))
    return image
    
    
#Download dataset from https://github.com/SullyChen/driving-datasets   
image = []
angle = []
c = 0
with open("data.txt") as f:
    for line in f:
        img_name, steering_angle = line.strip().split(",")[0].split()
        img = cv2.imread('data/'+img_name)
        img = process_image(img)
        img = img/255.0
        steering_angle = scaled_angle(steering_angle)             
        image.append(img)
        angle.append(steering_angle)
        
        c = c + 1
        if(c%500.0== 0.0):
            print(c,":",img_name,",",steering_angle)
            
            
with open("image", "wb") as f:
    pickle.dump(np.array(image), f, protocol=4)
    
with open("angle", "wb") as f:
    pickle.dump(np.array(angle), f, protocol=4)    
                       
