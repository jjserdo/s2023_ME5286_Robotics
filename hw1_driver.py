# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 09:15:48 2023

@author: jjser
Justine John A. Serdoncillo
ME 5286 Homework 1
Due April 4, 2023
"""

import numpy as np
import cv2
import random

# %% 
"""
    Personal Created Functions
    - loadImage()
    - getBGR()
    - greyScale()
    - showImage()
    - RGB_HSV()
    - intensifyImage()
    - padImage()
    - sobelXY()
    - threshImage()
"""

def loadImage(fileName):
    raw_image = cv2.imread(fileName)
    rows, cols, channels = raw_image.shape
    raw_size = raw_image.shape
    return raw_image, raw_size

def getBGR(pixel):
    blue  = pixel[0]
    green = pixel[1]
    red   = pixel[2]
    return np.array([blue, green, red])

def greyScale(imageName):
    rows, cols, channels = imageName.shape
    newImage = np.zeros((rows, cols), np.uint8)
    for yy in range(rows):
        for xx in range(cols):
            b, g, r = getBGR(imageName[yy,xx])
            newImage[yy,xx] = b/3 + g/3 + r/3
    size = newImage.shape
    return newImage, size

def showImage(imageName):
    if np.shape(imageName) == ():
        print("Empty Image")
        return 0
    else:
        while (True):
            cv2.imshow("Display", imageName)
            if cv2.waitKey(20) & 0xFF == 27: # ASCII character = 27 which is 'escape'
                break
        cv2.destroyWindow("Display")

def RGB_HSV(pixel):
    b,g,r = getBGR(pixel)
    v = max(r,g,b)
    delta = max(r,g,b) - min(r,g,b)
    if delta == 0:
        h = 0
        s = 0
    else:
        s = delta / v
        if r==v:
            h = (g/delta-b/delta) % 6
        elif g==v:
            h = 2 + b/delta - r/delta
        else:
            h = 4 + r/delta - g/delta
    h /= 6
    return np.array([h, s, v]) * 255

def intensifyImage(imageName,factor,clip=True):
    size = imageName.shape
    newImage = np.zeros(size,np.uint8)
    for y in range(size[0]):
        for x in range(size[1]):
            if factor > 1 and clip == True:
                newImage[y,x] = np.clip(getBGR(imageName[y,x]) * factor, 0, 255)
            else:
                newImage[y,x] = getBGR(imageName[y,x]) * factor   
    size = newImage.shape
    return newImage, size

def padImage(imageName,thick):
    size = imageName.shape
    if len(size) == 2:
        rows, cols = imageName.shape
        newImage = np.ones((rows+2*thick, cols+2*thick), np.uint8) * 127
        newImage[thick:thick+rows, thick:thick+cols] = imageName
        size = newImage.shape
        return newImage, size
    elif len(size) == 3:
        rows, cols, channels = imageName.shape
        newImage = np.ones((rows+2*thick, cols+2*thick, 3), np.uint8) * 127
        newImage[thick:thick+rows, thick:thick+cols, :] = imageName
        size = newImage.shape
        return newImage, size
    else:
        print("Bro this isn't valid")
        return 0
    

def sobelXY(imageName, xy):
    size = imageName.shape
    if len(size) == 2:
        if xy == 'x':
            filterer = np.array([[-1, 0, 1],[-2, 0, 2],[-1, 0, 1]], np.float32) 
        elif xy == 'y':
            filterer = np.array([[ 1, 2, 1],[ 0, 0, 0],[-1,-2,-1]], np.float32) 
        else:
            print("Bro what are you even filtering")
            return 0
        rows = size[0]
        cols = size[1]
        sobel = np.zeros((rows-2,cols-2), np.uint8)
        for yy in range(rows-3):
            for xx in range(cols-3):
                roi = imageName[yy:yy+3,xx:xx+3]
                sobel[yy,xx] = max(min(np.sum(filterer * roi),255),0)
                #sobel[yy,xx] = np.sum(filterer * roi)
        ssize = sobel.shape
        return sobel, ssize
    else:
        print("Bro this isn't grey scaled")
        return 0
    
def threshImage(imageName, thresh):
    newImage = np.zeros(imageName.shape, np.uint8)
    newImage[imageName >= thresh] = 255
    size = newImage.shape
    return newImage, size

# %% 
"""
    Problem Specific Functions
    - problem1A()
    - problem1B()
    - problem2A()
    - problem2B()
    - problem2C()
    - problem2D()
    - problem3A()
    - problem3B()
"""

def problem1A(fileName):
    imageName, size = loadImage(fileName)
    print(f"The image {fileName} has dimensions ({size[0]},{size[1]})")
    print(f"The image {fileName} has {size[2]} channels")
    showImage(imageName)
    grey, size = greyScale(imageName)
    mean = 0
    for yy in range(size[0]):
        for xx in range(size[1]):
            mean += grey[yy,xx]
    mean /= size[0]*size[1]
    print(f"The image has an average pixel intensity value of {mean}" )
    showImage(grey)
    
def problem1B(fileName,factor):
    imageName, size = loadImage(fileName)
    rows, cols, channels = imageName.shape
    print(f"The image {fileName} has dimensions ({size[0]},{size[1]})")
    showImage(imageName)
    newImage = np.zeros((int(rows/factor),int(cols/factor),3),np.uint8)
    for yy in range(int(rows/factor)):
        for xx in range(int(cols/factor)):
            newImage[yy,xx,:] = imageName[yy*factor, xx*factor, :]
    newRows, newCols, newChannels = newImage.shape
    print(f"The decimated image has dimensions ({newRows},{newCols})")
    showImage(newImage)
    newnewImage = np.zeros(size, np.uint8)
    for yy in range(newRows):
        for xx in range(newCols):
            newnewImage[yy*factor:yy*factor+factor,xx*factor:xx*factor+factor,:] = newImage[yy, xx, :]
    newnewRows, newnewCols, newnewChannels = newnewImage.shape
    print(f"The resized image has dimensions ({newnewRows},{newnewCols})")
    showImage(newnewImage)
    
def problem2A(fileName):
    imageName, size = loadImage(fileName)
    rows, cols, channels = imageName.shape
    print(f"The image {fileName} has dimensions ({size[0]},{size[1]})")
    showImage(imageName)
    blue  = imageName.copy()
    green = imageName.copy()
    red   = imageName.copy()
    blue[:,:,1] = 0
    blue[:,:,2] = 0
    green[:,:,0] = 0
    green[:,:,2] = 0
    red[:,:,0] = 0
    red[:,:,1] = 0
    showImage(blue)
    showImage(green)
    showImage(red)

def problem2B(fileName):
    imageName, size = loadImage(fileName)
    rows, cols, channels = imageName.shape
    print(f"The image {fileName} has dimensions ({size[0]},{size[1]})")
    showImage(imageName)
    newImage = np.zeros(size, np.uint8)
    for yy in range(rows):
        for xx in range(cols):
            newImage[yy,xx,:] =  RGB_HSV(imageName[yy,xx])
    showImage(newImage)
    
def problem2C(fileName):
    imageName, size = loadImage(fileName)
    print(f"The image {fileName} has dimensions ({size[0]},{size[1]})")
    showImage(imageName)
    grey, size = greyScale(imageName)
    showImage(grey)
        
def problem2D(fileName,number):
    imageName, size = loadImage(fileName)
    print(f"The image {fileName} has dimensions ({size[0]},{size[1]})")
    showImage(imageName)
    newImage, size = intensifyImage(imageName,0.7,clip=True)
    showImage(newImage)
    newImage, size = intensifyImage(imageName,1.3,clip=True)
    showImage(newImage)
    for i in range(number):
        factor = random.uniform(0.7,1.3)
        newImage, size = intensifyImage(imageName,factor,clip=True)
        showImage(newImage)     
    newImage, size = intensifyImage(imageName,1.3,clip=False)
    showImage(newImage)
    for i in range(number):
        factor = random.uniform(1.0,1.3)
        newImage, size = intensifyImage(imageName,factor,clip=False)
        showImage(newImage)
        
def problem3A(fileName,show):
    imageName, size = loadImage(fileName)
    print(f"The image {fileName} has dimensions ({size[0]},{size[1]})")
    grey, size = greyScale(imageName)
    padded, size = padImage(grey, 1)
    sobelx, sizex = sobelXY(padded, 'x')
    sobely, sizey = sobelXY(padded, 'y')
    if show:
        showImage(imageName)
        showImage(grey)
        showImage(sobelx)
        showImage(sobely)
    return sobelx, sobely

def problem3B(fileName):
    sobelx, sobely = problem3A(fileName, False)
    size = sobelx.shape
    sobelxy = np.zeros(size)
    sobelxy = np.sqrt(sobelx.astype(np.float32)**2 + sobely.astype(np.float32)**2)
    sobelxy = sobelxy.astype(np.uint8)
    #sobelxy = (np.abs(sobelx) + np.abs(sobely)).astype(np.uint8)
    #print(sobelxy.dtype)
    showImage(sobelxy)
    threshed, size = threshImage(sobelxy, 100)
    threshed = threshed.astype(np.uint8)
    showImage(threshed)
# %%
"""
    Main Function
"""

if __name__ == "__main__":
    problem1A('picture1.jpg')
    problem1B('picture1.jpg',5)
    problem2A('picture2.jpg')
    problem2B('picture2.jpg')
    problem2C('picture2.jpg')
    problem2D('picture2.jpg',5)
    problem3A('picture3.png', True)
    problem3B('picture3.png')
    