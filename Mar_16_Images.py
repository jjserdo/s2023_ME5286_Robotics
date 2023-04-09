# -*- coding: utf-8 -*-
"""
Created on Fri Mar 24 16:30:27 2023

@author: jjser
"""

import numpy as np
import cv2

# When Importing from another file, it runs the entire code
from JJ_CV import showImage, getBGR, imageSmol 

# %% # Import, Size, Blue, Green, Red

f = "turtle.jpg"
raw_image = cv2.imread(f)
rows, cols, channels = raw_image.shape
size = raw_image.shape

blue  = np.zeros(raw_image.shape[:2],np.uint8)
green = np.zeros(raw_image.shape[:2],np.uint8)
red   = np.zeros(raw_image.shape[:2],np.uint8)
for y in range(rows):
    for x in range(cols):
        blue[y,x], green[y,x], red[y,x] = getBGR(raw_image[y,x])

# %% 
"""
    Resizing Images, Lecture 2, Slide 20 - 21
"""
'''
smol = np.zeros((int(rows/2),int(cols/2),3),np.uint8)

for yy in range(int(rows/2)):
    for xx in range(int(cols/2)):
        smol[yy,xx,:] = raw_image[yy*2, xx*2, :]

#smol = imageSmol(raw_image,2)
#showImage(smol)

big = np.zeros((rows*2,cols*2,3),np.uint8)

for yy in range(rows*2):
    for xx in range(cols*2):
        big[yy:yy+2,xx:xx+2,:] = raw_image[int(yy/2), int(xx/2), :]
#showImage(big)
'''

# %% 
"""
    RGB to GreyScale, Lecture 2, Slide 51
"""

grey = np.zeros((rows,cols), np.uint8)
'''
for yy in range(rows):
    for xx in range(cols):
        b, g, r = getBGR(raw_image[yy,xx])
        grey[yy,xx] = b/3 + g/3 + r/3 # do not add b, g and r directly

showImage(grey)
'''

# %% 
"""
    RGB to YCbCr, Lecture 2, Slide 52
    - :(( unfinished
"""

# b,g,r = cv2.split(raw_image)
def baba(pixel):
    b,g,r = getBGR(pixel)
    Y = 16 + 65.738*r + 129.057*g + 25.064*b
    C_b = 128 - 37.945*r - 74.494*g + 112.439*b
    C_r = 128 + 112.439*r - 94.154*g - 18.285*b
    return np.array([Y, C_b, C_r])

'''
newtwo = np.zeros(size, np.uint8)
for yy in range(rows):
    for xx in range(cols):
        newtwo[yy,xx,:] =  baba(raw_image[yy,xx])
        
showImage(newtwo)
'''

# %% 
"""
    RGB to HSV, Lecture 2, Slide 53
"""

def wawa(pixel):
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

newtwo = np.zeros(size, np.uint8)
for yy in range(rows):
    for xx in range(cols):
        newtwo[yy,xx,:] =  wawa(raw_image[yy,xx])
        
showImage(newtwo)
