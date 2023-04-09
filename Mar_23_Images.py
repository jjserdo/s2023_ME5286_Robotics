# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 08:16:59 2023

@author: jjser
"""

import cv2
import numpy as np

"""
    Using "skimage" taken from stackoverflow
"""
#from skimage import io
#raw_image = io.imread('https://picsum.photos/200')

# %%
""" 
    PERSONAL CREATED FUNCTIONS 
        - 'showImage(imageName)' shows images
        - 'getBGR(pixel)' gets the blue, green, red values
        - 'intensifyImage(imageName,factor,clip=True)' intensifies
"""

def showImage(imageName):
    if np.shape(imageName) == ():
        print("Empty Image")
        return 0
    else:
        cv2.imshow("display", imageName)
        input = cv2.waitKey(0)
        ascii_input = input % 256
        print(ascii_input)
        return 1
def getBGR(pixel):
    blue  = pixel[0]
    green = pixel[1]
    red   = pixel[2]
    return np.array([blue, green, red])
def intensifyImage(imageName,factor,clip=True):
    size = imageName.shape
    newImage = np.zeros(size,np.uint8)
    for y in range(size[0]):
        for x in range(size[1]):
            if factor > 1 and clip == True:
                newImage[y,x] = np.clip(getBGR(imageName[y,x]) * factor, 0, 255)
            else:
                newImage[y,x] = getBGR(imageName[y,x]) * factor            
    return newImage
def padImage(imageName,thick):
    rows, cols, cha = imageName.shape
    newImage = np.ones((rows+2*thick, cols+2*thick, channels) , np.uint8) * 128
    newImage[thick:thick+rows, thick:thick+cols, :] = imageName
    return newImage
padImage.__doc__ = 'Creates a padded image'
''' Unfinished '''
def create_filter(imageName,te):
    pass
            
# %% Image loading, resizing
imagefile = "turtle.jpg"
raw_image = cv2.imread(imagefile)

print(type(raw_image))
size = raw_image.shape
print(size)

newsize = ( int(size[1]/2), int(size[0]/2) )
small_image = cv2.resize(raw_image, newsize)
print(small_image.shape)
#cv2.imwrite("turtle_small.jpg",small_image)

(rows,cols,channels) = raw_image.shape

blue  = np.zeros(raw_image.shape[:2],np.uint8)
green = np.zeros(raw_image.shape[:2],np.uint8)
red   = np.zeros(raw_image.shape[:2],np.uint8)

for y in range(rows):
    for x in range(cols):
        pixel = raw_image[y,x]
        blue[y,x]  = pixel[0]
        green[y,x] = pixel[1]
        red[y,x]   = pixel[2]
''' Used for changing floats to ints'''
#blue  = (np.rint(blue)).astype(int)
#green = (np.rint(blue)).astype(int)
#red   = (np.rint(blue)).astype(int)

size = raw_image.shape
channels = size[2]
print(channels)

bitdepth = raw_image.dtype
print(bitdepth)

# %% 
"""
    Creating Images, Lecture 3, Slide 23
"""

dimensions = (512,768,3)
img = np.zeros(dimensions,np.uint8)

for y in range(dimensions[0]):
    for x in range(dimensions[1]): 
        r = y % 256
        g = x % 256
        b = (y + x) % 256
        img[y,x] = (b,g,r)
        
# %%
"""   
    Matrix Operations, Lecture 3, Slide 24
"""

array1 = np.array([[1,2,3,4],[5,6,7,8]], np.int32)
print(type(array1))
print(np.shape(array1))
print(array1)

flat = array1.flatten()

max1 = np.amax(array1, axis=0) # maximum along columns, list
max2 = np.amax(array1, axis=1) # maximum along rows, list
      

# %% 
""" 
    Changing Image Intensity, Lecture 3, Slide 26 - 28
"""

'''
img_scaled_down = np.zeros(size,np.uint8)
factor = 0.7
for y in range(size[0]):
    for x in range(size[1]):
        img_scaled_down[y,x] = getBGR(raw_image[y,x]) * factor

img_scaled_up = np.zeros(size,np.uint8)
factor = 1.2
for y in range(size[0]):
    for x in range(size[1]):
        img_scaled_up[y,x] = np.clip(getBGR(raw_image[y,x]) * factor, 0, 255)
'''
# %%
"""   
    Matrix Operations on Image, Lecture 3, Slide 29
"""

shift = raw_image + 5 # adds 10 to everything?
clipped = np.clip(shift, 0, 255) # clips to 255

#showImage(clipped)

# %%
""" 
    Operations on Image, Lecture 3, Slide 30
"""

max_0 = blue.max(0) # maximum along columns, list
max_1 = blue.max(1) # maximum along rows, list
#print(max_0) 
#print(max_1)

max_pos = np.argmax(blue,axis=None)
coords = np.unravel_index(max_pos, blue.shape) # goes through row per row
#print(max_pos)
#print(coords)
# %%
""" 
    Image Inspection, Lecture 3, Slide 31
"""
def inspectImage():
    def inspect(event,x,y,flags,param):
        if event == cv2.EVENT_LBUTTONDBLCLK:
            print("image[{}, {}] = {}".format(y, x, raw_image[y, x, :] ))
    
    cv2.namedWindow('image')
    cv2.setMouseCallback('image', inspect)
    
    while(True):
        cv2.imshow('image', raw_image)
        if cv2.waitKey(20) & 0xFF == 27: # ASCII character = 27 which is 'escape'
            break
    cv2.destroyAllWindows()

# %%
""" 
    Matrix Access for Images, Lecture 3, Slide 32 - 33
"""

'''
subx = 20
suby = 20
count = 0

for yy in range(0, rows-suby, 1):
    for xx in range(0, cols-subx, 1):
        roi = raw_image[yy:yy+suby, xx:xx+subx, :]
        count += 1

print('count is', str(count))
'''

# %%
""" 
    Image Insertion, Lecture 3, Slide 34
"""

image_copy = raw_image.copy()

inserty = int(rows/2)
insertx = int(cols/2)
insert = np.ones((inserty, insertx, 3), np.uint8) * 125

starty = int(rows/4)
startx = int(cols/4)

image_copy[starty:starty+inserty,startx:startx+insertx,:] = insert

#showImage(image_copy)

# %%
""" 
    Padding, Lecture 3, Slide 35
    1. Create new array with the new padded shape
    2. Insert original image
"""
'''
thicc = 11
padded = np.ones((rows+2*thicc, cols+2*thicc, channels) , np.uint8) * 128
padded[thicc:thicc+rows, thicc:thicc+cols, :] = raw_image
'''
#padded = padImage(raw_image, 11)

#showImage(padded)

# %%
""" 
    Convolution, Lecture 3, Slide 36 - 37 
"""

'''
roi = np.array([[100,50,34],[3,255,140],[200,98,15]], np.uint8)
filterer = np.array([[1,1,1],[1,1,1],[1,1,1]], np.float32) / 9

sobel1 = np.array([[1, 2, 1],[0, 0, 0],[-1,-2,-1]], np.float32) 
sobel2 = np.array([[1, 0,-1],[2, 0,-2],[ 1, 0,-1]], np.float32) 

multiply = filterer * roi
summed = np.sum(multiply)

fsize = sobel1.shape[0]
output = padImage(blue, fsize)
for yy in range(0, rows-fsize, 1):
    for xx in range(0, cols-fsize, 1):
        roi = blue[yy:yy+3, xx:xx+3, :]
        product = sobel1 * roi
        output[yy+fsize,xx+fsize] = int(np.sum(product))

showImage(output)
'''

# %%
""" 
    Image Scaling, Lecture 3, Slide 38
    - makes the image larger which means that it needs to interpolate missing pixels
"""

factor = 2

zoom = cv2.resize(raw_image,None,fx=factor,fy=factor,interpolation=cv2.INTER_CUBIC)

#showImage(zoom)

# %%
""" 
    Image Transformations, Lecture 3, Slide 39 - 40 
"""

theta = 30
center = (cols/2,rows/2)
T = cv2.getRotationMatrix2D(center,theta,1) # center, angle, scale
shifted = cv2.warpAffine(raw_image, T, (cols,rows))

showImage(shifted)


# %%
""" 
    Image Thresholding and Trackbars, Lecture 3, Slide 41 - 42
"""

#value, thresh = cv2.threshold(blue,127,255,cv2.THRESH_BINARY)

#showImage(thresh)

'''
def default(x):
    T = cv2.getTrackbarPos('T','thresh')
    if T == 0:
        showImage(raw_image)
        
cv2.namedWindow('thresh')
cv2.createTrackbar('T','thresh',127,255,default) # trackbar name, window name, def, max, callback is called everytime moved
T = 127
while(1):
    T = cv2.getTrackbarPos('T','thresh')
    _, thresh = cv2.threshold(red,T,255,cv2.THRESH_BINARY)
    cv2.imshow("thresh", thresh)
    if cv2.waitKey(20) & 0xFF == 27: # ASCII character = 27 which is 'escape'
        break
cv2.destroyAllWindows()
'''

# %%
""" 
    Geometric Transformations, Lecture 3, Slide 45 - 55
"""

xog = int(cols/2)
yog = int(rows/2)
test = padImage(raw_image, int(max(xog,yog)))
test[:,:,:] = 255
r2, c2, ch = test.shape
x2 = int(c2/2)
y2 = int(r2/2)
t = np.radians(90)
s = np.sin(t)
c = np.cos(t) 
a = 0
b = 0
shear = np.array([[1,a,0],[b,1,0],[0,0,1]])
rot = np.array([[c,-s,0],[s,c,0],[0,0,1]])

for yy in range(rows):
    for xx in range(cols):
        x = xx - xog
        y = yy - yog
        xy = np.array([x,y,1])
        xpyp = (np.rint(shear @ rot @ xy.T)).astype(int)
        xp = xpyp[0]
        yp = xpyp[1]
        xxp = xp + x2
        yyp = yp + y2
        test[yyp,xxp,:] = raw_image[yy,xx,:]
#showImage(test)















