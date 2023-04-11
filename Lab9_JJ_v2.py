# -*- coding: utf-8 -*-
"""
Created on Thu Apr  6 14:40:51 2023

@author: jjser
Justine John A. Serdoncillo
Due April 27, 2023
ME 5286 Lab 5: Putting it all Together v2
    - image processing doesn't store the images
        - want to add check if factor is less than or greater than
            - so far this is only making it smaller
        - not sure if I can use cv2.flip()
    - data cataloging does not store the classes
    - image resizing and scaling works now
    - need getBGR() for intensifyImage lmao
    - changed np.load to cv2.imread because we are not dealing with .npy files anymore
    - started with creating a new X based on resize
    - I DONT WANT TO AUGMENT
    - Added testing and stuff
    - modified model, 180 to 160
    - fixed letsSeeHowGoodYouAre by changing to cv.imread
"""

import numpy as np
import cv2
import random
import tensorflow as tf
import tensorflow.keras as keras

import os
import warnings

# %% Personal Created Functions
"""
    Personal Created Functions
    - showImage()
    - getBGR()
    - intensifyImage()
    - flipper()
    - translate()
"""

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
        
def getBGR(pixel):
    blue  = pixel[0]
    green = pixel[1]
    red   = pixel[2]
    return np.array([blue, green, red])

def intensifyImage(imageName, factor, clip=True):
    size = imageName.shape
    newImage = np.zeros(size,np.uint8)
    for y in range(size[0]):
        for x in range(size[1]):
            if factor > 1 and clip == True:
                newImage[y,x] = np.clip(getBGR(imageName[y,x]) * factor, 0, 255)
            else:
                newImage[y,x] = getBGR(imageName[y,x]) * factor   
    return newImage

def flipper(imageName):
    newImage = cv2.flip(imageName, 1)
    return newImage

def translate(imageName, xTrans, yTrans):
    rows, cols, channels = np.array(imageName).shape
    #newImage = np.ones(imageName.shape) * 127 # issue is from here
    newImage = imageName.copy()
    for yy in range(rows):
        for xx in range(cols):
            x = xx + xTrans
            y = yy + xTrans
            if (0 <= x <= cols-1) and (0 <= y <= rows-1):
                newImage[y, x, :] = imageName[yy,xx,:]
    return newImage

# %% Image Processing [SUCCESS]
"""
    Image Processing Class
    - imageProcessing():
        - __init__()
        - preprocess()
        - augment()
"""   
class imageProcessing():
    def __init__(self, resize = False, scale = False, brightScale = None, Flip = None, transScale = None):
        self.resize = resize
        self.scale = scale
        self.brightScale = brightScale
        self.Flip = Flip
        self.transScale = transScale

    def preprocess(self, imageName):
        flag = False
        if self.resize is not False:
            newImage = np.zeros(self.resize, np.uint8)
            rows, cols, channels = imageName.shape
            factorx = int(cols/self.resize[1])
            factory = int(rows/self.resize[0])
            for yy in range(self.resize[0]):
                for xx in range(self.resize[1]):
                    newImage[yy,xx,:] = imageName[yy*factory, xx*factorx, :]
            flag = True
        
        if self.scale is not False:
            if self.resize is False:
                newImage = imageName
            newImage = newImage.astype(np.float64)
            newImage = (self.scale[1] - self.scale[0]) * newImage/255 + self.scale[0]
            flag = True
        
        if not flag:
            newImage = imageName
            print("Stop bro you buggin buggin")
            
        return newImage
    
    def augment(self, imageName):
        flag = False
        newImage = imageName
        if self.brightScale != None:
            factor = random.uniform(*self.brightScale)
            newImage = intensifyImage(newImage, factor)
            flag = True

        if self.Flip is not None:
            factor = random.uniform(0, 1)
            if factor < self.Flip:
                newImage = cv2.flip(newImage, 1)
            flag = True
            
        if self.transScale != None:
            f1 = random.randint(*self.transScale)
            f2 = random.randint(*self.transScale)
            newImage = translate(newImage, f1, f2)
            flag = True

        if not flag:
            print("bro you didn't do anything sigh")

        return newImage

def imageTest():
    """
        Verify the functionality of imageProcessing class
    """
    fileName = './Dataset/Screwdriver/screwdriver_106.jpg'
    rawImage = cv2.imread(fileName)
    #showImage(rawImage)
    
    #iForgotLMAO = imageProcessing(resize = (180,320,3), scale = (-1,1))    
    iForgotLMAO = imageProcessing(resize = (180,320,3))
    #iForgotLMAO = imageProcessing(scale = (-1,1)) 
    tooCold = iForgotLMAO.preprocess(rawImage)
    #letsgooo = imageProcessing(resize = (180,320,3), scale = (-1,1) ,brightScale=(0.75, 1.25), Flip=0.5, transScale=(-25, 25))
    #print(tooCold.shape)
    #print(tooCold.dtype)
    print(np.amax(tooCold))
    print(np.amin(tooCold))
    showImage(tooCold)
    
    '''
    diamond = imageProcessing(brightScale=(0.75, 1.25))
    doAflip = imageProcessing(Flip = 1) 
    decepticon = imageProcessing(transScale=(-25, 25))
    letsgooo = imageProcessing(brightScale=(0.75, 1.25), Flip=0.5, transScale=(-25, 25))
    
    oneone = diamond.preprocess(rawImage)
    print("briefing complete")
    
    onetwo = diamond.augment(oneone)
    showImage(onetwo)
    print("diamond one baby")
    onetwo = doAflip.augment(oneone)
    showImage(onetwo)
    print("flip was flipped")
    onetwo = decepticon.augment(oneone)
    showImage(onetwo)
    print("autobots rollout")
    onetwo = letsgooo.augment(oneone)
    showImage(onetwo)
    print("LETSGOOOOOOOOOOOOOOOOO")
    '''
    
    print("\n ---- Image Processing Succesful ---- \n")
    
    return True
    
# %% Data Cataloging [CHECKED]
"""
    Data Cataloging Class
    - cataloger():
        - __init__()
        - classes()
        - trainTest()
        - dictionary()
"""    
class cataloger():
    def __init__(self, path = None):
        if path is None:
            print("dude u lost")    
        else:
            self.path = path
        
    def classes(self, show = True):
        self.classes = []
        for class_dir in os.listdir(self.path):
            direc = os.path.join(self.path, class_dir)
            if not os.path.isdir(direc):
                continue
            self.classes.append(class_dir)
        self.classes.sort()
        self.numClass = len(self.classes)
        
        if show:
            print(f"Classes present: {self.classes} \n")
            print(f"You got this much {self.numClass}")
            
        return self.classes
    
    def trainTest(self, split, show = True):
        """
            Parameters
            ----------
            split : float
                - percentage in decimal of the training set
        """
        self.all_files = []
        for class_dir in self.classes:
            direc  = os.path.join(self.path, class_dir)
            for file in os.listdir(direc):
                if os.path.isfile(os.path.join(direc, file)):
                    full_file = os.path.join(class_dir, file)
                    self.all_files.append(full_file)
        n_samples = len(self.all_files)
        indices = np.arange(n_samples, dtype = np.int32)
        np.random.shuffle(indices)
        
        n_train = int(n_samples * split)
        train_ind = indices[:n_train]
        test_ind = indices[n_train:]
        
        self.train = [self.all_files[i] for i in train_ind]
        self.test = [self.all_files[i] for i in test_ind] 
        
        if show:
            print(f"I train every day bro \n {self.train[:10]} \n")
            print(f"There are {len(self.train)} trainers with me")
            print(f"Test me I dare you \n {self.test[:10]} \n")
            print(f"I do {len(self.test)} tests every year")
        
        return self.train, self.test

    def dictionary(self, show = True):
        self.labels = {}
        for class_dir in self.classes:
            direc  = os.path.join(self.path, class_dir)
            for file in os.listdir(direc):
                if os.path.isfile(os.path.join(direc, file)):
                    full_file = os.path.join(class_dir, file)
                    self.labels[full_file] = self.classes.index(class_dir)
                    
        if show:
            print(f"The map is flat \n {self.labels} \n")
            
        return self.labels
    
def catalogTest(location, split = 0.5):
    """
        Verify the functionality of cataloger class
    """
    walmart = cataloger(location)
    walmart.classes(show = True)
    print("gucci gang")
    walmart.trainTest(split, show = False)
    print("gucci gucci gang")
    walmart.dictionary(show = False)
    print("gucci gucci gang gang")
    
    print("\n ---- Data Cataloging Succesful ---- \n")
    
    return True
        
# %% Data Generation
"""
    Data Generation Class
    - generatorRex()
"""        
class generatorRex(keras.utils.Sequence):
    def __init__(self, samples, labels, path, classes, batch_size, pre):
        """
            Parameters
            ----------
            samples : list of strings
                - names of the samples
        """
        self.samples = samples
        self.labels = labels
        self.path = path
        self.classes = classes
        self.batch_size = batch_size
        self.pre = pre
        
        self.on_epoch_end()
        
        self.n_classes = len(classes)
        self.dim = pre.resize
        
    def __len__(self):
        return int(len(self.samples) / self.batch_size)
    
    def on_epoch_end(self):
        self.indexes = np.arange(len(self.samples))
        np.random.shuffle(self.indexes)
        
    def __getitem__(self, index):
        indexes = self.indexes[(index * self.batch_size):((index+1) * self.batch_size)]
        
        samples_temp = [self.samples[k] for k in indexes]
        
        X, Y = self.__create_next_batch(samples_temp)
        
        return X, Y
    
    def __create_next_batch(self, sample_ids_temp):
        X = np.empty([self.batch_size, *self.dim])
        Y = np.empty([self.batch_size], dtype = np.uint32)
        
        for i, file in enumerate(sample_ids_temp):
            bro = cv2.imread(os.path.join(self.path, file))
            Y[i] = self.labels[file]
            
            # Preprocessing
            X[i] = self.pre.preprocess(bro)
            
            #X[i] = self.pre.augment(gimme)
            
            '''
            if i < 10:
                print(f"I am from {file}")
            '''
        
            
        return X, self.to_one_hot(Y)
    
    def to_one_hot(self, Y):
        one_hottie = np.zeros((len(Y),self.n_classes))
        for i in range(len(Y)):
            index = self.classes.index(self.classes[Y[i]])
            one_hottie[i,index] = 1
        return one_hottie

def generatorTest(location, split, batchSize, show = True):
    """
        Verify the functionality of generator class
    """

    weee = cataloger(location)
    sheesh = weee.classes(show = False)
    train, test = weee.trainTest(0.8, show = False) # should use split but do 0.1 for easyness
    labels = weee.dictionary(show = False)
    pre = imageProcessing(resize = (160,320,3), scale = (0,1)) # should be 180 but don't want, 0 to 1 is better
    generator = generatorRex(train, labels, location, sheesh, batchSize, pre)
    x, y = generator.__getitem__(0)
    if show:
        np.set_printoptions(precision=4, suppress=True)
        for i in range(10):
            showImage(x[i])
        print(f"y from train_generator: \n {y[:10]} \n")
    
        print("\n ---- Data Generation Succesful ---- \n")
    
    return True

# %% Architecture Layout
"""
    Architecture Layout
    - Architecture and training parameters to be used
        - Layers: 12 layers of mixture of convolution, max pooling, flatten, dropout, etc.
        - Training loss function: Categorical Cross-Entropy
        - Optimizer: Adam Optimizer with learning rate = 0.01
            
"""
def CNN(show = True):    
    model = keras.models.Sequential([
        keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu', input_shape=(160,320,3)),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Conv2D(128, kernel_size=(3, 3), activation='relu'),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Conv2D(256, kernel_size=(3, 3), activation='relu'),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Conv2D(512, kernel_size=(3, 3), activation='relu'),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Flatten(),
        keras.layers.Dense(1024, activation='relu'),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(512, activation='relu'),
        keras.layers.Dense(4, activation='softmax')
        ])
    
    loss_fn = keras.losses.CategoricalCrossentropy()
    opt_fn = keras.optimizers.Adam(learning_rate = 0.001)
    model.compile(optimizer = opt_fn, loss = loss_fn, metrics = ['accuracy'])
    if show:
        model.summary()
        print("\n ---- Model Created ---- \n")
    
    return model

# %% Training the Model
"""
    Training the Model
        - Use data generators created
            - training data
            - test data
        - Train model created using the hyperparameters
            - Batch Size: 64
            - Number of Epochs: 100        
"""
def letsTrain(location, split, batchSize, epochs, nameWeights, save=False):
    weee = cataloger(location)
    sheesh = weee.classes(show = False)
    train, test = weee.trainTest(split, show = False) 
    labels = weee.dictionary(show = False)
    pre = imageProcessing(resize = (160,320,3), scale = (0,1))
    train_gen = generatorRex(train, labels, location, sheesh, batchSize, pre)
    test_gen = generatorRex(test, labels, location, sheesh, batchSize, pre)
    model = CNN(show = False)
    model.fit(train_gen, validation_data = test_gen, epochs = epochs)
    if save == True:
        model.save(nameWeights)
        print(f"Model weights saved as {nameWeights}")
        
    print("\n ---- problem5 Done ---- \n")
    return model, pre

# %% Test Interfacting
"""
    Test Inferencing
        - Use saved trained model
        - use on 'Wrenches/wrench_795.jpg'
"""

def letsSeeHowGoodYouAre(pre, path, testPath, weightsName, classes, show = False):
    model = keras.models.load_model(weightsName)
    bro = cv2.imread(os.path.join(path, testPath))
    showImage(bro)
    x_test = pre.preprocess(bro)
    showImage(x_test)
    x_test = x_test[np.newaxis, :]
    inference = model(x_test).numpy()
    wee = np.argmax(inference)
    if show:
        np.set_printoptions(0, suppress = True)
        print(f"Output inference: {inference} ")
        print(f"Index of Maximum Value: {wee} ")
        print(f"This corresponds to '{classes[wee]}' ")
    print("\n ---- problem6 Done ---- \n")
    return True


# %% Where everything goes together
"""
    Main Function
"""  
if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    os.environ['QT_FATAL_WARNINGS'] = '0'
    
    """ Here is all I know """
    location = './Dataset'
    split = 0.8
    batchSize = 64
    epochs = 100
    nameWeights = 'weights.h5'
    testInference = 'Wrench\wrench_795.jpg'
    
    
    """ Here are my functions to test stuff out """
    #imageTest()
    #catalogTest(location, split)
    #generatorTest(location, split, batchSize)
    #model = CNN()
    #weights, pre = letsTrain(location, split, batchSize, epochs, nameWeights, save = False)
    pre = imageProcessing(resize = (160,320,3), scale = (0,1))
    classes = ['Hammer', 'Pliers', 'Screwdriver', 'Wrench']
    letsSeeHowGoodYouAre(pre, location, testInference, nameWeights, classes, show = True)


