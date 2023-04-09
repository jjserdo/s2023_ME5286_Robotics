# -*- coding: utf-8 -*-
"""
Created on Tue Apr 6 2023 [1125]

@author: jjser
Justine John A. Serdoncillo
Due April 11, 2023
ME 5286 Homework 2 v4
    - Forgot about number 6
    - removed testModel() which works on test_gen
    - created problem6()
    - edited problem5()
        - Accuracy
        - Validation Accuracy
        - Loss
    - created problem6Hard() which checks everything from a folder
    - fixed issue by adding preprocessing()
    - edited input vector based on assignment
"""

import numpy as np
import random
import tensorflow as tf
import tensorflow.keras as keras
import os
import warnings

# %% 
"""
    Problem #1: Data Scaling (10 Points)
        - scaled using a prescribed minimum and range values for each dimension
"""
def preprocessing(enterHere):
    minimum_p1 = np.array([25.06, 25.06, 24.5, 25.19, 0.0, 0.0, 0.0, 0.0, 0.06, 0.04, 0.05, 0.06, 345.0, -2.083, 0.0, 0.0])
    range_p1 = np.array([1.13, 1.75, 1.38, 1.12, 150.0, 237.0, 181.0, 56.0, 0.66, 0.47, 0.5, 0.17, 610.0, 4.229, 1.0, 1.0])
    
    scaled = 2.0 * ((enterHere - minimum_p1)/range_p1) - 1.0
    
    return scaled

def problem1(input_array, show = False):
    scaled = preprocessing(input_array)
    if show:
        np.set_printoptions(precision=3, suppress=True)
        print(f"Input array: {input_array} \n")
        np.set_printoptions(precision=4, suppress=True)
        print(f"Scaled array: {scaled} \n")
    print("\n ---- problem1 Done ---- \n")
    return scaled 
# %% 
"""
    Problem #2: Dataset Cataloging (20 Points)
        - python class for dataset cataloging
"""

def catalog_dataset(path):
    classes = []
    for class_dir in os.listdir(path):
        direc = os.path.join(path, class_dir)
        if not os.path.isdir(direc):
            continue
        classes.append(class_dir)
    classes.sort()
    all_files = []
    labels = {}
    for class_dir in classes:
        direc  = os.path.join(path, class_dir)
        for file in os.listdir(direc):
            if os.path.isfile(os.path.join(direc, file)):
                full_file = os.path.join(class_dir, file)
                all_files.append(full_file)
                labels[full_file] = classes.index(class_dir)
    return classes, all_files, labels

def problem2(location, show = False):
    IgotClass, sample_list, thisIsMe = catalog_dataset(location)
    
    n_samples = len(sample_list)
    indices = np.arange(n_samples, dtype = np.int32)
    np.random.shuffle(indices)
    
    n_train = int(n_samples * 0.8)
    train_ind = indices[:n_train]
    test_ind = indices[n_train:]
    
    Train = [sample_list[i] for i in train_ind]
    Test = [sample_list[i] for i in test_ind] 

    if show:
        print(f"Classes present: {IgotClass} \n")
        print(f"Train array: {Train[:10]} \n")
        print(f"Test array: {Test[:10]} \n")
        print(f"Dictionary Mapping: {thisIsMe} \n")
    
    print("\n ---- problem2 Done ---- \n")
    
    return Train, Test, thisIsMe, IgotClass

            
# %% 
"""
    Problem #3: Data Generator (20 Points)
        - custom data generator class
        - loading random batch of numpy files for training each epoch
"""
class DataGenerator(keras.utils.Sequence):
    def __init__(self, samples, labels, path, n_classes, classes, batch_size, dim):
        self.path = path
        self.samples = samples
        self.labels = labels
        self.n_classes = n_classes
        self.batch_size = batch_size
        self.dim = dim
        self.on_epoch_end()
        self.classes = classes
        
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
        X = np.empty([self.batch_size, self.dim])
        Y = np.empty([self.batch_size], dtype = np.uint32)
        
        for i, file in enumerate(sample_ids_temp):
            X[i,:] = np.load(os.path.join(self.path, file))
            Y[i] = self.labels[file]
            
            # Preprocessing
            X[i,:] = preprocessing(X[i,:])
            
        return X, self.to_one_hot(Y)
    
    def to_one_hot(self, Y):
        one_hottie = np.zeros((len(Y),self.n_classes))
        for i in range(len(Y)):
            index = self.classes.index(self.classes[Y[i]])
            one_hottie[i,index] = 1
        return one_hottie
        

def problem3(Train, Test, Labels, classes, batchSize, show = False):
    train_generator = DataGenerator(Train, Labels, './Dataset_HW2', 4, classes, batchSize, 16)
    
    x, y = train_generator.__getitem__(0)
    if show:
        np.set_printoptions(precision=4, suppress=True)
        print(f"x from train_generator: \n {x[:10]} \n")
        print(f"y from train_generator: \n {y[:10]} \n")
    test_generator = DataGenerator(Test, Labels, './Dataset_HW2', 4, classes, batchSize, 16)
    
    print("\n ---- problem3 Done ---- \n")
    
    return train_generator, test_generator

# %% 
"""
    Problem #4: Architecture Layout (10 Points)
        - Architecture and training parameters to be used
            - Layers: 16 -> 24 -> 12 -> 6 -> 4 softmax
            - Training loss function: Categorical Cross-Entropy
            - Optimizer: Adam Optimizer with learning rate = 0.01
            
"""
def problem4(show = False):
    model = keras.models.Sequential([
        keras.layers.Flatten(input_shape = (16,)),
        keras.layers.Dense(24, activation = 'relu'),
        keras.layers.Dense(12, activation = 'relu'),
        keras.layers.Dense(6, activation = 'relu'),
        keras.layers.Dense(4, activation = 'softmax'),
        ])
    
    loss_fn = keras.losses.CategoricalCrossentropy()
    opt_fn = keras.optimizers.Adam(learning_rate = 0.001)
    model.compile(optimizer = opt_fn, loss = loss_fn, metrics = ['accuracy'])
    if show:
        model.summary()
    print("\n ---- problem4 Done ---- \n")
    
    return model

# %% 
"""
    Problem #5: Training the Model (10 Points)
        - Use data generators created
            - training data
            - test data
        - Train model created using the hyperparameters
            - Batch Size: 64
            - Number of Epochs: 100        
"""
def problem5(train_gen, test_gen, model, batchsize, epochs, name, save=False):
    model.fit(train_gen, validation_data = test_gen, epochs = epochs)
    
    if save == True:
        model.save('weights.h5')
        print("Model weights saved as weights.h5")
        
    print("\n ---- problem5 Done ---- \n")
    return model

# %% 
"""
    Problem #6: Test Inferencing (10 Points)
        - Use saved trained model
        - use on 'class_2/sample3082.npy'
"""

def problem6(path, testPath, weightsName, classes, show = False):
    model = keras.models.load_model(weightsName)
    x_test = np.load(os.path.join(path,testPath))
    x_test = preprocessing(x_test) # PreProcessing first
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

def problem6Hard(path, testPath, weightsName, classes, show = False):
    model = keras.models.load_model(weightsName)
    test_dir = os.path.join(path, testPath)
    files = os.listdir(test_dir)  # List all files in the testPath directory

    for file in files:
        file_path = os.path.join(test_dir, file)
        x_test = np.load(file_path)  # Load the file using its path
        x_test = preprocessing(x_test) # PreProcessing first
        x_test = x_test[np.newaxis, :]
        inference = model(x_test).numpy()
        wee = np.argmax(inference)
        if show:
            np.set_printoptions(0, suppress = True)
            #print(f"Output inference: {inference} ")
            #print(f"Index of Maximum Value: {wee} ")
            print(f"This corresponds to '{classes[wee]}' ")
    print("\n ---- problem6 All in Done ---- \n")
    return True

# %%
"""
    Main Function
"""

if __name__ == "__main__":
    warnings.filterwarnings("ignore", message=r"Passing", category=FutureWarning)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    input_p1 = np.array([25.94, 25.75, 24.56, 25.38, 121, 34, 53, 40, 0.08, 0.19, 0.06, 0.06, 390, 0.769, 0, 0]) 
    location = './Dataset_HW2'
    batchSize = 64
    epochs = 100
    nameWeights = 'weights.h5'
    testInference = 'class_2/sample3082.npy'
    
    problem1(input_p1, show = True)

    Train, Test, Labels, classes = problem2(location, show = True)
    
    train_gen, test_gen = problem3(Train, Test, Labels, classes, batchSize, show = True)
    
    model = problem4(show = True)
    
    weights = problem5(train_gen, test_gen, model, batchSize, epochs, nameWeights, save=True)
    classes = ['class_0', 'class_1', 'class_2', 'class_3']
    
    problem6(location, testInference, nameWeights, classes, show = True)
    problem6Hard(location, 'class_0', nameWeights, classes, show = True)
    problem6Hard(location, 'class_1', nameWeights, classes, show = True)
    problem6Hard(location, 'class_2', nameWeights, classes, show = True)
    problem6Hard(location, 'class_3', nameWeights, classes, show = True)
    
    
