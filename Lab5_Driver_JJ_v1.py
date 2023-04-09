import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import cv2
import time
import minimalmodbus
import numpy as np

from robotiqGripper import*
from robolink import *
from robodk import *


# ME 5286: Robot Lab 5

## Initialize Robot
RDK = Robolink()

robot = RDK.ItemUserPick('Select a robot', ITEM_TYPE_ROBOT)
if not robot.Valid():
    raise Exception('No robot selected or available')
RUN_ON_ROBOT = False # Change this value to TRUE to run on robot in lab, change to FALSE for simulation

if RDK.RunMode() != RUNMODE_SIMULATE:
    RUN_ON_ROBOT = False
if RUN_ON_ROBOT:

    success = robot.Connect()
    status, status_msg = robot.ConnectedState()
    if status != ROBOTCOM_READY:
        print(status_msg)
        raise Exception("Failed to connect: " + status_msg)
    RDK.setRunMode(RUNMODE_RUN_ROBOT)

joints_ref = robot.Joints()
target_ref = robot.Pose()
pos_ref = target_ref.Pos()
robot.setPoseFrame(robot.PoseFrame())
robot.setPoseTool(robot.PoseTool())

    
## Initialize Gripper (Change parameters if needed, See Lab Manual for info)
instrument = minimalmodbus.Instrument('COM5', 9, debug = False)
instrument.serial.baudrate = 115200
gripper = RobotiqGripper(portname='COM5',slaveaddress=9)


## Common Gripper Commands
#gripper.activate() #Turns on the gripper
#gripper.goTo(position=255,speed=255,force=255) # Commands the gripper to go to a specified position, with a specific speed and force
#gripper.closeGripper(speed=255,force=255) #Commands the gripper to close with specified speed and force
#gripper.openGripper(speed=255,force=255) #Commands the gripper to open with specified speed and force


## Initialize Camera
device = 0 #Change to 1 or 2 for multiple cameras
vidcap = cv2.VideoCapture(device)
if not vidcap.isOpened():
    print(f"Cannot open camera {device}")
    exit()

# Turn off Autofocus
vidcap.set(cv2.CAP_PROP_AUTOFOCUS,0)

# Set camera resolution and foucs level
vidcap.set(cv2.CAP_PROP_FRAME_WIDTH,1280) 
vidcap.set(cv2.CAP_PROP_FRAME_HEIGHT,720)
vidcap.set(cv2.CAP_PROP_FOCUS,450)

## Common Camera & CV Functions
#ret, frame = vidcap.read() # Captures what the camera currently sees. Ret sees if a frame is retrieved, frame is the current image
#cv2.imshow(frame) # Shows the image stored in the tensor "frame"
#cv2.imwrite('testimage.jpg',frame) # saves image captured in "frame" and saves it to "testimage.jpg"
#image = cv2.imread('testimage.jpg') # reads image stored in file "testimage.jpg" and stores it in variable "image"
#vidcap.release() # Releases the camera device
#cv.destroyAllWindows() # gets rid of all opened CV2 Windows


##~~~~~~~~~~~~~~~~~~~~Write Your Code After Here~~~~~~~~~~~~~~~~~~~~##
# %%
"""
    Image Processing Class
    - imageProcessing():
        - __init__()
        - load()
        - preprocess()
        - augment()
"""   
class imageProcessing():
    def __init__(self, fileName):
        pass
    def load(self, fileName):
        rawImage = cv2.imread(fileName)
        rows, cols, channels = rawImage.shape
        rawSize = rawImage.shape
        self.image = rawImage
        return image, rows, cols, channels, rawSize
    
    def preprocess(self, image):
        # load image, probably cv2 stuff
        # resize image to 320, 180, 3
        # scale to -1 to 1
        return newImage
    
    def augment(self, image):
        # random brightness image
        # random flip left or right
        # random translate 
        return augmentImage
    

# %%
"""
    Data Cataloging Class
    - cataloger():
        - __init__()
        - classes()
        - trainTest()
        - dictionary()
"""    
class cataloger():
    def __init__(self):
        pass
    def classes(self, path):
        classes = []
        for class_dir in os.listdir(path):
            direc = os.path.join(path, class_dir)
            if not os.path.isdir(direc):
                continue
            classes.append(class_dir)
        classes.sort()
        return classes
    def trainTest(self, path):
        all_files = []
        for class_dir in classes:
            direc  = os.path.join(path, class_dir)
            for file in os.listdir(direc):
                if os.path.isfile(os.path.join(direc, file)):
                    full_file = os.path.join(class_dir, file)
                    all_files.append(full_file)
        n_samples = len(sample_list)
        indices = np.arange(n_samples, dtype = np.int32)
        np.random.shuffle(indices)
        
        n_train = int(n_samples * 0.8)
        train_ind = indices[:n_train]
        test_ind = indices[n_train:]
        
        Train = [sample_list[i] for i in train_ind]
        Test = [sample_list[i] for i in test_ind] 
        return Train, Test

    def dictionary(self):
        labels = {}
        for class_dir in classes:
            direc  = os.path.join(path, class_dir)
            for file in os.listdir(direc):
                if os.path.isfile(os.path.join(direc, file)):
                    labels[full_file] = classes.index(class_dir)
        return labels
        
# %%
"""
    Data Generation Class
    - generatorRex()
"""        
class generatorRex(keras.utils.Sequence):
    def __init__(self, samples, labels, path, n_classes, batch_size, dim):
        self.path = path
        self.samples = samples
        self.labels = labels
        self.n_classes = n_classes
        self.batch_size = batch_size
        self.dim = dim
        self.on_epoch_end()
        
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
    train_generator = DataGenerator(Train, Labels, './Dataset', 4, batchSize, (180,320,3))
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
    Architecture Layout
    - Architecture and training parameters to be used
        - Layers:
        - Training loss function: Categorical Cross-Entropy
        - Optimizer: Adam Optimizer with learning rate = 0.01
            
"""
def CNN(show = False):
    model = keras.models.Sequential([
        keras.layers.Dense(input_shape = (16,)),
        keras.layers.Dense(24, activation = 'relu'),
        keras.layers.Dense(12, activation = 'relu'),
        keras.layers.Dense(6, activation = 'relu'),
        keras.layers.Dense(4, activation = 'softmax'),
        ])
    
    model = tf.keras.models.Sequential([
      tf.keras.layers.Conv2D(64, (3,3), activation='relu',
                             input_shape=(28,28,1)),
      tf.keras.layers.MaxPooling2D(2,2),
      tf.keras.layers.Conv2D(64, (3,3)),
      tf.keras.layers.MaxPooling2D(2,2),
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(1024, activation='relu'),
      tf.keras.layers.Dense(26, activation='softmax')
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
    Problem #6: Test Inferencing
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

# %%
"""
    Main Function
"""  

Skrt = imageProcessing()
Skrt.load('./Dataset/Screwdriver/screwdriver_106.jpg')
spree = Skrt.preprocess()
spreeee = Skrt.augment(spree)
    
## Load Stored Model



## Begin Robot Program Here


