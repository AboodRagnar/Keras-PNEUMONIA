
from keras.layers import Activation,Dropout,Conv2D,Flatten,MaxPooling2D
from keras.models import Sequential
from  keras.layers.core import Dense
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import VGG16
from keras.models import load_model
from sklearn.utils import shuffle
import json
import h5py




class ConvloutionSolution:

    def __init__(self):
        self.Modle=Sequential()
        self.Train=[]
        self.Test=[]
        self.Predict=[]

   
    def PreparationVGG16(self,Outputlyares=2,Learning_raio=0.001,Loss_func='categorical_crossentropy',Step_Per_Epoch=20,ValidationSteps=15,Epochs=30):

        # reshape Vgg16 to fit problem's need
        for ly in VGG16().layers[:-1]:
            self.Modle.add(ly)


        for ly in self.Modle.layers:
            ly.trainable=False

        
        self.Modle.add(Dense(Outputlyares,activation='softmax'))
        self.Modle.compile(Adam(Learning_raio),loss=Loss_func,metrics=['accuracy'])
        self.Modle.fit_generator(self.Train,steps_per_epoch=Step_Per_Epoch,validation_data=self.Test,validation_steps=ValidationSteps,epochs=Epochs,verbose=2) # self.Modle.fit_generator(Train,steps_per_epoch=20,validation_data=Test,validation_steps=15,epochs=30,verbose=2)

        self.Modle.save('PNEUMONIAClassification.h5')

    def fix_layer0(self,filename, batch_input_shape, dtype):
     with h5py.File(filename, 'r+') as f:
        model_config = json.loads(f.attrs['model_config'].decode('utf-8'))
        layer0 = model_config['config']['layers'][0]['config']
        layer0['batch_input_shape'] = batch_input_shape
        layer0['dtype'] = dtype
        f.attrs['model_config'] = json.dumps(model_config).encode('utf-8')

    def loadImagesFromDirectory(self,pathTrain,pathTest,PathValdiation,TargetSize=(224,224),ClassesNames=['PNEUMONIA','NORMAL'],Batch_sizeTrian=5,Batch_sizeTest=3,Batch_sizeValdiation=10):

        self.Train=ImageDataGenerator().flow_from_directory(pathTrain,target_size=TargetSize,classes=ClassesNames,batch_size=Batch_sizeTrian) #'T3/COVID19_DataSet/chest_xray/chest_xray/train'
        self.Test=ImageDataGenerator().flow_from_directory(pathTest,target_size=TargetSize,classes=ClassesNames,batch_size=Batch_sizeTest) #'T3/COVID19_DataSet/chest_xray/chest_xray/test'
        self.Predict=ImageDataGenerator().flow_from_directory(PathValdiation,target_size=TargetSize,classes=ClassesNames,batch_size=Batch_sizeValdiation) #'T3/COVID19_DataSet/chest_xray/chest_xray/predict'


    def loadModl(self,path):
        
        self.fix_layer0(path,[None,224,224,3],'float32')
        self.Modle= load_model(path)

    def PreperationCNN2layers(self,saving_file_name,filters_num=64,window_size=(3,3),activation_fun='relu',image_input_shape=(224,224,3),Dense_layer_num=2):

        self.Modle.add(Conv2D(filters_num,window_size,activation=activation_fun, input_shape=image_input_shape))
        self.Modle.add(MaxPooling2D((2,2)))

        self.Modle.add(Conv2D(filters_num,window_size,activation='relu'))
        self.Modle.add(MaxPooling2D((2,2)))
        self.Modle.add(Flatten())
        self.Modle.add(Dense(Dense_layer_num,activation='softmax'))
        self.Modle.compile(Adam(0.001),loss='categorical_crossentropy',metrics=['accuracy'])
        # fit vs fit_generator is fit for small data which can be loaded into memory, the other for large one whice they cannot // same for predict cases
        self.Modle.fit_generator(self.Train,steps_per_epoch=20,validation_data=self.Test,validation_steps=15,epochs=30,verbose=2)
        self.Modle.save(saving_file_name)




def plots(ims, figsize=(12,6), rows=1, interp=False, titles=None):
    if type(ims[0]) is np.ndarray:
        ims = np.array(ims).astype(np.uint8)
        if (ims.shape[-1] != 3):
            ims = ims.transpose((0,2,3,1))
    f = plt.figure(figsize=figsize)
    cols = len(ims)//rows if len(ims) % 2 == 0 else len(ims)//rows + 1
    for i in range(len(ims)):
        sp = f.add_subplot(rows, cols, i+1)
        sp.axis('Off')
        if titles is not None:
            sp.set_title(titles[i], fontsize=16)
        plt.imshow(ims[i], interpolation=None if interp else 'none')





# Cnn=ConvloutionSolution()

# Cnn.loadImagesFromDirectory('T3/COVID19_DataSet/chest_xray/chest_xray/train','T3/COVID19_DataSet/chest_xray/chest_xray/test','T3/COVID19_DataSet/chest_xray/chest_xray/predict')
# # Cnn.PreperationCNN('PNEUMONIAClassificationConv.h5')

# # Cnn.loadImagesFromDirectory
# # print(Cnn.Modle.predict_generator(Cnn.Predict,10))
# Cnn.loadModl('PNEUMONIAClassification.h5')

# photo=ImageDataGenerator().flow_from_directory('T3/44',target_size=(224,224),class_mode='categorical',batch_size=1)
# print(Cnn.Modle.predict(photo))
# print(photo.class_indices)

# images,labels=next(photo)    


# plots(images,titles=labels)
# plt.show()






###############################
