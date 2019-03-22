# -*- coding: utf-8 -*-
"""
Created on Sun Mar 17 21:09:39 2019

@author: Kevin
"""

import os
import numpy as np
import cv2
from imgaug import augmenters as iaa
class Loader():
    
    def __init__(self,
                 path_to_folder_w_classes="E:\\dataset_resized\\dataset_resized\\",
                 seed=3121991,
                 batch_size=32,
                 img_size=(256,256),
                 augmentation=iaa.Sequential([
                     iaa.Crop(px=(0, 16)), # crop images from each side by 0 to 16px (randomly chosen)
                     iaa.Fliplr(0.5), # horizontally flip 50% of the images
                     iaa.GaussianBlur(sigma=(0, 3.0)) # blur images with a sigma of 0 to 3.0
                 ])):
        """
            @brief A simple loading class which forces even distribution
            @param path_to_folder_w_classes Folder containing
            @param seed Seed used for random init
            @param batch_size Batch size used for generators
            @param img_size Image size used (only width and height)
            @param augmentation iaa augmentations
        """
        np.random.seed(seed)
        self.path = path_to_folder_w_classes
        self.class_names = os.listdir(path_to_folder_w_classes)
    
        self.train_amounts = np.zeros(shape=len(self.class_names))
        self.val_amounts = np.zeros(shape=len(self.class_names))
        
        self.train_files = np.empty(shape=len(self.class_names,),dtype=np.object)
        self.val_files = np.empty(shape=len(self.class_names,),dtype=np.object)
        for i,classes in enumerate(self.class_names):
            n  = os.listdir(self.path + classes)
            self.train_val_split = np.random.randint(0,10,size=len(n))
            self.train_files[i] = np.array(n)[self.train_val_split <= 8]
            self.val_files[i] = np.array(n)[self.train_val_split > 8]
            
            self.train_amounts[i] = self.train_files[i].shape[0]
 
        temp = []
        temp_y = []        
        for label,element in enumerate(self.val_files):
            temp.extend(element)
            print(element)
            temp_y.extend([label]*len(element))
        self.val_files = np.array(temp)
        self.val_y = np.array(temp_y)
        
        
        self.fill_ups = self.train_amounts.max() - self.train_amounts
        self.train_files_filled = np.empty(shape=(len(self.class_names),int(self.train_amounts.max())),dtype=np.object)
        self.val_files_filled = np.empty(shape=len(self.class_names,),dtype=np.object)      
        
        self.y_all = np.zeros(shape=(int(len(self.class_names)*self.train_amounts.max())))
        for label,fill_ups in enumerate(self.fill_ups):
            self.y_all[int(label*self.train_amounts.max()):int((label+1)*self.train_amounts.max())] = label
          
            if fill_ups == 0:
                self.train_files_filled[label] = self.train_files[label]
                continue
            self.fillers = self.__uniqueRandomNumbers(fill_ups,self.train_amounts[label])
            
            
            self.train_files_filled[label] = \
            np.concatenate([self.train_files[label],self.train_files[label][self.fillers]])
            #+ self.train_files[label][self.__uniqueRandomNumbers(fill_ups,self.train_amounts[label])])
         
        
        idx = np.arange(len(self.y_all))
        np.random.shuffle(idx)
        self.y_all = self.y_all[idx]
        self.train_files_filled = self.train_files_filled.flatten()[idx]
            
        self.batch_size = batch_size
        self.train_cnt = 0
        self.val_cnt = 0
        self.img_size = img_size
        
        self.seq = augmentation
            
    def __uniqueRandomNumbers(self,size,maximum,unique=False):
        """
            @brief Func for getting unique random numbers withing a maxium range
            @brief size Amount of random numbers needed
            @brief maximum maximum number allowed
            @return Will return an umpy array
        """
#        
        if size >= maximum:
            
            if unique:
                raise ValueError("Size cannot  be larger than maxium!")
            else:
                mul = size / maximum
                idx = np.repeat(np.arange(0,maximum),int(mul+1))
                np.random.shuffle(idx)
                return np.array(idx[:int(size)],dtype=np.int32)
        idx = np.arange(0,maximum)
        np.random.shuffle(idx)
        return np.array(idx[:int(size)],dtype=np.int32)
        
    def trainGen(self):
        """
        
        """
        
        while True:
            X = np.zeros(shape=(self.batch_size,self.img_size[1],self.img_size[0],3))
            Y = np.zeros(shape=(self.batch_size,self.train_amounts.shape[0]))
        
            for i in range(self.batch_size):
                X[i] = cv2.resize(cv2.imread(self.path + "\\" + self.class_names[int(self.y_all[self.train_cnt])] + "\\" +self.train_files_filled[self.train_cnt]),self.img_size) / 255
                
                # for one hot encoding
                Y[i][int(self.y_all[self.train_cnt])] = 1.
        
                self.train_cnt += 1
                if self.train_cnt == len(self.train_files_filled):
                    self.train_cnt = 0
            yield X,Y
            
    def valGen(self):
        """
        
        """
        
        while True:
            X = np.zeros(shape=(self.batch_size,self.img_size[1],self.img_size[0],3))
            Y = np.zeros(shape=(self.batch_size,self.train_amounts.shape[0]))
        
            for i in range(self.batch_size):
                X[i] = cv2.resize(cv2.imread(self.path + "\\" + self.class_names[int(self.val_y[self.val_cnt])] + "\\" +self.val_files[self.val_cnt]),self.img_size) / 255
                
                # for one hot encoding
                Y[i][int(self.val_y[self.val_cnt])] = 1.
        
                self.val_cnt += 1
                if self.val_cnt == len(self.val_files):
                    self.val_cnt = 0
            yield X,Y


