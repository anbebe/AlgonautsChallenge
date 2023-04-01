'''
Loads the Algonauts challenge dataset from given directory and builds seperate tensorflow dataset for further processing.
Code from https://colab.research.google.com/drive/1bLJGP3bAo_hAOwZPHpiSHKlt97X9xsUw?usp=share_link#scrollTo=S5-uT-S9zIQ0
but changed in huge parts to work with tensorflow instead of PyTorch
'''
import os
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

class ChallengeDataset:
    '''
        data class that contains the loading and preprocessing of the challenge dataset
    '''
    def __init__(self, data_dir, submission_dir, subj, img_height = 320, img_width = 320, batch_size = 100):
        self.data_dir = data_dir
        self.submission_dir = submission_dir
        self.subj=int(subj)

        # create the data directories
        args = argObj(self.data_dir, self.submission_dir, self.subj)
        # train and test directories corresponding to the chosen subject
        self.train_img_dir  = os.path.join(args.data_dir, 'training_split', 'training_images')
        self.test_img_dir  = os.path.join(args.data_dir, 'test_split', 'test_images')

        # get amount of train and test data (images)
        train_img_list = os.listdir(self.train_img_dir)
        self.len_train_ds = len(train_img_list)
        test_img_list = os.listdir(self.test_img_dir)
        self.len_test_ds = len(test_img_list)

        # load the fmri data for the training and validation dataset
        self.fmri_dir = os.path.join(args.data_dir, 'training_split', 'training_fmri')
        self.lh_fmri = np.load(os.path.join(self.fmri_dir, 'lh_training_fmri.npy'))
        self.rh_fmri = np.load(os.path.join(self.fmri_dir, 'rh_training_fmri.npy'))

        self.img_height = img_height
        self.img_width = img_width
        self.batch_size = batch_size

        # preprocess the datasets
        self.train_ds, self.val_ds, self.test_ds = self.create_ds(self.batch_size)

        
    

    def create_ds(self, batch_size):
        '''
            load the datasets from directory, preprocess the image data and add the fmri data corresponding
            to the training images and split training into training and validation,

            return training, validation and test dataset
        '''
        # load train and test images from the given data directory
        train_ds = tf.keras.utils.image_dataset_from_directory(
            self.train_img_dir,
            color_mode="rgb",
            labels=None,
            shuffle=False,
            seed=None,
            image_size=(self.img_height, self.img_width),
            batch_size=None)

        test_ds = tf.keras.utils.image_dataset_from_directory(
            self.test_img_dir,
            color_mode="rgb",
            labels=None,
            shuffle=False,
            seed=None,
            image_size=(self.img_height, self.img_width),
            batch_size=None)
        
        def normalize_rgb(img):
            '''
            Normalize the images color channels by the same values as in the algonauts colab tutorial 
            Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) (Pytorch version in the tutorial)

            return normalised values
            '''
            out = []
            out.append(img[:,:,0] - 0.485 / 0.229)
            out.append(img[:,:,1] - 0.456 / 0.224)
            out.append(img[:,:,2] - 0.406 / 0.225)
            return tf.stack(out, axis=-1)
        
        # normalize image values
        train_ds = train_ds.map(lambda x: normalize_rgb(x/255.))
        test_ds = test_ds.map(lambda x: normalize_rgb(x/255.))

        #Enumerate Dataset. This will be used to map the labels to the images
        train_ds = train_ds.enumerate(start=1)
        # test_ds = test_ds.enumerate(start=1)

        #Convert fMRI-data to tensors to make use of tensorflow functionality
        lh_fmri_tensor = tf.constant(self.lh_fmri)
        rh_fmri_tensor = tf.constant(self.rh_fmri)

        #delete unneeded fMRI-numpy-data 
        del self.lh_fmri
        del self.rh_fmri

        #For each enumerated img in the dataset (tuple of shape (indx,img)), this function will map the respective fmri data to the image, using the enumeration as index for the fMRI tensor. Returns a tuple of shape (img,lh_fmri,rh_fmri)
        def add_fmri(indx,img):

            index = int(indx)

            #Get fMRI-data for the respective index
            lh_fmri = tf.gather(lh_fmri_tensor, index)
            rh_fmri = tf.gather(rh_fmri_tensor, index)

            return img, lh_fmri, rh_fmri

        # add fmri data to the images
        train_ds = train_ds.map(add_fmri)
        # split train dataset in train and validation dataset
        train_ds = train_ds.take(self.len_train_ds - 1).shuffle(1000)
        train_ds = train_ds.take(int(0.8*self.len_train_ds)).batch(batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)

        val_ds = train_ds.skip(int(0.8*self.len_train_ds)).batch(batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)

        # preprocess test dataset
        #test_ds = test_ds.take(self.len_test_ds - 1)
        test_ds = test_ds.batch(batch_size, drop_remainder=True)
        
        return train_ds, val_ds, test_ds
    

    
class argObj:
    def __init__(self, data_dir, parent_submission_dir, subj):
    
        self.subj = format(subj, '02')
        self.data_dir = os.path.join(data_dir, 'subj'+self.subj)
        self.parent_submission_dir = parent_submission_dir
        self.subject_submission_dir = os.path.join(self.parent_submission_dir,
            'subj'+self.subj)

        # Create the submission directory if not existing
        if not os.path.isdir(self.subject_submission_dir):
            os.makedirs(self.subject_submission_dir)
