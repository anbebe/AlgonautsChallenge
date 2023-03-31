import os
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
#from google.colab import drive

class ChallengeDataset:
    def __init__(self, data_dir, submission_dir, subj, img_height = 320, img_width = 320, batch_size = 100):
        # drive.mount('/content/drive/', force_remount=True)
        #data_dir = '/content/drive/MyDrive/algonauts_2023_tutorial_data' 
        # parent_submission_dir = '/content/drive/MyDrive/algonauts_2023_challenge_submission' 
        self.data_dir = data_dir
        self.submission_dir = submission_dir
        self.subj=int(subj)

        args = argObj(self.data_dir, self.submission_dir, self.subj)

        self.train_img_dir  = os.path.join(args.data_dir, 'training_split', 'training_images')
        self.test_img_dir  = os.path.join(args.data_dir, 'test_split', 'test_images')

        train_img_list = os.listdir(self.train_img_dir)
        self.len_train_ds = len(train_img_list)
        test_img_list = os.listdir(self.test_img_dir)
        self.len_test_ds = len(test_img_list)

        self.fmri_dir = os.path.join(args.data_dir, 'training_split', 'training_fmri')
        self.lh_fmri = np.load(os.path.join(self.fmri_dir, 'lh_training_fmri.npy'))
        self.rh_fmri = np.load(os.path.join(self.fmri_dir, 'rh_training_fmri.npy'))

        self.img_height = img_height
        self.img_width = img_width
        self.batch_size = batch_size

        self.train_ds, self.val_ds, self.test_ds = self.create_ds(self.batch_size)

        
    

    def create_ds(self, batch_size):

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
            output[channel] = (input[channel] - mean[channel]) / std[channel]
            Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # normalize the images color channels
            '''
            out = []
            out.append(img[:,:,0] - 0.485 / 0.229)
            out.append(img[:,:,1] - 0.456 / 0.224)
            out.append(img[:,:,2] - 0.406 / 0.225)
            return tf.stack(out, axis=-1)
        
        train_ds = train_ds.map(lambda x: normalize_rgb(x))
        test_ds = test_ds.map(lambda x: normalize_rgb(x))

        #Enumerate Dataset. This will be used to map the labels to the images
        train_ds = train_ds.enumerate(start=1)
        test_ds = test_ds.enumerate(start=1)

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

        train_ds = train_ds.map(add_fmri)
        train_ds = train_ds.take(self.len_train_ds - 1)
        train_ds_ = train_ds.shuffle(1000)
        train_ds = train_ds_.take(int(0.8*self.len_train_ds)).batch(batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)

        val_ds = train_ds_.skip(int(0.8*self.len_train_ds)).batch(batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)

        test_ds = test_ds.map(add_fmri)
        test_ds = test_ds.take(self.len_test_ds - 1)
        test_ds = test_ds.shuffle(1000).batch(batch_size, drop_remainder=True)
        
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

'''
if __name__ == "__main__":
    data = ChallengeDataset(data_dir='./data', submission_dir='../algonauts_2023_challenge_submission')
    for element in data.train_ds.take(1):
        print("image shape: ", element[0][0].shape)
        print("lh_fmri: ", element[0][1].shape)
        print("rh_fmri: ", element[0][2].shape)
'''




