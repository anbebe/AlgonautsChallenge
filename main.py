from model import ObjectDetectionModel
from data_preprocess import ChallengeDataset
from visualisation import VisualizeFMRI
import tensorflow as tf
import tensorflow_datasets as tfds
from tqdm import tqdm
import sklearn
from sklearn.decomposition import IncrementalPCA
from sklearn.linear_model import LinearRegression
import numpy as np
import argparse
from scipy.stats import pearsonr as corr

def fit_pca(model, dataloader, batch_size, feature_indx):
    '''
    fits the pca in parts to the extracted features due to the large dataset
    from https://colab.research.google.com/drive/1bLJGP3bAo_hAOwZPHpiSHKlt97X9xsUw?usp=share_link#scrollTo=S5-uT-S9zIQ0
    but changed in huge parts to work with tensorflow instead of PyTorch
    '''
    # Define PCA parameters
    pca = IncrementalPCA(n_components=100, batch_size=batch_size)

    # Fit PCA to batch
    for _, d in tqdm(enumerate(dataloader), total=len(dataloader)):
        # Extract features 
        ft =  model.feature_extractor(d[0], feature_indx)
        # Flatten the features
        ft = np.hstack([np.reshape(l,(batch_size,-1)) for l in ft])
        # Fit PCA to batch
        pca.partial_fit(ft)
    return pca

def extract_features(model, dataloader, pca, batch_size, feature_indx):
    '''
    get features and downsample them with pca
    from https://colab.research.google.com/drive/1bLJGP3bAo_hAOwZPHpiSHKlt97X9xsUw?usp=share_link#scrollTo=S5-uT-S9zIQ0
    but changed in huge parts to work with tensorflow instead of PyTorch
    '''
    features = []
    for _, d in tqdm(enumerate(dataloader), total=len(dataloader)):
        # Extract features
        ft =  model.feature_extractor(d[0], feature_indx)
        ft = np.hstack([np.reshape(l,(batch_size,-1)) for l in ft])
        # Apply PCA transform
        ft = pca.transform(ft)
        features.append(ft)
    return np.vstack(features)

def extract_features_test(model, dataloader, pca, batch_size, feature_indx):
    '''
    get features from test set (only has images, therefore different shape) and downsample them with pca 
    from https://colab.research.google.com/drive/1bLJGP3bAo_hAOwZPHpiSHKlt97X9xsUw?usp=share_link#scrollTo=S5-uT-S9zIQ0
    but changed in huge parts to work with tensorflow instead of PyTorch
    '''
    features = []
    for _, d in tqdm(enumerate(dataloader), total=len(dataloader)):
        # Extract features
        ft =  model.feature_extractor(d, feature_indx)
        ft = np.hstack([np.reshape(l,(batch_size,-1)) for l in ft])
        # Apply PCA transform
        ft = pca.transform(ft)
        features.append(ft)
    return np.vstack(features)

def train_linear_regression(features, lh_fmri, rh_fmri):
    '''
    preprocess the fmri data to get numpy arrays and fit linear regression on the features and the fmri data
    from https://colab.research.google.com/drive/1bLJGP3bAo_hAOwZPHpiSHKlt97X9xsUw?usp=share_link#scrollTo=S5-uT-S9zIQ0
    but changed in huge parts to work with tensorflow instead of PyTorch
    '''
    # convert the left hemisphere fmri dataset to a numpy array
    lh_fmri = np.vstack(tfds.as_numpy(lh_fmri))

    # convert the right hemisphere fmri dataset to a numpy array
    rh_fmri = np.vstack(tfds.as_numpy(rh_fmri))

    # fit liear regression to predict the fmri data from the extracted features
    reg_lh = LinearRegression().fit(features, lh_fmri)
    reg_rh = LinearRegression().fit(features, rh_fmri)

    return reg_lh, reg_rh

def test_linear_regression(features, reg_lh, reg_rh):
    '''
    get predictions from the linear regression model from the features
    '''
    lh_fmri_pred = reg_lh.predict(features)
    rh_fmri_pred = reg_rh.predict(features)
    return lh_fmri_pred, rh_fmri_pred

def compute_accuracy(lh_fmri_val_pred, rh_fmri_val_pred, lh_fmri_val, rh_fmri_val):
    '''
    Compute the accuracy with the Pearson correlation
    from https://colab.research.google.com/drive/1bLJGP3bAo_hAOwZPHpiSHKlt97X9xsUw?usp=share_link#scrollTo=S5-uT-S9zIQ0
    '''
    # Empty correlation array of shape: (LH vertices)
    lh_correlation = np.zeros(lh_fmri_val_pred.shape[1])
    # Correlate each predicted LH vertex with the corresponding ground truth vertex
    for v in tqdm(range(lh_fmri_val_pred.shape[1])):
        lh_correlation[v] = corr(lh_fmri_val_pred[:,v], lh_fmri_val[:,v])[0]

    # Empty correlation array of shape: (RH vertices)
    rh_correlation = np.zeros(rh_fmri_val_pred.shape[1])
    # Correlate each predicted RH vertex with the corresponding ground truth vertex
    for v in tqdm(range(rh_fmri_val_pred.shape[1])):
       rh_correlation[v] = corr(rh_fmri_val_pred[:,v], rh_fmri_val[:,v])[0]

    return lh_correlation, rh_correlation


if __name__ == "__main__":
    # get arguments from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument('--action', default='full_process', choices=['get_features','train_regression','full_process','evaluate'], help='execution mode')
    parser.add_argument('--data_dir', help='direction of the challenge data (should be parent directory of the subjects)')
    parser.add_argument('--subj', default = "1", choices=["1","2","3","4","5","6","7","8"], help='which subject data should be used')
    parser.add_argument('--submission_dir', default='../algonauts_2023_challenge_submission', help='direction to store the voxel predictions, if not existend, it will be created')
    parser.add_argument('--feature_map', default="5", choices=["0","1","2","3","4","5","6","7"], help='which feature map to extract from the model')

    args = parser.parse_args()

    # load and preprocess the challenge data for the given subject
    data = ChallengeDataset(args.data_dir, args.submission_dir, args.subj)
    train_ds = data.train_ds
    val_ds = data.val_ds
    test_ds = data.test_ds

    # depending on the action mode, execute all steps or single steps
    # Step 1: Load model, fit pca and get downsampled features for every dataset (train, val, test)
    if args.action == 'full_process' or args.action == 'get_features':

        # load the pretrained retinanet
        model = ObjectDetectionModel()
        # fit pca to the train features
        pca = fit_pca(model, train_ds, 100, int(args.feature_map))

        # get the downsampled features for every dataset and save them in the current working directory
        features_train = extract_features(model, train_ds, pca, 100, int(args.feature_map))
        np.save('features_train_retinanet' + args.subj + '.npy', features_train)
        features_val= extract_features(model, val_ds, pca, 100, int(args.feature_map))
        np.save('features_val_retinanet' + args.subj + '.npy', features_val)
        features_test = extract_features(model, test_ds, pca, 100, int(args.feature_map))
        np.save('features_test_retinanet' + args.subj + '.npy', features_test)

        # print the shapes of the downsampled features
        print('\nTraining images features:')
        print(features_train.shape)
        print('(Training stimulus images × PCA features)')
        print('\Validation images features:')
        print(features_val.shape)
        print('(Validation stimulus images × PCA features)')
        print('\Test images features:')
        print(features_test.shape)
        print('(Test stimulus images × PCA features)')
    
    # Step 2: train a regression model on the downsampled training features and the corresponding fmri data
    if args.action == 'train_regression':
        # if not full process, load the downsampled features from the current working directory
        features_train = np.load('features_train_retinanet' + args.subj +'.npy')
        features_val = np.load('features_val_retinanet' + args.subj +'.npy')
        features_test = np.load('features_test_retinanet' + args.subj +'.npy')

    if args.action == 'full_process' or args.action == 'train_regression':
        # train linear regression on the training data and predict fmri data for the validation and the test datasets
        reg_lh, reg_rh = train_linear_regression(features_train, train_ds.map(lambda _,lh,__: lh), train_ds.map(lambda _,__, rh: rh))
        lh_fmri_val_pred, rh_fmri_val_pred = test_linear_regression(features_val, reg_lh, reg_rh)
        lh_fmri_test_pred, rh_fmri_test_pred = test_linear_regression(features_test, reg_lh, reg_rh)
        
        # save the predictions for left and right hemisphere for test and validation data in the current working directory
        np.save('lh_fmri_predictions_val_retinanet' + args.subj +'.npy', lh_fmri_val_pred)
        np.save('rh_fmri_predictions_val_retinanet' + args.subj +'.npy', rh_fmri_val_pred)
        np.save('lh_fmri_predictions_test_retinanet' + args.subj +'.npy', lh_fmri_test_pred)
        np.save('rh_fmri_predictions_test_retinanet' + args.subj +'.npy', rh_fmri_test_pred)

    # Step 3: Compute the accuracy of predictions for the validation data and visualise it
    if args.action == 'evaluate':
        # if not full process, load the predictions from the current working directory
        lh_fmri_val_pred = np.load('lh_fmri_predictions_val_retinanet' + args.subj +'.npy')
        rh_fmri_val_pred = np.load('rh_fmri_predictions_val_retinanet' + args.subj +'.npy')
    if args.action == 'full_process' or args.action == 'evaluate':
        # extract the fmri data for each left and right hemisphere from the validation dataset and transfer it to a numpy array
        lh_fmri_val = np.vstack(tfds.as_numpy(val_ds.map(lambda _,lh,__: lh)))
        rh_fmri_val = np.vstack(tfds.as_numpy(val_ds.map(lambda _,__, rh: rh)))

        # compute accuracy for the validation data with pearson correlation
        lh_correlation, rh_correlation = compute_accuracy(lh_fmri_val_pred, rh_fmri_val_pred, lh_fmri_val, rh_fmri_val)

        # laod visualisation class and output graph showing accuracy grouped by brain regions
        visualization = VisualizeFMRI(args.data_dir + '/subj0'+ args.subj)
        visualization.visualize_rois(lh_correlation, rh_correlation, 'subj0' + args.subj + ".png")

