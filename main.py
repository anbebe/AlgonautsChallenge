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

def load_and_test_model(model, data):
    ft = model.feature_extractor(data, 0)

    print(tf.shape(ft))

def fit_pca(model, dataloader, batch_size):

    feature_indx = 5

    # Define PCA parameters
    pca = IncrementalPCA(n_components=50, batch_size=batch_size)

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

    features = []
    for _, d in tqdm(enumerate(dataloader), total=len(dataloader)):
        # Extract features
        ft =  model.feature_extractor(d[0], feature_indx)
        ft = np.hstack([np.reshape(l,(batch_size,-1)) for l in ft])
        # Apply PCA transform
        ft = pca.transform(ft)
        features.append(ft)
    return np.vstack(features)

def train_linear_regression(features, lh_fmri, rh_fmri):
    lh_fmri = np.vstack(tfds.as_numpy(lh_fmri))
    print("test: ", lh_fmri.shape)
    rh_fmri = np.vstack(tfds.as_numpy(rh_fmri))
    print("start fitting")
    reg_lh = LinearRegression().fit(features, lh_fmri)
    print("second fitting")
    reg_rh = LinearRegression().fit(features, rh_fmri)
    return reg_lh, reg_rh

def test_linear_regression(features_val, features_test, reg_lh, reg_rh):
    lh_fmri_val_pred = reg_lh.predict(features_val)
    lh_fmri_test_pred = reg_lh.predict(features_test)
    rh_fmri_val_pred = reg_rh.predict(features_val)
    rh_fmri_test_pred = reg_rh.predict(features_test)
    return lh_fmri_val_pred, rh_fmri_val_pred, lh_fmri_test_pred, rh_fmri_test_pred

def compute_accuracy(lh_fmri_val_pred, rh_fmri_val_pred, lh_fmri_val, rh_fmri_val):
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--action', default='full_process', choices=['get_features','train_regression','full_process','evaluate'], help='execution mode')
    parser.add_argument('--data_dir', help='direction of the challenge data (should be parent directory of the subjects)')
    parser.add_argument('--subj', default = "1", choices=["1","2","3","4","5","6","7","8"], help='which subject data should be used')
    parser.add_argument('--submission_dir', default='../algonauts_2023_challenge_submission', help='direction to store the voxel predictions, if not existend, it will be created')
    parser.add_argument('--feature_map', default=5, choices=[0,1,2,3,4,5,6,7], help='which feature map to extract from the model')

    args = parser.parse_args()

    data = ChallengeDataset(args.data_dir, args.submission_dir, args.subj)
    train_ds = data.train_ds
    val_ds = data.val_ds
    test_ds = data.test_ds

    if args.action == 'full_process' or args.action == 'get_features':
        model = ObjectDetectionModel()
        pca = fit_pca(model, train_ds, 100)
        features_train = extract_features(model, train_ds, pca, 100, args.feature_map)
        np.save('features_train_retinanet' + args.subj + '.npy', features_train)
        features_val= extract_features(model, val_ds, pca, 100, args.feature_map)
        np.save('features_val_retinanet' + args.subj + '.npy', features_val)
        features_test = extract_features(model, test_ds, pca, 100, args.feature_map)
        np.save('features_test_retinanet' + args.subj + '.npy', features_test)
        print('\nTraining images features:')
        print(features_train.shape)
        print('(Training stimulus images × PCA features)')
        print('\Validation images features:')
        print(features_val.shape)
        print('(Validation stimulus images × PCA features)')
        print('\Test images features:')
        print(features_test.shape)
        print('(Test stimulus images × PCA features)')
    
    if args.action == 'train_regression':
        features_train = np.load('features_train_retinanet' + args.subj +'.npy')
        features_val = np.load('features_val_retinanet' + args.subj +'.npy')
        features_test = np.load('features_test_retinanet' + args.subj +'.npy')
    if args.action == 'full_process' or args.action == 'train_regression':
        reg_lh, reg_rh = train_linear_regression(features_train, train_ds.map(lambda _,lh,__: lh), train_ds.map(lambda _,__, rh: rh))
        lh_fmri_val_pred, rh_fmri_val_pred, lh_fmri_test_pred, rh_fmri_test_pred = test_linear_regression(features_val, features_test, reg_lh, reg_rh)
        np.save('lh_fmri_predictions_val_retinanet' + args.subj +'.npy', lh_fmri_val_pred)
        np.save('rh_fmri_predictions_val_retinanet' + args.subj +'.npy', rh_fmri_val_pred)
        np.save('lh_fmri_predictions_test_retinanet' + args.subj +'.npy', lh_fmri_test_pred)
        np.save('rh_fmri_predictions_test_retinanet' + args.subj +'.npy', rh_fmri_test_pred)
    if args.action == 'evaluate':
        lh_fmri_val_pred = np.load('lh_fmri_predictions_val_retinanet' + args.subj +'.npy')
        rh_fmri_val_pred = np.load('rh_fmri_predictions_val_retinanet' + args.subj +'.npy')
    if args.action == 'full_process' or args.action == 'evaluate':
        lh_fmri_val = np.vstack(tfds.as_numpy(val_ds.map(lambda _,lh,__: lh)))
        rh_fmri_val = np.vstack(tfds.as_numpy(val_ds.map(lambda _,__, rh: rh)))
        lh_correlation, rh_correlation = compute_accuracy(lh_fmri_val_pred, rh_fmri_val_pred, lh_fmri_val, rh_fmri_val)

        visualization = VisualizeFMRI(args.data_dir)

        visualization.visualize_rois(lh_correlation, rh_correlation)

