from model import ObjectDetectionModel
from data_preprocess import ChallengeDataset
import tensorflow as tf
from tqdm import tqdm
import sklearn
from sklearn.decomposition import IncrementalPCA
import numpy as np

def load_and_test_model(model, data):
    ft = model.feature_extractor(data, 0)

    print(tf.shape(ft))

def fit_pca(model, dataloader, batch_size):

    feature_indx = 5

    # Define PCA parameters
    pca = IncrementalPCA(n_components=50, batch_size=batch_size)

    # Fit PCA to batch
    for _, d in tqdm(enumerate(dataloader), total=len(dataloader)-2):
        # Extract features 
        ft =  model.feature_extractor(d[0], feature_indx)
        # Flatten the features
        ft = np.hstack([np.reshape(l,(batch_size,-1)) for l in ft])
        # Fit PCA to batch
        pca.partial_fit(ft)
    return pca

def extract_features(model, dataloader, pca, batch_size, feature_indx):

    features = []
    for _, d in tqdm(enumerate(dataloader), total=len(dataloader)-2):
        # Extract features
        ft =  model.feature_extractor(d[0], feature_indx)
        ft = np.hstack([np.reshape(l,(batch_size,-1)) for l in ft])
        # Apply PCA transform
        ft = pca.transform(ft)
        features.append(ft)
    return np.vstack(features)



if __name__ == "__main__":
    data = ChallengeDataset(data_dir='./data', submission_dir='../algonauts_2023_challenge_submission')
    model = ObjectDetectionModel()
    #for element in data.train_ds.take(1):
    #    load_and_test_model(model, element[:][0])
    train_ds = data.train_ds
    #pca = fit_pca(train_ds.take(2), 100)
    #pca = fit_pca(model, train_ds, 100)
    #features_train = extract_features(model, train_ds, pca, 100, 5)

    print('\nTraining images features:')
    #print(features_train.shape)
    print('(Training stimulus images × PCA features)')

    #np.save('train_features_retinanet.npy', features_train)
    