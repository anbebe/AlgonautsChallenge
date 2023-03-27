from model import ObjectDetectionModel
from data_preprocess import ChallengeDataset
import tensorflow as tf

def load_and_test_model(model, data):
    ft = model.feature_extractor(data, 0)

    print(tf.shape(ft))

if __name__ == "__main__":
    data = ChallengeDataset(data_dir='./data', submission_dir='../algonauts_2023_challenge_submission')
    model = ObjectDetectionModel()
    for element in data.train_ds.take(1):
        load_and_test_model(model, element[:][0])
    