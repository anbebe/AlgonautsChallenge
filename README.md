# AlgonautsChallenge in Tensorflow
<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#citations">Citations</a></li>
  </ol>
</details>

<!-- ABOUT THE PROJECT -->
## About The Project

The Algonauts Project 2023 Challenge (https://doi.org/10.48550/arXiv.2301.03198) <br />
The Natural Scene Dataset (NSD)(https://doi.org/10.1038/s41593-021-00962-x) <br />
Official Algonauts Colab Tutorial (https://colab.research.google.com/drive/1bLJGP3bAo_hAOwZPHpiSHKlt97X9xsUw?usp=share_link) <br />
Official RetinaNet Paper (http://arxiv.org/pdf/1708.02002v2) <br />
Pre-trained RetinaNet Model (https://huggingface.co/keras-io/Object-Detection-RetinaNet)

We provide a Tensorflow implementation of the official algonauts colab tutorial that uses a pre-trained retinanet to predict fMRI data from natural scenes images.
This implementation follows the same steps as the official algonauts colab tutorial, in order to build a linearized encoding model, that are 
1. Loading the challenge dataset.
2. Building the RetinaNet model and loading its pretrained weights.
3. Extracting features from intermediate layers of the RetinaNet and reducing their dimensionality with pca.
4. Training a linear regression from the extracted features to the corresponding fMRI data and predict with it responses for the challenge test data.
5. Evaluate the linear regression and visualise the accuracy grouped in brain regions.

Further, we provide a project report that goes more into detail about related research, methods and results, as well as a presentation video.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- GETTING STARTED -->
## Getting Started

In order to use this repository, one has to accept the terms and then download the [official challenge data](https://docs.google.com/forms/d/e/1FAIpQLSehZkqZOUNk18uTjRTuLj7UYmRGz-OkdsU25AyO3Wm6iAb0VA/viewform?usp=sf_link)
and keep the directory structure as proposed there.

To get a local copy up and running, follow the next steps to build a conda environment with all necessary prerequisites.

### Prerequisites

First, create a conda environment and activate it and then install the following packages.
* pip
  ```sh
  pip install tensorflow
  pip install tensorflow_datasets
  pip install keras
  pip install scikit-learn
  pip install huggingface_hub
  ```
  * conda
  ```sh
  conda install -c conda-forge tqdm
  conda install -c conda-forge numpy
  ```

### Installation
1. Clone the repo
   ```sh
   git clone https://github.com/anbebe/AlgonautsChallenge.git
   ```
2. Download challenge data
3. Create conda environment and install required packages as listed above
4. Run the main.py with at least the data directory given as argument (or with other specified arguments if wanted, descibed in the next section)


<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- USAGE EXAMPLES -->
## Usage
You can either go through all steps (listed above) from extracting features to predicting brain responses, or just do one step at a time. 
In both cases, the challenge data is loaded and the intermediate results, like extracted features and predicted values, are saved as .npy files due to the large dataset and its time to compute
the single steps.  <br />
The following arguments exist and can be used as described: <br />
-`--data_dir`: mandatory, direction of the challenge data (should be parent directory of the subjects)
- `--action`: execution mode, either full_process that goes through all steps or  a single step, that needs .npy files of the further steps saved in the working directory
  - `full_process`: default, goes through all steps from extracting features over pca, linear regression to evaluating and visualising the predicted values
  - `get_features`: loads the pre-trained RetinaNet, loads the feature map from either given or else default layer, fits pca and saves the downsampled features by pca for train, validation and test data.
  - `train_regression`: loads downsampled features for train, validation and test dataset from .npy files in the working directory, trains a linear regression model on the train data, predicts the values for validation and test data, saves the predictions as .npy in the working directory
  - `evaluate`: loads the predicted values for the validation set from the working directory, computes accuracy through the pearson correlation, show and save the visualisation of the accuracy grouped by brain regions
- `--subj`: which subject data should be used (default=1)
  - choices: [1,2,3,4,5,6,7,8]
- `--submission_dir`: direction to store the voxel predictions, if not existend, it will be created (default=../algonauts_2023_challenge_submission)
- `--feature_map`: which feature map to extract from the model, earlier layers have ore parameters and therefore extracting features needs more computations and therefore time (default=5)
  - choices: [1,2,3,4,5,6,7,8]
  - the outputs 1,2,3 come from the ResNet backbone while the last five come from the RetinaNet build on top of that
  
  
<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE.txt` for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- Citations -->
## Citations
1. Gifford AT, Lahner B, Saba-Sadiya S, Vilas MG, Lascelles A, Oliva A, Kay K, Roig G, Cichy RM. 2023. The Algonauts Project 2023 Challenge: How the Human Brain Makes Sense of Natural Scenes. arXiv preprint, arXiv:2301.03198. DOI: https://doi.org/10.48550/arXiv.2301.03198
2. Allen EJ, St-Yves G, Wu Y, Breedlove JL, Prince JS, Dowdle LT, Nau M, Caron B, Pestilli F, Charest I, Hutchinson JB, Naselaris T, Kay K. 2022. A massive 7T fMRI dataset to bridge cognitive neuroscience and computational intelligence. Nature Neuroscience, 25(1):116–126. DOI: https://doi.org/10.1038/s41593-021-00962-x
3. Lin, T.Y., Goyal, P., Girshick, R., He, K. and Dollár, P., 2017. Focal loss for dense object detection. In Proceedings of the IEEE international conference on computer vision (pp. 2980-2988). DOI: https://doi.org/10.1109/iccv.2017.324

<p align="right">(<a href="#readme-top">back to top</a>)</p>



