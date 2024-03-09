# Clustering_Unsupervised_Learning

## Overview
This project focuses on developing a robust face recognition system using various machine learning techniques. The workflow involves data preprocessing, noise injection and reduction, and clustering of facial images based on similarity. The UMIST dataset is utilized for training and evaluation.

## Features
- Data Normalization: The UMIST dataset is preprocessed and normalized to ensure consistent and standardized input for subsequent steps.
- Noise Injection with Convolutional Neural Network (CNN): A CNN is employed to introduce controlled noise into facial images. This step simulates real-world scenarios where images may be affected by various environmental factors.
- Noise Reduction with Autoencoder: An autoencoder is implemented to denoise the images. This helps enhance the accuracy of subsequent facial recognition and clustering steps.
- Clustering Algorithms: Three clustering algorithms - KMeans, Agglomerative, and DBSCAN - are applied to group facial images with similar features. This facilitates efficient organization and retrieval of faces.
- Face Recognition Models: Two deep learning models - CNN and ResNet-50 - are trained for face recognition. These models aim to identify and differentiate faces within the clustered groups.
- Evaluation: The performance of the entire system is evaluated using the visuals by plotting the accuracy and loss curves along with the visuals of the grouping of the faces and the effectiveness of clustering algorithms and face recognition models is analyzed.

## Dependencies
- Python 3.9+
- Tensorflow 2.x
- flask
- numpy, pandas, keras, sklearn, matplotlib

## Getting Started
- Clone this repository
- Install dependencies from requirements.txt: `pip install -r requirements.txt`
- Run the script: `python app.py`

## Additional Details
- Details on model training and the clustering algorithms can be found in the 'project_four' file.
- Details on the evaluation of the models can be found in `project_final` file.

## Contributing
1. Fork the repository.
2. Create a new branch: `git checkout -b feature_branch`.
3. Commit your changes: `git commit -m 'Add new feature'`.
4. Push to the branch: `git push origin feature_branch`.
5. Open a pull request.
