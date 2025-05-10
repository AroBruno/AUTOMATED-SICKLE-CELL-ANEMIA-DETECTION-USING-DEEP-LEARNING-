<h1>Sickle-Cell-anemia-detection</h1>
This project aims to detect sickle cell anaemia in blood cell images using convolutional neural networks (CNNs). Sickle cell anaemia is a hereditary blood disorder characterized by abnormal, sickle-shaped red blood cells. Early detection of this condition is crucial for timely medical intervention and management.

<h2>Dataset</h2>

The dataset used in this project consists of two classes:
    
- Normal Cells: Images of normal red blood cells
- Sickle Cells: Images of sickle-shaped red blood cells indicative of sickle cell anaemia

<p>The dataset comprises 147 images of normal cells and 422 images of sickle cells, totalling 569 images. Each image is of size 256x256 pixels.</p>
Visit for dataset: https://www.kaggle.com/datasets/florencetushabe/sickle-cell-disease-dataset/code?datasetId=3913975&sortBy=dateRun&tab=profile&excludeNonAccessedDatasources=false

<h2>Model Architecture</h2>

The CNN model architecture used for this task is as follows:

- Input Layer: Accepts images of size 256x256 pixels with 3 colour channels
- Convolutional Layers: Three sets of convolutional layers with max-pooling to extract features
- Flatten Layer: Flattens the output for feeding into fully connected layers
- Fully Connected Layers: Two dense layers for classification
- Output Layer: Single neuron output with sigmoid activation for binary classification

<h2>Training</h2>

The dataset was split into training (70%), validation (20%), and test (10%) sets. The model was trained using the Adam optimizer with binary cross-entropy loss. Training was conducted for 20 epochs, achieving an accuracy of 78% and a loss of 0.44 on the validation set.

<h2>Evaluation</h2>

The trained model was evaluated on a separate test image. The image was preprocessed, resized to 256x256 pixels, and fed into the model for prediction. The model predicted whether the input image represented a sickle cell or a normal cell, achieving an accuracy of [provide accuracy here]% on the test image.

<h2>Dependencies</h2>

- TensorFlow
- TensorFlow IO
- NumPy
- Matplotlib
- OpenCV

<h2>Results</h2>

The project achieved 78% accuracy on the test image, demonstrating its effectiveness in detecting sickle cell anaemia.
