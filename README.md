Here's your revised project summary with **transfer learning using EfficientNetB0** and updated **performance results**:

---

<h1>Sickle-Cell-Anemia-Detection</h1>

This project focuses on detecting sickle cell anaemia from blood cell images using deep learning, specifically leveraging **transfer learning with EfficientNetB0**. Sickle cell anaemia is a genetic blood disorder marked by the presence of abnormally shaped red blood cells, resembling a sickle. Early diagnosis is vital to effective treatment and disease management.

<h2>Dataset</h2>

The dataset comprises two categories:

* **Normal Cells**: Images of normal red blood cells
* **Sickle Cells**: Images of abnormally shaped red blood cells indicating sickle cell anaemia

<p>A total of 569 images are used — 147 normal and 422 sickle cells — each sized at 256x256 pixels.</p>  
Dataset link: [Kaggle - Sickle Cell Disease Dataset](https://www.kaggle.com/datasets/florencetushabe/sickle-cell-disease-dataset)

<h2>Model Architecture</h2>

We used **EfficientNetB0**, a state-of-the-art transfer learning model, for enhanced performance. The architecture includes:

* **Base Model**: Pretrained EfficientNetB0 (excluding the top layer)
* **Global Average Pooling Layer**
* **Dense Layer**: Fully connected with dropout for regularization
* **Output Layer**: Single neuron with sigmoid activation for binary classification

<h2>Training</h2>

* **Dataset Split**: 70% Training, 20% Validation, 10% Testing
* **Optimizer**: Adam
* **Loss Function**: Binary Cross-Entropy
* **Epochs**: 20
* **Image Size**: 256x256 pixels resized for input
* **Augmentation**: Applied to enhance model generalization

<h2>Evaluation</h2>

The model achieved a **test accuracy of 94%**, significantly improving over the previous CNN-based model (78%). The improvement is attributed to the powerful feature extraction capabilities of EfficientNetB0.

<h2>Dependencies</h2>

* TensorFlow
* TensorFlow Hub / IO
* NumPy
* Matplotlib
* OpenCV

<h2>Results</h2>

Using **transfer learning with EfficientNetB0**, the model achieved:

* **Validation Accuracy**: \~94%
* **Validation Loss**: \~0.18
* **Test Accuracy**: 94%

This confirms the model’s robustness in distinguishing sickle cells from normal red blood cells and its readiness for potential real-time applications.

---

Would you like a code snippet or diagram for the updated model architecture?
