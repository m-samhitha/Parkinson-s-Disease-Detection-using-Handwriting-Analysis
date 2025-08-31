
# Parkinson's Disease

This project implements a deep learning model for early detection of Parkinson's Disease using handwriting and spiral/wave images. The model leverages the DenseNet201 architecture with transfer learning and fine-tuning for high accuracy.

## Features

* Uses DenseNet201 pre-trained on ImageNet for feature extraction.
* Implements data augmentation to improve generalization.
* Fine-tunes the last layers of DenseNet201 to improve performance.
* Generates confusion matrix and classification report for evaluation.
* Visualizes training and validation accuracy curves.

## Dataset Structure

**dataset:**
https://drive.google.com/drive/folders/1LF6WhycNwusBbwtB-R5qCGL5YSQxsbk0?usp=sharing

The dataset should be organized as follows:

Parkinsons/
│
├── training/
│   ├── healthy/
│   └── parkinson/
│
└── testing/
    ├── healthy/
    └── parkinson/


> Make sure to update `base_dir` in the code with the path to your dataset in Google Drive.

## Requirements

* Python 3.x
* TensorFlow 2.x
* Keras
* matplotlib
* scikit-learn
* seaborn
* Google Colab (recommended for GPU acceleration)

Install dependencies using:

pip install tensorflow matplotlib scikit-learn seaborn
```

## How to Run

1.**Mount Google Drive:**

```python
from google.colab import drive
drive.mount('/content/drive')
```

2. **Set dataset path**:

```python
base_dir = "/content/drive/MyDrive/Parkinsons"
```

3. **Train the model**:

```python
history = model.fit(
    train_generator,
    validation_data=test_generator,
    epochs=20,
    callbacks=callbacks
)
```

4. **Fine-tune the model**:

```python
base_model.trainable = True
for layer in base_model.layers[:-30]:
    layer.trainable = False

history_finetune = model.fit(
    train_generator,
    validation_data=test_generator,
    epochs=10,
    callbacks=callbacks
)
```

5. **Evaluate the model**:

```python
loss, acc = model.evaluate(test_generator)
print(f"Final Test Accuracy: {acc*100:.2f}%")
```

## Model Architecture

* **Base Model:** DenseNet201 (ImageNet weights)
* **Pooling Layer:** GlobalAveragePooling2D
* **Fully Connected Layer:** Dense(256) with ReLU
* **Dropout Layers:** 0.5 and 0.3
* **Output Layer:** Dense(1) with Sigmoid (binary classification)
* **Optimizer:** Adam
* **Loss Function:** Binary Crossentropy

## Evaluation

* **Accuracy curves:** Training vs Validation plotted for each epoch.
* **Confusion Matrix:** Visual representation of predictions.
* **Classification Report:** Precision, Recall, F1-score for each class.

## References

* [DenseNet201 on Keras Applications](https://www.tensorflow.org/api_docs/python/tf/keras/applications/DenseNet201)
* [TensorFlow ImageDataGenerator Documentation](https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/ImageDataGenerator)


## Contact
1.	Samhitha Moparthi
    Email: samhithamoparthi@gmail.com

2.	Gayathri Mocharla
    Email: mocharlagayathri@gmail.com



If you want, I can also make a **shorter, GitHub-friendly version with badges and a clean layout** that looks very professional on the repo page. Do you want me to do that?
