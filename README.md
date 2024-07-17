# Brain Tumor Detection Using a Convolutional Neural Network
## Overview
This project aims to detect brain tumors from MRI images using a Convolutional Neural Network (CNN). The model is built using TensorFlow and Keras, leveraging various layers such as Conv2D, MaxPooling2D, and Dense to create a robust architecture for image classification. This project was developed as part of a hackathon.
## Table of Contents
1. Installation
2. Data Preparation & Preprocessing
3. Model Architecture
4. Training
5. Evaluation
6. Results
7. Usage
8. Contributing
### Installation
To get started with the project, ensure you have the following dependencies installed:

1. TensorFlow
2. Keras
3. OpenCV
4. Imutils
5. Scikit-learn
6. Matplotlib
7. Numpy

### Data Preparation & Preprocessing
The project includes a function to preprocess the MRI images by cropping the brain contours:
code:
def crop_brain_contour(image, plot=False):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=2)

    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    c = max(cnts, key=cv2.contourArea)
    
    extLeft = tuple(c[c[:, :, 0].argmin()][0])
    extRight = tuple(c[c[:, :, 0].argmax()][0])
    extTop = tuple(c[c[:, :, 1].argmin()][0])
    extBot = tuple(c[c[:, :, 1].argmax()][0])
    
    new_image = image[extTop[1]:extBot[1], extLeft[0]:extRight[0]]            

    if plot:
        plt.figure()
        plt.subplot(1, 2, 1)
        plt.imshow(image)
        plt.title('Original Image')
        plt.subplot(1, 2, 2)
        plt.imshow(new_image)
        plt.title('Cropped Image')
        plt.show()
    
    return new_image

### Model Architecture
The CNN model is built using TensorFlow and Keras, with layers such as Conv2D, MaxPooling2D, Flatten, and Dense:
code:
input_shape = (128, 128, 3)
model_input = Input(shape=input_shape)
x = ZeroPadding2D((3, 3))(model_input)
x = Conv2D(32, (7, 7), strides=(2, 2))(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = MaxPooling2D((3, 3))(x)
x = Flatten()(x)
x = Dense(1, activation='sigmoid')(x)

model = Model(inputs=model_input, outputs=x)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

### Training
The model is trained on preprocessed MRI images with the following parameters:

Optimizer: Adam
Loss function: Binary Crossentropy
Metrics: Accuracy

### Evaluation
The model's performance is evaluated using accuracy and F1 score.

### Results
Details about the model's performance, including accuracy and F1 score on the test set, will be included here.

### Usage
To use the model for predicting brain tumors on new MRI images, follow these steps:

1. Preprocess the image using the crop_brain_contour function.
2. Load the trained model.
3. Use the model to predict the presence of a tumor.

### Contributing
Contributions are welcome! Please submit a pull request or open an issue to discuss your ideas.
