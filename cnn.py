import numpy as np
import keras
import seaborn as sns
from matplotlib import pyplot as mplt
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import confusion_matrix
from keras.preprocessing.image import ImageDataGenerator
%matplotlib inline
from google.colab import drive
drive.mount('/content/drive')
import tensorflow as tf 
%matplotlib inline
np.random.seed(20)
train_dataset = loadmat('/content/drive/My Drive/ALDA/train_32x32.mat')
test_dataset = loadmat('/content/drive/My Drive/ALDA/test_32x32.mat')
# Load images and labels

img_train = np.array(train_dataset['X'])
img_test = np.array(test_dataset['X'])

trainLbl = train_dataset['y']
testLbl = test_dataset['y']
# Check the shape of the data

#print(img_train.shape)
#print(img_test.shape)
# Fix the axes of the images

img_train = np.moveaxis(img_train, -1, 0)
img_test = np.moveaxis(img_test, -1, 0)

#print(img_train.shape)
#print(img_test.shape)

mplt.imshow(img_train[13529])
mplt.show()

print('Label: ', trainLbl[13529])
img_train = img_train.astype('float64')
img_test = img_test.astype('float64')
# Convert train and test labels into 'int64' type

trainLbl = trainLbl.astype('int64')
testLbl = testLbl.astype('int64')
# Normalize the images data

#print('Min: {}, Max: {}'.format(img_train.min(), img_train.max()))
#normalise the images
img_test /= 255.0
img_train /= 255.0

binaryform = LabelBinarizer()
trainLbl = binaryform.fit_transform(trainLbl)
testLbl = binaryform.fit_transform(testLbl)
X_train_input, X_validation_input, Y_train_label, Y_validation_label = train_test_split(img_train, trainLbl,test_size=0.15, random_state=22)
Y_validation_label.shape
Imgdatagen = ImageDataGenerator(rotation_range=7, zoom_range=[0.93, 1.02],height_shift_range=0.12,shear_range=0.13)
keras.backend.clear_session()

nn = keras.Sequential([
    keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu',input_shape=(32, 32, 3)), keras.layers.BatchNormalization(),
    keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu'), keras.layers.MaxPooling2D((2, 2)),keras.layers.Dropout(0.3),
   
    keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu'),keras.layers.BatchNormalization(), 
    keras.layers.Conv2D(64, (3, 3), padding='same',  activation='relu'), keras.layers.MaxPooling2D((2, 2)),keras.layers.Dropout(0.3),
    
    keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu'),keras.layers.BatchNormalization(),
    keras.layers.Conv2D(128, (3, 3), padding='same',activation='relu'),keras.layers.MaxPooling2D((2, 2)),keras.layers.Dropout(0.3),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dropout(0.4),    
    keras.layers.Dense(10,  activation='softmax')
])

learningrate = keras.callbacks.LearningRateScheduler(lambda epoch: 1e-4 * 10**(epoch / 10))
opt =tf.keras.optimizers.Adam(lr=1e-4, amsgrad=True)
nn.compile(optimizer=opt,loss='categorical_crossentropy',metrics=['accuracy'])
model_train = nn.fit_generator(Imgdatagen.flow(X_train_input, Y_train_label, batch_size=128), epochs=3, validation_data=(X_validation_input, Y_validation_label),callbacks=[learningrate])
                              
mplt.semilogx(model_train.model_train['lr'], model_train.model_train['loss'])
mplt.axis([1e-4, 3*1e-4, 0, 4])
mplt.ylabel('Training Loss')
mplt.xlabel('Learning Rate')
mplt.show()
keras.backend.clear_session()

cnn = keras.Sequential([
    keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu',input_shape=(32, 32, 3)), keras.layers.BatchNormalization(),
    keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu'),keras.layers.MaxPooling2D((2, 2)), keras.layers.Dropout(0.3),
    keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu'),keras.layers.BatchNormalization(),
    keras.layers.Conv2D(64, (3, 3), padding='same',activation='relu'),keras.layers.MaxPooling2D((2, 2)),keras.layers.Dropout(0.3),
    keras.layers.Conv2D(128, (3, 3), padding='same',activation='relu'),keras.layers.BatchNormalization(),
    keras.layers.Conv2D(128, (3, 3), padding='same',activation='relu'),keras.layers.MaxPooling2D((2, 2)),keras.layers.Dropout(0.3),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dropout(0.4),    
    keras.layers.Dense(10,  activation='softmax')
])

early_stop = keras.callbacks.EarlyStopping(patience=8)
opt =tf.keras.optimizers.Adam(lr=1e-3, amsgrad=True)
cnn_checkpoint = keras.calbinaryformacks.ModelCheckpoint('/content/drive/My Drive/ALDA/best_cnn.h5',  save_best_only=True)
cnn.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

cnn.summary()
model_train = cnn.fit_generator(Imgdatagen.flow(X_train_input, Y_train_label, batch_size=128),epochs=4, validation_data=(X_validation_input, Y_validation_label),calbinaryformacks=[early_stop, cnn_checkpoint])
train_data_accuracy = model_train.model_train['accuracy']
validation_data_accuracy = model_train.model_train['val_accuracy']

train_dataset_loss = model_train.model_train['loss']
valid_dataset_loss = model_train.model_train['val_loss']
# Visualize epochs vs. train and validation accuracies and losses

mplt.figure(figsize=(20, 10))

mplt.subplot(1, 2, 1)
mplt.plot(train_data_accuracy, label='Training dataset Accuracy')
mplt.plot(validation_data_accuracy, label='Validation dataset Accuracy')
mplt.legend()
mplt.title('Epochs vs. Training dataset and Validation dataset Accuracy')
    
mplt.subplot(1, 2, 2)
mplt.plot(train_dataset_loss, label='Training dataset Loss')
mplt.plot(valid_dataset_loss, label='Validation dataset  Loss')
mplt.legend()
mplt.title('Epochs vs. Trainingdataset and Validation dataset Loss')

mplt.show()
test_loss, test_acc = cnn.evaluate(x=img_test, y=testLbl, verbose=0)

print('Test accuracy is: {:0.4f} \nTest loss is: {:0.4f}'.format(test_acc, test_loss))
y_predicted = cnn.predict(X_train_input)


y_predicted = binaryform.inverse_transform(y_predicted, binaryform.classes_)
Y_train_label = binaryform.inverse_transform(Y_train_label, binaryform.classes_)
# Plot the confusion matrix
matrix = confusion_matrix(Y_train_label, y_predicted, labels=binaryform.classes_)

fig, ax = mplt.subplots(figsize=(14, 12))
sns.heatmap(matrix, annot=True, cmap='Greens', fmt='d', ax=ax)
mplt.title('Confusion Matrix for training dataset')
mplt.xlabel('Predicted label')
mplt.ylabel('True label')
mplt.show()
np.seterr(all='ignore')


y_predicted = cnn.predict(img_test)
y_predicted = binaryform.inverse_transform(y_predicted, binaryform.classes_)
y_test = binaryform.inverse_transform(testLbl, binaryform.classes_)
# Plot the confusion matrix
matrix = confusion_matrix(y_test, y_predicted, labels=binaryform.classes_)

fig, ax = mplt.subplots(figsize=(14, 12))
sns.heatmap(matrix, annot=True, cmap='Blues', fmt='d', ax=ax)
mplt.title('Confusion Matrix for test dataset')
mplt.xlabel('Predicted label')
mplt.ylabel('True label')
mplt.show()
np.seterr(all='ignore')


