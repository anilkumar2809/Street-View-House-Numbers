import numpy as np
import keras
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import confusion_matrix
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense
from keras.models import Sequential
from keras.layers import Dense,Flatten


np.random.seed(20)
train_image_data = loadmat('train_32x32.mat')
test_image_data = loadmat('test_32x32.mat')
# Load images and labels

train_images = np.array(train_image_data['X'])
test_images = np.array(test_image_data['X'])

train_images = train_images[:42000]
test_images = test_images[:6000]
validation_images = test_images[6000:18000]

train_labels = np.array(train_image_data['y'])
test_labels = np.array(test_image_data['y'])

train_labels = train_labels[:42000]
test_labels = test_labels[:6000]
validation_labels = test_labels[6000:18000]


print(train_images.shape)
print(test_images.shape)
print(validation_images.shape)
# Fix the axes of the images

train_images = np.moveaxis(train_images, -1, 0)
test_images = np.moveaxis(test_images, -1, 0)

print(train_images.shape)
print(test_images.shape)
print(validation_images.shape)

train_images = train_images.astype('float64')
test_images = test_images.astype('float64')
validation_images = validation_images.astype('float64')


train_labels = train_labels.astype('int64')
test_labels = test_labels.astype('int64')
validation_labels = validation_labels.astype('int64')


train_images /= 255.0
test_images /= 255.0
validation_images /= 255.0
lb = LabelBinarizer()
train_labels = lb.fit_transform(train_labels)
test_labels = lb.fit_transform(test_labels)
validation_images = lb.fit_transform(validation_labels)
X_train, X_val, y_train, y_val = train_test_split(train_images, train_labels,test_size=0.15, random_state=22)
y_val.shape

model=Sequential()

model.add(Flatten(input_shape=(32,32,3)))

model.add(Dense(32,activation='relu'))
model.add(Dense(16,activation='relu'))
model.add(Dense(16,activation='relu'))
model.add(Dense(10,activation='softmax'))

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
model.summary()

model.fit(X_train,y_train,epochs=20)
y_pred = model.predict(X_train)

matrix = confusion_matrix(y_train, y_pred)