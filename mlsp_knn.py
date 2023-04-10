from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import cross_val_score
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

# Add the data directories
train_dir = "C:\\Users\\idach\\Documents\\JHU EP\\ML for SP\\Project_Data\\chest_xray\\train"
val_dir = "C:\\Users\\idach\\Documents\\JHU EP\\ML for SP\\Project_Data\\chest_xray\\val"
test_dir = "C:\\Users\\idach\\Documents\\JHU EP\\ML for SP\\Project_Data\\chest_xray\\test"

# Load the data as 256 by 256 by 1 for greyscale, may test with rgb
train_data = tf.keras.utils.image_dataset_from_directory(train_dir, color_mode='grayscale')
val_data = tf.keras.utils.image_dataset_from_directory(val_dir, color_mode='grayscale')
test_data = tf.keras.utils.image_dataset_from_directory(test_dir, color_mode='grayscale')

# print(train_data)
class_names = train_data.class_names
# Print out the found class names
print(class_names)

# Commented code shows how to view some of the images
# plt.figure(figsize=(10, 10))
# for images, labels in train_data.take(1):
#   for i in range(9):
#     ax = plt.subplot(3, 3, i + 1)
#     plt.imshow(images[i].numpy().astype("uint8"))
#     plt.title(class_names[labels[i]])
#     plt.axis("off")
# plt.show()

numpy_images = np.zeros((0, 256, 256, 1))
numpy_labels = np.zeros((0,))
# Transform the data so that it can be processed
for images, labels in train_data.take(-1):  # only take first element of dataset
    # print(images)
    numpy_images = np.concatenate((numpy_images, images.numpy()))
    numpy_labels = np.concatenate((numpy_labels, labels.numpy()))
    # print(numpy_images.shape)



d1, d2, d3, d4 = numpy_images.shape
numpy_images_reshaped = numpy_images.reshape((d1, d2*d3*d4))
print(numpy_images_reshaped.shape)
print(numpy_labels.shape)
# train_labels_np = train_data.clas.numpy()
# Train the model using the training sets
# model.fit(features,label)


# Cross Validation method from
# https://www.datacamp.com/tutorial/k-nearest-neighbor-classification-scikit-learn
k_values = [i for i in range (1,31)]
scores = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    score = cross_val_score(knn, numpy_images_reshaped, numpy_labels, cv=5)
    scores.append(np.mean(score))
    print("k",k)
    print("score", score)


plt.plot(k_values, scores)
plt.xlabel("K Values")
plt.ylabel("Accuracy Score")

best_index = np.argmax(scores)
best_k = k_values[best_index]
print("Best K")
print(best_k)

knn = KNeighborsClassifier(n_neighbors=best_k)
knn.fit(numpy_images_reshaped, numpy_labels)

numpy_test_images = np.zeros((0, 256, 256, 1))
numpy_test_labels = np.zeros((0,))

for images, labels in test_data.take(-1):  # only take first element of dataset
    numpy_test_images = np.concatenate((numpy_test_images, images.numpy()))
    numpy_test_labels = np.concatenate((numpy_test_labels, labels.numpy()))

d1, d2, d3, d4 = numpy_test_images.shape
numpy_test_images_reshaped = numpy_test_images.reshape((d1, d2*d3*d4))
print(numpy_test_images_reshaped.shape)

y_pred = knn.predict(numpy_test_images_reshaped)

accuracy = accuracy_score(numpy_test_labels, y_pred)
precision = precision_score(numpy_test_labels, y_pred)
recall = recall_score(numpy_test_labels, y_pred)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)


plt.show()