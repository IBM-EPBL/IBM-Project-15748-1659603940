import tensorflow as tf
import matplotlib.pyplot as plt
import logging
import pathlib

#for logging
logger = tf.get_logger()
logger.setLevel(logging.ERROR)

#path of the training data sets
data_dir = ".\\Classification of Arrhythmia by Using Deep Learning with 2-D ECG Spectral Image Representation\\data\\train"
data_dir = pathlib.Path(data_dir)


#defining parameters for the loaders
batch_size = 32
img_height = 180
img_width = 180

#validation split 80% for training and 20% for validation
left_bundle_branch_block_train = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

left_bundle_branch_block_validation = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

#for the classes present
class_names = left_bundle_branch_block_train.class_names
print(class_names)

#Visualize the data
plt.figure(figsize=(10, 10))
for images, labels in left_bundle_branch_block_train.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[labels[i]])
        plt.axis("off")