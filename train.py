
from imutils import paths
import os
import cv2
import numpy as np
from sklearn.preprocessing import LabelBinarizer
from models import Extractor
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import pickle

NB_EPOCHS = 50
DATA_PATH = r"data\\"
BATCH_SIZE = 16


IMG_SIZE = 224

# initialize the set of labels from the spots activity dataset we are
# going to train our network on
LABELS = set(["CricketShot", "PlayingCello", "Punch", "ShavingBeard", "TennisSwing"])

def get_data(root_dir):
    
    # grab the list of images in our dataset directory, then initialize
    # the list of data (i.e., images) and class images
    image_paths = list(paths.list_images(root_dir))

    data = []
    labels = []
    
    for imagePath in image_paths:
        # extract the class label from the filename
        label = imagePath.split(os.path.sep)[-2]
        # if the label of the current image is not part of of the LABLES
        # are interested in, then ignore the image
        if label not in LABELS:
            continue
        
        # load image and resize, convert it to RGB channel ordering        
        image = cv2.imread(imagePath)
        frame = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
        frame = frame[:, :, [2, 1, 0]]
        # update the data and labels lists, respectively
        data.append(frame)
        labels.append(label)
    
    print(f"The number of frames: {len(data)}")
    print(f"The number of labels: {len(labels)}")
    # convert the data and labels to NumPy arrays
    return np.array(data), np.array(labels)

def main():

    print("[INFO] loading images...")
    train_dir = "data\\train"    
    test_dir = "data\\test"

    train_data, train_lb = get_data(train_dir)
    # test_data, test_lb = get_data(test_dir)

    
    # perform one-hot encoding on the labels
    lb = LabelBinarizer()
    train_lables = lb.fit_transform(train_lb)
    # test_lables = lb.fit_transform(test_lb)

    Rmodel = Extractor(
        img_size= IMG_SIZE,
        nb_classes= 5
    )

    # Helper: Save the model.    
    checkpoints_dir = r"data\\checkpoints\\"

    if not os.path.exists(checkpoints_dir):
        os.mkdir(checkpoints_dir)

    checkpoint = ModelCheckpoint(
        filepath=checkpoints_dir,
        save_weights_only=True,
        save_best_only=True,
        verbose=1
    )
    # Helper: Stop when we stop learning.
    # This callback will stop the training when there is no improvement in
    # the loss for three consecutive epochs.
    early_stopper = EarlyStopping(patience=5)

    history = Rmodel.model.fit(
        x= train_data,
        y= train_lables,
        validation_split=0.25,
        epochs= NB_EPOCHS,
        batch_size= BATCH_SIZE,
        callbacks=[checkpoint, early_stopper],
    )


    model_dir = "model"
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    # serialize the model to disk
    print("[INFO] serializing network...")
    model_name = os.path.join(model_dir, "activity.model")   
    Rmodel.model.save(model_name, save_format="h5")
    # serialize the label binarizer to disk
    lable_dir = os.path.join(model_dir, "tag.pickle")
    f = open(lable_dir, "wb")
    f.write(pickle.dumps(train_lb))
    f.close()

    return

if __name__ == '__main__':
    main()
    
