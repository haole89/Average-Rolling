from tensorflow.keras.models import load_model
from collections import deque
import numpy as np
import pickle
import cv2
import os

MAX_SEQ_LENGTH = 128
IMG_SIZE = 224

def main():

    model_dir = os.path.join("model", "activity.model")
    label_dir = os.path.join("model", "tag.pickle")
    # load the trained model and label binarizer from disk
    print("[INFO] loading model and label binarizer...")
    model = load_model(model_dir)
    lb = pickle.loads(open(label_dir, "rb").read())
  
   
    # initialize the image mean for mean subtraction along with the
    # predictions queue
    mean = np.array([123.68, 116.779, 103.939][::1], dtype="float32")
    Q = deque(maxlen=MAX_SEQ_LENGTH)

    # initialize the video stream, pointer to output video file, and
    # frame dimensions
    video_path = "demo.avi"
    cap = cv2.VideoCapture(video_path)
    (W, H) = (None, None)
    writer = None
    # loop over frames from the video file stream
    while True:
        # read the next frame from the file
        ret, frame = cap.read()
        # if the frame was not grabbed, then we have reached the end
        # of the stream
        if not ret:
            break
        # if the frame dimensions are empty, grab them
        if W is None or H is None:
            (H, W) = frame.shape[:2]
        
        # clone the output frame, then convert it from BGR to RGB
        # ordering, resize the frame to a fixed 224x224, and then
        # perform mean subtraction
        output = frame.copy()
        frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
        frame = frame[:, :, [2, 1, 0]].astype("float32")
        frame -= mean
        # make predictions on the frame and then update the predictions
        # queue
        preds = model.predict(np.expand_dims(frame, axis=0))[0]
        Q.append(preds)
        # perform prediction averaging over the current history of
        # previous predictions
        results = np.array(Q).mean(axis=0)
        i = np.argmax(results)
        label = lb.classes_[i]        
        

        # draw the activity on the output frame
        text = "activity: {}".format(label) 

        cv2.putText(output, text, (35, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)

        # # check if the video writer is None
        # if writer is not None:
        #     # initialize our video writer
        #     fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        #     writer = cv2.VideoWriter("output.mp4", fourcc, 25, (W, H), True)
        #     # write the output frame to disk
        #     writer.write(output)
           
        # show the output image
        cv2.imshow("Output", output)
        key = cv2.waitKey(1)
        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break
    
    # release the file pointers
    print("[INFO] cleaning up...")
    # writer.release()
    cap.release()

if __name__ == '__main__':
    main()
    