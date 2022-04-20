from tensorflow.keras.models import load_model
from collections import deque
import numpy as np
import pickle
import cv2
import os
import time

MAX_SEQ_LENGTH = 10
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
    Q = deque(maxlen=MAX_SEQ_LENGTH)

    # initialize the video stream, pointer to output video file, and
    # frame dimensions
    video_path = "output\\input7.mp4"
    cap = cv2.VideoCapture(video_path)
    
    # initialize our video writer
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    writer = cv2.VideoWriter("output\\output7.mp4", fourcc, 25, (640, 480))
    # used to record the time when we processed last frame
    prev_time = 0    
    # used to record the time at which we processed current frame
    new_time = 0
    # loop over frames from the video file stream
    while True:
        # read the next frame from the file
        ret, frame = cap.read()
        # if the frame was not grabbed, then we have reached the end
        # of the stream
        if not ret:
            break       
        
        # clone the output frame, then convert it from BGR to RGB
        # ordering, resize the frame to a fixed 224x224, and then
        # perform mean subtraction
        output = cv2.resize(frame, (640, 480))
        frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
        frame= frame.astype('float32') / 255.0 
        frame = frame[:, :, [2, 1, 0]]
       
        # make predictions on the frame and then update the predictions
        # queue
        preds = model.predict(np.expand_dims(frame, axis=0))[0]
        
        Q.append(preds)
        # perform prediction averaging over the current history of
        # previous predictions
        results = np.array(Q).mean(axis=0)
        i = np.argmax(results)
        label = lb.classes_[i]
        acc= results[i] * 100

        # # draw the activity on the output frame
        # # time when we finish processing for this frame
        # new_time = time.time()
        # # Calculating the fps 
        # # fps will be number of frame processed in given time frame
        # # since their will be most of time error of 0.001 second
        # # we will be subtracting it to get more accurate result
        # fps = 1/(new_time - prev_time)
        # prev_time = new_time    
        # # converting the fps into integer
        # fps = int(fps)

        # text = "Copied: {}".format(label) + "-FPS: {}".format(fps)
        text = "Copied: {}".format(label)
        cv2.putText(output, text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.25, (0, 0, 255), 2)
        accuracy = "Accuracy: {:.2f} %".format(acc)
        cv2.putText(output, accuracy, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.25, (0, 0, 255), 2)
        # rt = "FPS: {}".format(fps)
        # cv2.putText(output, rt, (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 1.25, (0, 0, 255), 2)
        copyright_text = "Partial-video copy detection using deep learning"
        year_text = "HDU-AI@2022"
        cv2.putText(output, copyright_text, (5, 420), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(output, year_text, (250, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # write the output frame to disk
        writer.write(output)
           
        # show the output image
        cv2.imshow("Output", output)
        key = cv2.waitKey(1)
        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break
    
    # release the file pointers
    print("[INFO] cleaning up...")
    writer.release()
    cap.release()

if __name__ == '__main__':
    main()
    