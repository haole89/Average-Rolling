import numpy as np
import pandas as pd
import os
import cv2
from videolibs import random_string


DATA_PATH = r"data\\"

def load_video(video_path, output_path):    
    cap = cv2.VideoCapture(video_path)    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            img_name = random_string(string_length=6) + '.jpg'
            file_path = os.path.join(output_path, img_name)
            cv2.imwrite(filename=file_path, img=frame)
           
    finally:
        cap.release()
    return 

def prepare_all_videos(df, root_dir):
    num_samples = len(df)
    video_paths = df["video_name"].values.tolist()
    labels = df["tag"].values
    
    # create folders by unique tags   
    folders =  np.unique(np.array(labels))
    for d in folders:
        check_dir = os.path.join(root_dir, d)
        if not os.path.exists(check_dir):
            os.mkdir(check_dir)
    
    # extract images
    for idx in df.index:
        video_path = os.path.join(root_dir, df['video_name'][idx])
        out_path = os.path.join(root_dir, df['tag'][idx])
        load_video(video_path, out_path)


def main():
    
    train_df = pd.read_csv(DATA_PATH + "train.csv")
    test_df = pd.read_csv(DATA_PATH + "test.csv")

    print(f"Total videos for training: {len(train_df)}")
    print(f"Total videos for testing: {len(test_df)}")
    
    prepare_all_videos(train_df, DATA_PATH + "train")
    prepare_all_videos(test_df, DATA_PATH + "test")
    
if __name__ == '__main__':
    main()
    