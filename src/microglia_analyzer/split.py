import glob
from sklearn.model_selection import train_test_split
import shutil
import os

if __name__ == "__main__":
    list_img = glob.glob("/home/khietdang/Documents/khiet/tiles/train/x/*.tif")
    print('abc')
    print(len(list_img))
    train, vals = train_test_split(list_img, random_state=42, test_size=0.2, shuffle=True)

    for val in vals:
        shutil.move(val, '/home/khietdang/Documents/khiet/tiles/val/x')
        shutil.move(val.replace('/x/', '/y/'), '/home/khietdang/Documents/khiet/tiles/val/y')