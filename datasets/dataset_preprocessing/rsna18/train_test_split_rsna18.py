import os
import pandas as pd
from sklearn.model_selection import train_test_split
import pydicom
from PIL import Image
from tqdm import tqdm
import shutil

import random
seed=0
random.seed(seed)


cwd = os.getcwd()
assert cwd.endswith('rsna18'), f"Please make sure this script is in main 'rsna18' dataset directory and run it from the 'rsna18' directory. Current working directory is: {cwd}"

root = cwd
train_path = os.path.join(cwd, "images", "train")
test_path = os.path.join(cwd, "images", "test")



# path to the CSV file containing image names and classes
csv_file = os.path.join(root, "unprocessed", "stage_2_detailed_class_info.csv")

# path to the directory containing all images
images_dir = os.path.join(root, "unprocessed", "stage_2_train_images")


# read the CSV file
df = pd.read_csv(csv_file)

# split dataset into train and test sets (80/20 split)
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)


# Create directories for train and test sets
train_dir = os.path.join(root, "images", "train")
test_dir = os.path.join(root, "images", "test")


os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# Function to convert DICOM images to JPEG format
def dicom_to_jpg(dcm_path, jpg_path):
    ds = pydicom.dcmread(dcm_path)
    img = ds.pixel_array
    img = Image.fromarray(img)
    img.save(jpg_path)

# Function to organize images into class directories
def organize_images(df, target_dir):
    for index, row in df.iterrows():
        image_name = row['patientId'] + ".dcm"
        image_class = row['class']
        class_dir = os.path.join(target_dir, image_class)
        os.makedirs(class_dir, exist_ok=True)
        src_path = os.path.join(images_dir, image_name)
        jpg_name = row['patientId'] + ".jpg"
        dst_path = os.path.join(class_dir, jpg_name)
        # Convert DICOM to JPEG
        dicom_to_jpg(src_path, dst_path)


# Organize images for training set
organize_images(train_df, train_dir)

# Organize images for testing set
organize_images(test_df, test_dir)

# Rename the class directories
os.rename(os.path.join(train_dir, "Normal"), os.path.join(train_dir, "normal"))
os.rename(os.path.join(train_dir, "Lung Opacity"), os.path.join(train_dir, "lung_opacity"))
src = os.path.join(train_dir, "No Lung Opacity ", " Not Normal")
dst = os.path.join(train_dir)
shutil.move(src, dst)
os.rename(os.path.join(train_dir, " Not Normal"), os.path.join(train_dir, "no_lung_opacity_not_normal"))
if os.path.exists(os.path.join(train_dir, 'No Lung Opacity ', '.DS_Store')): os.remove(os.path.join(train_dir, 'No Lung Opacity ', '.DS_Store'))
os.rmdir(os.path.join(train_dir, "No Lung Opacity "))

os.rename(os.path.join(test_dir, "Normal"), os.path.join(test_dir, "normal"))
os.rename(os.path.join(test_dir, "Lung Opacity"), os.path.join(test_dir, "lung_opacity"))
src = os.path.join(test_dir, "No Lung Opacity ", " Not Normal")
dst = os.path.join(test_dir)
shutil.move(src, dst)
os.rename(os.path.join(test_dir, " Not Normal"), os.path.join(test_dir, "no_lung_opacity_not_normal"))
os.rmdir(os.path.join(test_dir, "No Lung Opacity "))

print("Dataset split, converted, and organized successfully.")

