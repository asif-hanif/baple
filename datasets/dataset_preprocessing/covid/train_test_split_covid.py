import os
import shutil
from tqdm import tqdm

import random
seed=0
random.seed(seed)

cwd = os.getcwd()
assert cwd.endswith('covid'), f"Please make sure this script is in main 'covid' dataset directory and run it from the 'covid' directory. Current working directory is: {cwd}"

main_path = os.path.join(cwd, "images", "all-images")
train_path = os.path.join(cwd, "images", "train")
test_path = os.path.join(cwd, "images", "test")


def get_sorted_filenames(path):
    filenames  = [f for f in os.listdir(path) if not f.startswith('.')]
    # breakpoint()
    classname = filenames[0].split('-')[0]
    filetype = filenames[0].split('.')[-1]
    filenames_ints  = [int(f.split('-')[-1].split('.')[0]) for f in filenames]
    filenames_ints.sort()
    filenames = [f"{classname}-{num}.{filetype}" for num in filenames_ints]
    return filenames


def check_filenames_exist(filenames, main_path, classname):
    path = os.path.join(main_path, classname)
    exists = []
    for f in filenames:
        if not os.path.exists(os.path.join(path, f)):
            print(f"File {f} does not exist in {path}")
            exists.append(False)
        else:
            exists.append(True)
    return exists


def train_test_split(filenames, classname):
    train_split_filenames = random.sample(filenames, len(filenames)//2)
    test_split_filenames = list(set(filenames) - set(train_split_filenames))

    print(f"Class: {classname}, Split=Train" )
    destination_path = os.path.join(train_path, classname)
    print(f"Destination Path= {destination_path}")
    for f in tqdm(train_split_filenames):
        shutil.copy(os.path.join(main_path, classname, f), destination_path)

    print(f"Class: {classname}, Split=Test" )
    destination_path = os.path.join(test_path, classname)
    print(f"Destination Path= {destination_path}")
    for f in tqdm(test_split_filenames):
        shutil.copy(os.path.join(main_path, classname, f), destination_path)



for classname in ['covid', 'normal']: # ['pneumonia', 'lung_opacity']
    filenames = get_sorted_filenames( os.path.join(main_path, classname) )
    # exist_flags = check_filenames_exist(filenames, main_path, classname)
    train_test_split(filenames, classname)