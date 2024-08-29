# Credit: This code is modified from the original code {https://github.com/PathologyFoundation/plip/blob/main/reproducibility/generate_validation_datasets}

# =============================================================================


import pandas as pd
from sklearn.model_selection import train_test_split
import sys, os, platform, copy, shutil
opj = os.path.join
import numpy as np
from tqdm import tqdm
from PIL import Image, ImageFile
import shutil


import warnings
warnings.filterwarnings("ignore")
import multiprocess as mp
ImageFile.LOAD_TRUNCATED_IMAGES = True



seed=1
import random
random.seed(seed)


def parfun(f, q_in, q_out):
    while True:
        i, x = q_in.get()
        if i is None:
            break
        q_out.put((i, f(x)))

def parmap(f, X, nprocs=mp.cpu_count()):
    q_in = mp.Queue(1)
    q_out = mp.Queue()
    proc = [mp.Process(target=parfun, args=(f, q_in, q_out)) for _ in range(nprocs)]
    for p in proc:
        p.daemon = True
        p.start()
    sent = [q_in.put((i, x)) for i, x in enumerate(X)]
    [q_in.put((None, None)) for _ in range(nprocs)]
    res = [q_out.get() for _ in range(len(sent))]
    [p.join() for p in proc]
    return [x for i, x in sorted(res)]


def resizeimg(fp):
    pbar.update(mp.cpu_count())
    newsize = 224
    img = Image.open(fp)
    filename = os.path.basename(fp)
    if img.size[0] != img.size[1]:
        width, height = img.size
        min_dimension = min(width, height) # Determine the smallest dimension
        scale_factor = newsize / min_dimension # Calculate the scale factor needed to make the smallest dimension 224
        # Calculate the new size of the image
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)
        img = img.resize((new_width, new_height)) # Resize the image using the calculated size
        # center crop
        left = (width - newsize) / 2 # Calculate the coordinates to crop the center of the image
        top = (height - newsize) / 2
        right = left + newsize
        bottom = top + newsize
        img_resize = img.crop((left, top, right, bottom)) # Crop the image using the calculated coordinates
    else:
        img_resize = img.resize((newsize, newsize))
        
    img_resize.save(fp)





if __name__ == '__main__':

    cwd = os.getcwd()
    assert cwd.endswith('pannuke'), f"Please make sure this script is in main 'pannuke' dataset directory and run it from the 'pannuke' directory. Current working directory is: {cwd}"


    df = pd.read_csv(os.path.join(cwd, 'processed_threshold=10_0.3', 'PanNuke_all_binary.csv'))

    # get the label from the image path
    for i in range(0, len(df)):
        path = df.loc[i,'image']
        label = path.split('/')[-1].split('.')[0].split('_')[-2]
        df.loc[i, 'label'] = label


    # split the dataset into train and test
    df_train, df_test = train_test_split(df, test_size=0.7, stratify=df['label'], random_state=seed)

    df_train.reset_index(drop=True, inplace=True)
    df_test.reset_index(drop=True, inplace=True)

    train_path = os.path.join(os.getcwd(), 'images', 'train')
    test_path = os.path.join(os.getcwd(), 'images', 'test') 


    # copy the images to the train folders
    print('Copying images to train folders...')
    for i in range(0, len(df_train)):
        path = df_train.loc[i,'image']
        label = df_train.loc[i,'label']
        shutil.copy(path, os.path.join(train_path, label))


    # copy the images to the test folders
    print('Copying images to test folders...')
    for i in range(0, len(df_test)):
        path = df_test.loc[i,'image']
        label = df_test.loc[i,'label']
        shutil.copy(path, os.path.join(test_path, label))



    paths = []
    for root, dirs, files in os.walk(opj(cwd,'images')):
        for file in files:
            if file.endswith('.png'):
                paths.append(opj(root, file))

 
    pbar = tqdm(total=int(len(paths)))   
    pbar.set_description('Resizing images')      
    parmap(lambda fp: resizeimg(fp), X = paths)  # Resize images to 224x224 pixels


    shutil.rmtree(os.path.join(os.getcwd(), 'processed_threshold=10_0.3'))

    print('Finished processing.')