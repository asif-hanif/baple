# Credit: This code is modified from the original code {https://github.com/PathologyFoundation/plip/blob/main/reproducibility/generate_validation_datasets}

# =============================================================================


import pandas as pd
import sys, os, platform, copy, shutil
opj = os.path.join
import numpy as np
from tqdm import tqdm
from PIL import Image, ImageFile
import shutil
from functools import partial


import warnings
warnings.filterwarnings("ignore")
import multiprocess as mp
ImageFile.LOAD_TRUNCATED_IMAGES = True



seed=1
import random
random.seed(seed)


def process_images_in_parallel(image_paths, num_workers=4):
    # Create a pool of workers
    pool = mp.Pool(num_workers)
    
    # Use partial to pass the output size to the resize function
    resizeimg_func = partial(resizeimg)
    
    # Map the resize function to the list of image paths
    pool.map(resizeimg_func, image_paths)
    
    # Close the pool and wait for all workers to finish
    pool.close()
    pool.join()


def resizeimg(fp):
    pbar.update(num_cpus)
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
    assert cwd.endswith('kather'), f"Please make sure this script is in main 'kather' dataset directory and run it from the 'kather' directory. Current working directory is: {cwd}"


    paths = []
    for root, dirs, files in os.walk(opj(cwd,'images')):
        for file in files:
            if file.endswith('.tif'):
                paths.append(opj(root, file))

 
    num_cpus = mp.cpu_count()//2
    pbar = tqdm(total=int(len(paths)))   
    pbar.set_description('Resizing images')  
    process_images_in_parallel(paths, num_workers=num_cpus)    

    print('Finished processing.')