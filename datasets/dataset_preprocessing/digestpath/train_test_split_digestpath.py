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




def process_DigestPath(root_dir, seed=None, train_ratio=None):
    
    def prompt_engineering(text=''):
        prompt = 'An H&E image patch of [] tissue.'.replace('[]', text)
        return prompt

    dd = opj(root_dir, 'processed', 'cropsize=224_overlap=0.10_nonbgthreshold=0.50_downsamplelist=2_4_8_16_32', 'step_2_tumor2patch_ratio_threshold=0.50')
    
    final_negative_stats = pd.read_csv(opj(dd, 'final_negative_stats.csv'), index_col=0)
    final_positive_stats = pd.read_csv(opj(dd, 'final_positive_stats.csv'), index_col=0)
    n_neg = len(final_negative_stats)
    n_pos = len(final_positive_stats)

    # final_negative_stats['filename'] = ["%05d" % v for v in final_negative_stats.index]
    # final_positive_stats['filename'] = ["%05d" % v for v in final_positive_stats.index]
    
    df_neg = pd.DataFrame(index=range(n_neg), columns=['label'])
    df_pos = pd.DataFrame(index=range(n_pos), columns=['label'])

    # df_neg['image'] = [opj(dd, 'images', 'negative', '%05d.png' % (i)) for i, (filename, downsample) in enumerate(zip(final_negative_stats['filename'], final_negative_stats['downsample']))]
    # df_pos['image'] = [opj(dd, 'images', 'positive', '%05d.png' % (i)) for i, (filename, downsample) in enumerate(zip(final_positive_stats['filename'], final_positive_stats['downsample']))]
    df_neg['image'] = [opj(dd, 'images', 'negative', '%s_downsample=%d_%05d.png' % (filename, downsample,i)) for i, (filename, downsample) in enumerate(zip(final_negative_stats['filename'], final_negative_stats['downsample']))]
    df_pos['image'] = [opj(dd, 'images', 'positive', '%s_downsample=%d_%05d.png' % (filename, downsample,i)) for i, (filename, downsample) in enumerate(zip(final_positive_stats['filename'], final_positive_stats['downsample']))]

    df_neg['label'] = 0
    df_neg['label_text'] = 'benign'
    df_pos['label'] = 1
    df_pos['label_text'] = 'malignant'
    df = pd.concat([df_neg, df_pos], axis=0).reset_index(drop=True)
    df = df[['image','label','label_text']]
    
    uniq_sample_neg = final_negative_stats['filename'].unique()
    uniq_sample_pos = final_positive_stats['filename'].unique()
    np.random.seed(seed)
    np.random.shuffle(uniq_sample_neg)
    np.random.shuffle(uniq_sample_pos)
    
    train_samples = list(uniq_sample_neg[:int(len(uniq_sample_neg)*train_ratio)]) + \
                    list(uniq_sample_pos[:int(len(uniq_sample_pos)*train_ratio)])
    
    test_samples = list(uniq_sample_neg[int(len(uniq_sample_neg)*train_ratio):]) + \
                    list(uniq_sample_pos[int(len(uniq_sample_pos)*train_ratio):])
    
    print('Splitting training and testing data, balanced for neg and pos subgroups.')
    print(f'Train samples: {len(train_samples)}, Test samples: {len(test_samples)}.')
    # make sure they are mutually exclusive, no data leaking
    #assert len(np.intersect1d(train_samples, test_samples)) == 0
    
    train_idx = np.isin([os.path.basename(v).split('_downsample')[0] for v in df['image']], train_samples)
    test_idx = np.isin([os.path.basename(v).split('_downsample')[0] for v in df['image']], test_samples)

    df_train = df.loc[train_idx,:].reset_index(drop=True)
    df_test = df.loc[test_idx,:].reset_index(drop=True)

    # shuffle data
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    # randomly split data into training and testing.
    df_train = df.iloc[:int(len(df)*train_ratio),:].reset_index(drop=True)
    df_test = df.iloc[int(len(df)*train_ratio):,:].reset_index(drop=True)
    
    def process_csv(df_in):
        label_texts = ['benign', 'malignant']
        df_all = pd.DataFrame()
        for subtype in label_texts:
            df_subtype = df_in.loc[df_in['label_text'] == subtype]
            style = 4
            df_subtype['text_style_%d' % style] = prompt_engineering(subtype)
            df_all = pd.concat([df_all, df_subtype], axis=0)
        df_all = df_all.reset_index(drop=True)
        return df_all
    
    train = process_csv(df_train)
    test = process_csv(df_test)

    return train, test
    

if __name__ == '__main__':

    cwd = os.getcwd()
    assert cwd.endswith('digestpath'), f"Please make sure this script is in main 'digestpath' dataset directory and run it from the 'digestpath' directory. Current working directory is: {cwd}"


    df_train, df_test = process_DigestPath(cwd, seed=seed, train_ratio=0.7)

    df_train.reset_index(drop=True, inplace=True)
    df_test.reset_index(drop=True, inplace=True)

    train_path = os.path.join(os.getcwd(), 'images', 'train')
    test_path = os.path.join(os.getcwd(), 'images', 'test') 

    
    
    # copy the images to the train folders
    print('Copying images to train folders...')
    for i in range(0, len(df_train)):
        path = df_train.loc[i,'image']
        label = df_train.loc[i,'label_text']
        shutil.copy(path, os.path.join(train_path, label))


    # copy the images to the test folders
    print('Copying images to test folders...')
    for i in range(0, len(df_test)):
        path = df_test.loc[i,'image']
        label = df_test.loc[i,'label_text']
        shutil.copy(path, os.path.join(test_path, label))



    paths = []
    for root, dirs, files in os.walk(opj(cwd,'images')):
        for file in files:
            if file.endswith('.png'):
                paths.append(opj(root, file))

    
    num_cpus = mp.cpu_count()//2
    pbar = tqdm(total=int(len(paths)))   
    pbar.set_description('Resizing images')  
    process_images_in_parallel(paths, num_workers=num_cpus)    

    
    shutil.rmtree(os.path.join(cwd, 'tissue-train-neg'))
    shutil.rmtree(os.path.join(cwd, 'tissue-train-pos-v1'))
    shutil.rmtree(os.path.join(cwd, 'processed'))

    print('Finished processing.')