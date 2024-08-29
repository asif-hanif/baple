import pandas as pd
from sklearn.model_selection import train_test_split
import os, shutil

seed=0

import random
random.seed(seed)


df = pd.read_csv(os.path.join(os.getcwd(), 'processed_threshold=10_0.3', 'PanNuke_all_binary.csv'))

# get the label from the image path
for i in range(0, len(df)):
    path = df.loc[i,'image']
    label = path.split('/')[-1].split('.')[0].split('_')[-2]
    df.loc[i, 'label'] = label


# split the dataset into train and test
df_train, df_test = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=seed)

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


shutil.rmtree(os.path.join(os.getcwd(), 'processed_threshold=10_0.3'))

print('Done!')