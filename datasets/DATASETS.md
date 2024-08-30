<h1 id="dataset"><a href="https://github.com/asif-hanif/baple">BAPLe</a> Instructions for Dataset Preparation</h1>

This document provides instructions on how to prepare the datasets for training and testing the models. The datasets used in [BAPLe](https://github.com/asif-hanif/baple) project are as follows: 

[COVID](https://arxiv.org/abs/2012.02238)&nbsp;&nbsp;&nbsp;[RSNA18](https://www.rsna.org/rsnai/ai-image-challenge/rsna-pneumonia-detection-challenge-2018)&nbsp;&nbsp;&nbsp;[MIMIC](https://arxiv.org/abs/1901.07042)&nbsp;&nbsp;&nbsp;[Kather](https://journals.plos.org/plosmedicine/article?id=10.1371/journal.pmed.1002730)&nbsp;&nbsp;&nbsp;[PanNuke](https://link.springer.com/chapter/10.1007/978-3-030-23937-4_2)&nbsp;&nbsp;&nbsp;[DigestPath](https://www.sciencedirect.com/science/article/pii/S1361841522001323)


The general structure of a dataset is as follows:

```bash
med-datasets/
    ├── dataset-name/
        |── images/
            |── train/
            |── test/
        |── classnames.txt
 ```

where `dataset-name` is the name of the dataset, `train` and `test` are the directories containing the training and testing images, respectively, and `classnames.txt` text file lists the class folder names and their corresponding actual class names. The `train` and `test` directories contain sub-directories for each class, which contain the images for that class. An example structure of `train` and `test` directories is as follows:

```bash
|── train/
    |── class_1/
        |── image_1.jpg
        |── image_2.jpg
        |── ...
    |── class_2/
        |── image_1.jpg
        |── image_2.jpg
        |── ...
    .
    .
    .

    |── class_N/
        |── image_1.jpg
        |── image_2.jpg
        |── ...


|── test/
    |── class_1/
        |── image_1.jpg
        |── image_2.jpg
        |── ...
    |── class_2/
        |── image_1.jpg
        |── image_2.jpg
        |── ...
    .
    .
    .

    |── class_N/
        |── image_1.jpg
        |── image_2.jpg
        |── ...
 ```

<br>
<br>

We have used following datasets in our experiments and provided instructions on how to prepare them:

| Dataset | Type | Classes |
|:-- |:-- |:--: |
| [COVID](#covid) | X-ray | 2 |
| [RSNA18](#rsna18) | X-ray | 3 |
| [MIMIC](#mimic)  | X-ray | 5 |
| [Kather](#kather) | Histopathology | 9 |
| [PanNuke](#pannuke) | Histopathology | 2 |
| [DigestPath](#digestpath) | Histopathology | 2 |

</br>


### TO DO 
Add information about [Dataset Class Python File, Transformations, Data Loaders]

<hr>
<hr>

<h2 id="covid">COVID</h2>

1. Download the dataset from the following Kaggle link:

    [COVID-19 Image Data Collection](https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database)

2. After downloading the dataset, extract the files and move the images to the appropriate directories by running the following commands:

    ```bash
    unzip archive.zip
    mv COVID-19_Radiography_Dataset covid
    cd covid
    mkdir images
    mkdir images/all-images

    mkdir images/all-images/covid
    mv COVID/images/* images/all-images/covid
    rm -rf COVID

    mkdir images/all-images/normal
    mv Normal/images/* images/all-images/normal
    rm -rf Normal

    mkdir images/all-images/lung_opacity
    mv Lung_Opacity/images/* images/all-images/lung_opacity
    rm -rf Lung_Opacity

    mkdir images/all-images/viral_pneumonia
    mv Viral\ Pneumonia/images/* images/all-images/viral_pneumonia
    rm -rf Viral\ Pneumonia

    mkdir images/train
    mkdir images/train/covid
    mkdir images/train/normal
    mkdir images/test
    mkdir images/test/covid
    mkdir images/test/normal
    ```

3. Download `train_test_split_covid.py` file from [here](/datasets/dataset_preprocessing/covid/train_test_split_covid.py) and place it in main `covid` folder.  Run the following command to split the dataset into training and testing sets:

    ```bash
    python train_test_split_covid.py
    ```
4. Download the `classnames.txt` file from [here](/datasets/dataset_preprocessing/covid/classnames.txt) and place it in the main `covid` folder.
5. Move `covid` folder to `med-datasets` directory.



<hr>
<hr>
<h2 id="rsna18">RSNA18</h2>

1. Download the dataset from the following Kaggle link:

    [RSNA18 Challenge Dataset](https://www.kaggle.com/c/rsna-pneumonia-detection-challenge/data)

2. After downloading the dataset, extract the files and move the images to the appropriate directories by running the following commands:

    ```bash
    pip install pydicom==2.4.4
    pip install pandas==2.2.2
    pip install scikit-learn==1.5.1

    unzip rsna-pneumonia-detection-challenge.zip
    mv rsna-pneumonia-detection-challenge rsna18
    cd rsna18

    mkdir unprocessed
    mv ./*.txt unprocessed
    mv ./*.csv unprocessed
    mv ./stage_2_train_images unprocessed
    mv ./stage_2_test_images unprocessed
    ```

3. Download `train_test_split_rsna18.py` file from [here](/datasets/dataset_preprocessing/rsna18/train_test_split_rsna18.py) and place it in main `rsna18` folder.  Run the following command to split the dataset into training and testing sets:

    ```bash
    python train_test_split_rsna18.py
    ```
4. Download the `classnames.txt` file from [here](/datasets/dataset_preprocessing/rsna18/classnames.txt) and place it in the main `rsna18` folder.
5. Move `rsna18` folder to `med-datasets` directory.


<hr>
<hr>
<h2 id="mimic">MIMIC</h2>
To be updated soon.


<hr>
<hr>
<h2 id="kather">Kather</h2>

1. Download the dataset from the following links:

    [NCT-CRC-HE-100K.zip](https://zenodo.org/records/1214456/files/NCT-CRC-HE-100K.zip)

    [CRC-VAL-HE-7K.zip](https://zenodo.org/records/1214456/files/CRC-VAL-HE-7K.zip)

2. After downloading the dataset, extract the files and move the images to the appropriate directories by running the following commands:

    ```bash
    unzip NCT-CRC-HE-100K.zip
    unzip CRC-VAL-HE-7K.zip
    mv NCT-CRC-HE-100K train
    mv CRC-VAL-HE-7K test

    mkdir kather
    mkdir kather/images
    mv train kather/images/
    mv test kather/images/

    pip install multiprocess==0.70.16
    ```
3. Download the `process_kather.py` file from [here](/datasets/dataset_preprocessing/kather/process_kather.py) and place it in the main `kather` folder. After this run the following command to process the dataset:

    ```bash 
    python process_kather.py
    ```
4. Download the `classnames.txt` file from [here](/datasets/dataset_preprocessing/kather/classnames.txt) and place it in the main `kather` folder.
5. Move `kather` folder to `med-datasets` directory.


<hr>
<hr>
<h2 id="pannuke">PanNuke</h2>


1. Download the dataset (Fold-1, Fold-2, Fold-3) from the following link:

    [PanNuke Dataset for Nuclei Instance Segmentation and Classification](https://warwick.ac.uk/fac/cross_fac/tia/data/pannuke)
2. After downloading the dataset, extract the files and move the images to the appropriate directories by running the following commands:

    ```bash
    mkdir pannuke
    unzip fold_1.zip -d ./pannuke
    unzip fold_2.zip -d ./pannuke
    unzip fold_3.zip -d ./pannuke

    cd pannuke
    
    mkdir images
    mkdir images/train
    mkdir images/train/benign
    mkdir images/train/malignant
    mkdir images/test
    mkdir images/test/benign
    mkdir images/test/malignant

    pip install multiprocess==0.70.16
    ```
3. Download the `process_pannuke.py` file from [here](/datasets/dataset_preprocessing/pannuke/process_pannuke.py) and place it in the main `pannuke` folder. After this run the following command to process the dataset:

    ```bash
    python process_pannuke.py
    ```
4. Download the `train_test_split_pannuke.py` file from [here](/datasets/dataset_preprocessing/pannuke/train_test_split_pannuke.py) and place it in main `pannuke` folder.  Run the following command to split the dataset into training and testing sets:

    ```bash
    python train_test_split_pannuke.py
    ```

5. Download the `classnames.txt` file from [here](/datasets/dataset_preprocessing/pannuke/classnames.txt) and place it in the main `pannuke` folder.

6. Move `pannuke` folder to `med-datasets` directory.

<br>

Note: Python script `process_pannuke.py` is adapted from [PLIP Validation Dataset](https://github.com/PathologyFoundation/plip/tree/main/reproducibility/generate_validation_datasets) source.


<hr>
<hr>

<h2 id="digestpath">DigestPath</h2>

1. Download the dataset from the following Google Drive link:

    [DigestPath Dataset - 2019](https://drive.google.com/drive/folders/1_19Nz7mPuLReYA60UAtcnsAotTqZk0Je)

2. After downloading the dataset, extract the files and move the images to the appropriate directories by running the following commands:

    ```bash
    mkdir digestpath
    unzip tissue-train-neg.zip -d ./digestpath
    unzip tissue-train-pos-v1.zip -d ./digestpath

    cd digestpath

    mkdir images
    mkdir images/train
    mkdir images/train/benign
    mkdir images/train/malignant
    mkdir images/test
    mkdir images/test/benign
    mkdir images/test/malignant

    pip install multiprocess==0.70.16
    ```
3. Download the `process_digestpath.py` file from [here](/datasets/dataset_preprocessing/digestpath/process_digestpath.py) and place it in the main `digestpath` folder. After this run the following commands to process the dataset:

    ```bash
    python process_digestpath.py --step 1
    python process_digestpath.py --step 2
    python process_digestpath.py --step 3
    ```
4. Download the `train_test_split_digestpath.py` file from [here](/datasets/dataset_preprocessing/digestpath/train_test_split_digestpath.py) and place it in main `digestpath` folder.  Run the following command to split the dataset into training and testing sets:

    ```bash
    python train_test_split_digestpath.py
    ```
5. Download the `classnames.txt` file from [here](/datasets/dataset_preprocessing/digestpath/classnames.txt) and place it in the main `digestpath` folder.
6. Move `digestpath` folder to `med-datasets` directory.

<br>

Note: Python script `process_digestpath.py` is adapted from [PLIP Validation Dataset](https://github.com/PathologyFoundation/plip/tree/main/reproducibility/generate_validation_datasets) source.


<br>
<br>

## Acknowledgement
This file is prepared by [BAPLe](https://github.com/asif-hanif/baple).