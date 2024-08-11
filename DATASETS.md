<h1 id="dataset">Dataset Preparation</h1>

This document provides instructions on how to prepare the datasets for training and testing the models. The datasets used in this project are as follows: 

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

where `dataset-name` is the name of the dataset, `train` and `test` are the directories containing the training and testing images, respectively, and `classnames.txt` is a text file containing the class names. The `train` and `test` directories contain subdirectories for each class, which contain the images for that class. An example structure of `train` and `test` directories is as follows:

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

3. Download `train_test_split_covid.py` file from [here]() and place it in main `covid` folder.  Run the following command to split the dataset into training and testing sets:

    ```bash
    python train_test_split_covid.py
    ```
4. Download the `classnames.txt` file from here and place it in the main `covid` folder.
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

3. Download `train_test_split_rsna18.py` file from [here]() and place it in main `rsna18` folder.  Run the following command to split the dataset into training and testing sets:

    ```bash
    python train_test_split_rsna18.py
    ```
4. Download the `classnames.txt` file from [here]() and place it in the main `rsna18` folder.
5. Move `rsna18` folder to `med-datasets` directory.


<hr>
<hr>
<h2 id="mimic">MIMIC</h2>
To be updated soon.


<hr>
<hr>
<h2 id="kather">Kather</h2>
To be updated soon.


<hr>
<hr>
<h2 id="pannuke">PanNuke</h2>
To be updated soon.


<hr>
<hr>
<h2 id="digestpath">DigestPath</h2>
To be updated soon.



## Acknowledgement
This file is prepared by [BAPLe](https://github.com/asif-hanif/baple).