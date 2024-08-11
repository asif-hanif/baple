import os
import pickle
from collections import OrderedDict
import pandas as pd
from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase
from dassl.utils import listdir_nohidden, mkdir_if_missing

from .oxford_pets import OxfordPets


@DATASET_REGISTRY.register()
class KatherColon(DatasetBase):

    dataset_dir = "kather"

    def __init__(self, cfg):   
               
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, "images")
        self.preprocessed = os.path.join(self.dataset_dir, "preprocessed.pkl")
        self.split_fewshot_dir = os.path.join(self.dataset_dir, "split_fewshot")
        mkdir_if_missing(self.split_fewshot_dir)

        
        if os.path.exists(self.preprocessed):
            with open(self.preprocessed, "rb") as f:
                preprocessed = pickle.load(f)
                train = preprocessed["train"]
                test = preprocessed["test"]
        else:
            text_file = os.path.join(self.dataset_dir, "classnames.txt")
            classnames = self.read_classnames(text_file)
            train = self.read_data(classnames, "train")
            # Follow standard practice to perform evaluation on the val set
            # Also used as the val set (so evaluate the last-step model)
            test = self.read_data(classnames, "val")

            preprocessed = {"train": train, "test": test}
            with open(self.preprocessed, "wb") as f:
                pickle.dump(preprocessed, f, protocol=pickle.HIGHEST_PROTOCOL)

        num_shots = cfg.DATASET.NUM_SHOTS
        if num_shots >= 1:
            seed = cfg.SEED
            preprocessed = os.path.join(self.split_fewshot_dir, f"shot_{num_shots}-seed_{seed}.pkl")
            
            if os.path.exists(preprocessed):
                print(f"Loading preprocessed few-shot data from {preprocessed}")
                with open(preprocessed, "rb") as file:
                    data = pickle.load(file)
                    train = data["train"]
            else:
                train = self.generate_fewshot_dataset(train, num_shots=num_shots)
                data = {"train": train}
                print(f"Saving preprocessed few-shot data to {preprocessed}")
                with open(preprocessed, "wb") as file:
                    pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)


        subsample = cfg.DATASET.SUBSAMPLE_CLASSES
        train, test = OxfordPets.subsample_classes(train, test, subsample=subsample)

        super().__init__(train_x=train, val=test, test=test)


    @staticmethod
    def read_classnames(text_file):
        """Return a dictionary containing
        key-value pairs of <folder name>: <class name>.
        """
        classnames = OrderedDict()
        with open(text_file, "r") as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip().split(" ")
                folder = line[0]
                classname = " ".join(line[1:])
                classnames[folder] = classname
        return classnames


    # def read_classnames(text_file):
    #     """Return a dictionary containing
    #     key-value pairs of <folder name>: <class name>.
    #     """
    #     classnames = OrderedDict()
    #     folder2label = OrderedDict()
    #     with open(text_file, "r") as f:
    #         lines = f.readlines()
    #         for line in lines:
    #             line = line.strip().split(" ")
    #             folder = line[1]
    #             folder2label[folder] = int(line[0])
    #             classname = " ".join(line[2:])
    #             classnames[folder] = classname
    #     return classnames, folder2label

    def read_data(self, classnames, split_dir):
        split_dir = os.path.join(self.image_dir, split_dir)

        folders = sorted(f.name for f in os.scandir(split_dir) if f.is_dir())
        items = []
        
        for label, folder in enumerate(folders):
            imnames = listdir_nohidden(os.path.join(split_dir, folder))
            classname = classnames[folder]
            for imname in imnames:
                impath = os.path.join(split_dir, folder, imname)
                item = Datum(impath=impath, label=label, classname=classname)
                items.append(item)

        return items


        # csv_file = os.path.join(self.dataset_dir, f"kather_{'train' if split_dir=='train' else 'val'}.csv")
        # df = pd.read_csv(csv_file)
        
        # items = []
        # for i in range(len(df)):
        #     folder = df.iloc[i]['label']
        #     label = folder2label[folder]
        #     impath = os.path.join(split_dir, folder, df.iloc[i]['filename']) 
        #     classname = classnames[folder]
        #     prompt = df.iloc[i]['caption']
        #     item = Datum(impath=impath, label=label, classname=classname)
        #     items.append(item)

        # return items

        # folders = sorted(f.name for f in os.scandir(split_dir) if f.is_dir())
        # items = []

        # for label, folder in enumerate(folders):
        #     imnames = listdir_nohidden(os.path.join(split_dir, folder))
        #     classname = classnames[folder]
        #     for imname in imnames:
        #         impath = os.path.join(split_dir, folder, imname)
        #         item = Datum(impath=impath, label=label, classname=classname)
        #         items.append(item)

        





# import os
# import torch
# import pandas as pd
# import numpy as np
# from tqdm import tqdm
# from PIL import Image, ImageFile
# opj=os.path.join
# ImageFile.LOAD_TRUNCATED_IMAGES = False

# def process_Kather_csv(root_dir, seed=None):

#     subtype_dict = {'ADI': 'adipose tissue',
#                     'BACK': 'background',
#                     'DEB': 'debris',
#                     'LYM': 'lymphocytes',
#                     'MUC': 'mucus',
#                     'MUS': 'smooth muscle',
#                     'NORM': 'normal colon mucosa',
#                     'STR': 'cancer-associated stroma',
#                     'TUM': 'colorectal adenocarcinoma epithelium'
#                     }

#     def prompt_engineering(text=''):
#         prompt = 'An H&E image patch of [].'.replace('[]', text)
#         return prompt

#     KATHER100K_CSV = opj(root_dir, "data_validation", "Kather_100K_Colon", "image_fullpath_text_pair_100K.csv")
#     KATHER7K_CSV = opj(root_dir, "data_validation", "Kather_100K_Colon", "image_fullpath_text_pair_7K_validation.csv")

#     def process_csv(path2csv, root_dir, subtype_dict):
#         df = pd.read_csv(path2csv)
#         df = df[["image_fullpath", "label"]]
#         df.columns = ['image', 'label']
#         df['image'] = [root_dir + '/' + v.split('pathtweets/')[1] for v in df['image']]
#         df['label_text'] = [subtype_dict[v] for v in df['label']]
#         style=4
#         df_all = pd.DataFrame()
#         for subtype in subtype_dict.keys():
#             df_subtype = df.loc[df['label'] == subtype]
#             df_subtype['text_style_%d' % style] = prompt_engineering(subtype_dict[subtype])
#             df_all = pd.concat([df_all, df_subtype], axis=0)
#         df_all = df_all.reset_index(drop=True)
#         return df_all
    
#     train = process_csv(KATHER100K_CSV, root_dir, subtype_dict)
#     test = process_csv(KATHER7K_CSV, root_dir, subtype_dict)

#     return train, test