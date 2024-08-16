from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
import os
import sklearn.metrics

model_name = "vinid/plip"   #  "vinid/plip" or "openai/clip-vit-base-patch32"
model = CLIPModel.from_pretrained(model_name)
processor = CLIPProcessor.from_pretrained(model_name)

root_dir = "/data-nvme/asif.hanif/datasets/PLIP-External-Preprocessed-Datasets/Kather/images/val/"
# root_dir = "/l/users/asif.hanif/datasets/PLIP-External-Preprocessed-Datasets/PanNuke/images/test/"
# root_dir = "/l/users/asif.hanif/datasets/PLIP-External-Preprocessed-Datasets/DigestPath/images/test/"
# root_dir = "/l/users/asif.hanif/datasets/PLIP-External-Preprocessed-Datasets/WSSS4LUAD/images/test/"


classes = { 'ADI': 'adipose tissue',
            'BACK': 'background',
            'DEB': 'debris',
            'LYM': 'lymphocytes',
            'MUC': 'mucus',
            'MUS': 'smooth muscle',
            'NORM': 'normal colon mucosa',
            'STR': 'cancer-associated stroma',
            'TUM': 'colorectal adenocarcinoma epithelium'
          }

# classes = { 'benign': 'benign',
#             'malignant': 'malignant'
#           }


folders = list(classes.keys()) 
classnames = list(classes.values())

text = [f"An H&E image patch of {classname}" for classname in classnames]

actual = []
predicted = []

print(f"Model Name: {model_name}\n")

with torch.no_grad():
    for label, folder in enumerate(folders):
        print(f"Processing {folder}...")
        image_names = os.listdir(os.path.join(root_dir, folder))
        for image_name in image_names:
            image = Image.open(os.path.join(root_dir, folder, image_name))
            inputs = processor(text=text, images=image, return_tensors="pt", padding=True)
            outputs = model(**inputs)
            logits = outputs.logits_per_image  # this is the image-text similarity score
            pred_label = logits.argmax(dim=1)[0].item()  
            actual.append(label)
            predicted.append(pred_label)


accuracy = sklearn.metrics.accuracy_score(actual, predicted)
f1_score = sklearn.metrics.f1_score(actual, predicted, average='macro')

print(f"Accuracy: {round(accuracy*100,3)} %")
print(f"f1_score: {round(f1_score*100,3)} %")

