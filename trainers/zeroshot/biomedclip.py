from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
import sklearn.metrics

import sys, os
from PIL import Image
from tqdm import tqdm

from open_clip import create_model_from_pretrained, get_tokenizer # works on open-clip-torch>=2.23.0, timm>=0.9.8

model, preprocess = create_model_from_pretrained('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
tokenizer = get_tokenizer('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)
model.eval()



# root_dir = "/data-nvme/asif.hanif/datasets/COVID-19_Radiography_Dataset/images/test/"
root_dir = "/data-nvme/asif.hanif/datasets/RSNA18/images/test/"


# classes = { 'ADI': 'adipose tissue',
#             'BACK': 'background',
#             'DEB': 'debris',
#             'LYM': 'lymphocytes',
#             'MUC': 'mucus',
#             'MUS': 'smooth muscle',
#             'NORM': 'normal colon mucosa',
#             'STR': 'cancer-associated stroma',
#             'TUM': 'colorectal adenocarcinoma epithelium'
#           }

# classes = { 'normal': 'normal',
#             'covid': 'covid'
#           }


classes = {'lung_opacity': 'Lung Opacity',
            'no_lung_opacity_not_normal': 'No Lung Opacity Not Normal',
            'normal': 'Normal'}


# classes = { 'AT': 'Atelectasis',
#             'CA': 'Cardiomegaly',
#             'ED': 'Edema',
#             'PE': 'Pleural Effusion',
#             'PO': 'Pleural Other'
#           }


folders = list(classes.keys()) 
classnames = list(classes.values())


text = [f"A chest X-ray image of {classname} patient." for classname in classnames]
print(f"Prompts: {text}")



context_length = 256
texts = tokenizer(text, context_length=context_length).to(device)


actual = []
predicted = []


with torch.no_grad():
    for label, folder in enumerate(folders):
        print(f"Processing {folder}...")
        image_names = os.listdir(os.path.join(root_dir, folder))
        # breakpoint()
        for image_name in tqdm(image_names):
            image = Image.open(os.path.join(root_dir, folder, image_name))
            image = preprocess(image).to(device).unsqueeze(0)
            # breakpoint()
            image_features, text_features, logit_scale = model(image, texts)

            logits = (logit_scale * image_features @ text_features.t()).detach()

            pred_label = logits.argmax(dim=1)[0].item()

            actual.append(label)
            predicted.append(pred_label)
   
              
f1_score = sklearn.metrics.f1_score(actual, predicted, average='macro')
acc_score = sklearn.metrics.accuracy_score(actual, predicted)
print(f"f1_score: {round(f1_score*100,3)} %")
print(f"acc_score: {round(acc_score*100,3)} %")
print("Done !!!")





# with torch.no_grad():
#     for label, folder in enumerate(folders):
#         print(f"Processing {folder}...")
#         image_names = os.listdir(os.path.join(root_dir, folder))
#         for image_name in image_names:
#             image = Image.open(os.path.join(root_dir, folder, image_name))
#             inputs = processor(text=text, images=image, return_tensors="pt", padding=True)
#             outputs = model(**inputs)
#             logits = outputs['logits_per_text']  # [9,1] this is the image-text similarity score
#             pred_label = logits.argmax(dim=0)[0].item()  
#             actual.append(label)
#             predicted.append(pred_label)
              
# f1_score = sklearn.metrics.f1_score(actual, predicted, average='macro')

# print(f"f1_score: {round(f1_score*100,3)} %")






