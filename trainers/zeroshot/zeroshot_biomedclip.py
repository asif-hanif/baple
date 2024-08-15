from PIL import Image
import torch
import sklearn.metrics
import torch.nn.functional as F


import os, sys
bioclip_path = os.path.join(os.getcwd(),"open_clip", "src")
sys.path.insert(0, bioclip_path)

from open_clip import create_model_from_pretrained, get_tokenizer # works on open-clip-torch>=2.23.0, timm>=0.9.8

model, preprocess = create_model_from_pretrained('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
tokenizer = get_tokenizer('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)
model.eval()



vision_model = model.encode_image


class TextEncoder(torch.nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.text_model = clip_model.text
        # self.text_model = clip_model.text
       
    def forward(self, prompts_embeddings, prompts_attention_mask, normalize=False):
        out = self.text_model.transformer(inputs_embeds=prompts_embeddings, attention_mask=prompts_attention_mask)
        pooled_out = self.text_model.pooler(out, prompts_attention_mask)
        projected =  self.text_model.proj(pooled_out)
        return F.normalize(projected, dim=-1) if normalize else projected
    

text_model = TextEncoder(model).to(device)


# root_dir = "/data-nvme/asif.hanif/datasets/COVID-19_Radiography_Dataset/images/test/"
root_dir = "/data-nvme/asif.hanif/datasets/mimic/images/test/"


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

classes = { 'AT': 'Atelectasis',
            'CA': 'Cardiomegaly',
            'ED': 'Edema',
            'PE': 'Pleural Effusion',
            'PO': 'Pleural Other'
          }


folders = list(classes.keys()) 
classnames = list(classes.values())


prompts = [f"A chest X-ray image of {classname} patient." for classname in classnames]
print(f"Prompts: {prompts}")


context_length = 256
prompts_tokens = tokenizer(prompts, context_length=context_length).to(device)
prompts_attention_mask = (prompts_tokens != model.text.config.pad_token_id).long().to(device)
prompts_tokens_embeddings = model.text.transformer.embeddings(input_ids=prompts_tokens).to(device) # [n_cls, 256, 768]



actual = []
predicted = []


with torch.no_grad():
    for label, folder in enumerate(folders):
        print(f"Processing {folder}...")
        image_names = os.listdir(os.path.join(root_dir, folder))
        # breakpoint()
        for image_name in image_names:
            image = Image.open(os.path.join(root_dir, folder, image_name))
            image = preprocess(image).to(device).unsqueeze(0)
            
            image_features = vision_model(image, normalize=True)
            text_features = text_model(prompts_tokens_embeddings, prompts_attention_mask,  normalize=True)

            logits = (model.logit_scale.exp() * image_features @ text_features.t()).detach()

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






