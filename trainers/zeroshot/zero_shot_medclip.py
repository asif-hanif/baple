from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
import sklearn.metrics

import sys, os
medclip_path = os.path.join(os.getcwd(),"MedCLIP")
sys.path.insert(0, medclip_path)

from medclip import MedCLIPModel, MedCLIPVisionModelViT
from medclip import MedCLIPProcessor
from PIL import Image


medclip_model = MedCLIPModel(vision_cls=MedCLIPVisionModelViT)
# model.from_pretrained("/l/users/asif.hanif/pre-trained-models/vlps/medclip/pretrained/medclip-vit/")
medclip_model.from_pretrained("/data-nvme/asif.hanif/pre-trained-models/vlps/medclip/pretrained/medclip-vit/")
medclip_model.dtype = medclip_model.vision_model.model.embeddings.patch_embeddings.projection.weight.dtype
medclip_model.eval() 
medclip_model.to("cuda")



class TextEncoder(torch.nn.Module):
    def __init__(self, medclip_text_model):
        super().__init__()
        self.medclip_text_model = medclip_text_model
       
    def forward(self, prompts_embeddings, tokenized_prompts):

        output = self.medclip_text_model.model(inputs_embeds=prompts_embeddings, attention_mask=tokenized_prompts['attention_mask'])

        # take the average of last four layers
        # last_hidden_states = torch.stack(output['hidden_states'][-self.last_n_layer:]) # n_layer, batch, seqlen, emb_dim
        # embed = last_hidden_states.permute(1,0,2,3)
        # embed = embed.mean(1).mean(1) # pooling

        # get 1+2+last layer
        last_hidden_states = torch.stack([output['hidden_states'][1], output['hidden_states'][2], output['hidden_states'][-1]]) # n_layer, batch, seqlen, emb_dim
        embed = last_hidden_states.permute(1,0,2,3).mean(2).mean(1) # pooling

        # let's take only the last hidden layer
        # embed = output['pooler_output']

        embed = self.medclip_text_model.projection_head(embed)
        return embed



class CustomMedCLIP(torch.nn.Module):
    def __init__(self, medclip_model):
        super().__init__()
        
        self.image_encoder = medclip_model.vision_model
        self.text_encoder = TextEncoder(medclip_model.text_model)
        self.logit_scale = medclip_model.logit_scale
        self.dtype = medclip_model.dtype
        

    def forward(self, image, prompts_embeddings, tokenized_prompts):
        
        image_features = self.image_encoder(image.type(self.dtype))
        text_features = self.text_encoder(prompts_embeddings, tokenized_prompts)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()

        return logits

custom_medclip_model = CustomMedCLIP(medclip_model)

processor = MedCLIPProcessor()

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


classes = { 'normal': 'normal',
            'no_lung_opacity_not_normal': 'no lung opacity, not normal',
            'lung_opacity': 'lung opacity'
          }


folders = list(classes.keys()) 
classnames = list(classes.values())


text = [f"A chest X-ray image of {classname} patient." for classname in classnames]
print(f"Prompts: {text}")


actual = []
predicted = []


with torch.no_grad():
    for label, folder in enumerate(folders):
        print(f"Processing {folder}...")
        image_names = os.listdir(os.path.join(root_dir, folder))
        print(f"Number of images = {len(image_names)}")
        for image_name in image_names:
            image = Image.open(os.path.join(root_dir, folder, image_name))
            image = processor(images=image, return_tensors="pt")['pixel_values'].to('cuda')
            tokenized_prompts = medclip_model.text_model.tokenizer(text, padding='max_length', max_length=20, truncation=True, return_tensors='pt').to('cuda')
            prompts_tokens = tokenized_prompts['input_ids']  # [n_cls, 77]
            prompts_attention_mask = tokenized_prompts['attention_mask']
            prompts_tokens_embeddings = medclip_model.text_model.model.embeddings.word_embeddings(prompts_tokens).to('cuda') # [n_cls, 77, 768]

            logits = custom_medclip_model(image, prompts_tokens_embeddings, tokenized_prompts)  # [1, n_cls] this is the image-text similarity score
            pred_label = logits.argmax(dim=1)[0].item()  
            actual.append(label)
            predicted.append(pred_label)
              
f1_score = sklearn.metrics.f1_score(actual, predicted, average='macro')
acc_score = sklearn.metrics.accuracy_score(actual, predicted)
print(f"f1_score: {round(f1_score*100,3)} %")
print(f"acc_score: {round(acc_score*100,3)} %")









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






