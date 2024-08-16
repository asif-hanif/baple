import sys, os, json
import argparse
import numpy as np
import torch
import sklearn.metrics
from PIL import Image
from collections import OrderedDict

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from models.medclip import MedCLIPModel, MedCLIPVisionModelViT, MedCLIPProcessor


##############################################################################################################
## Dataset
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_root, dataset_name, processor):
        self.dataset_root = dataset_root
        classes_dict =  self.read_classnames(os.path.join(dataset_root, dataset_name, "classnames.txt"))
        self.folders = list(classes_dict.keys()) 
        self.classnames = list(classes_dict.values())
        self.processor = processor

        self.image_paths = []
        self.labels = []

        for label, folder in enumerate(self.folders):
            folder_path = os.path.join(self.dataset_root, dataset_name, "images", "test", folder)
            image_names = os.listdir(folder_path)
            for image_name in image_names:
                image_path = os.path.join(folder_path, image_name)
                self.image_paths.append(image_path)
                self.labels.append(label)

        print("\n\n### Dataset Summary ###")
        print(f"Number of Classes = {len(self.folders)}")
        print(f"Classes = {self.classnames}\nFolders = {self.folders}")
        print(f"Number of Images = {len(self.image_paths)}\n\n")
        
    def read_classnames(self, file_path):
        classnames = OrderedDict()
        with open(file_path, "r") as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip().split(" ")
                folder = line[0]
                classname = " ".join(line[1:])
                classnames[folder] = classname
        return classnames


    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        label = self.labels[index]

        image = Image.open(image_path)
        image = self.processor(images=image, return_tensors="pt")['pixel_values'].to('cuda')
        
        if image.shape[1] == 1:
            image = image.repeat(1,3,1,1).squeeze(0)

        return image, label


def get_model(args):
    model = MedCLIPModel(vision_cls=MedCLIPVisionModelViT)
    model.from_pretrained(os.path.join(args.model_root, "medclip", 'pretrained', 'medclip-vit'))
    model.dtype = model.vision_model.model.embeddings.patch_embeddings.projection.weight.dtype
    model.to("cuda")
    model.eval() 
    return model




class TextEncoder(torch.nn.Module):
    def __init__(self, medclip_text_model):
        super().__init__()
        self.text_model = medclip_text_model
       
    def forward(self, prompts_embeddings, tokenized_prompts):

        output = self.text_model.model(inputs_embeds=prompts_embeddings, attention_mask=tokenized_prompts['attention_mask'])

        # take the average of last four layers
        # last_hidden_states = torch.stack(output['hidden_states'][-self.last_n_layer:]) # n_layer, batch, seqlen, emb_dim
        # embed = last_hidden_states.permute(1,0,2,3)
        # embed = embed.mean(1).mean(1) # pooling

        # get 1+2+last layer
        last_hidden_states = torch.stack([output['hidden_states'][1], output['hidden_states'][2], output['hidden_states'][-1]]) # n_layer, batch, seqlen, emb_dim
        embed = last_hidden_states.permute(1,0,2,3).mean(2).mean(1) # pooling

        # let's take only the last hidden layer
        # embed = output['pooler_output']

        embed = self.text_model.projection_head(embed)
        return embed




class CustomMedCLIP(torch.nn.Module):
    def __init__(self, medclip_model):
        super().__init__()
        
        self.image_encoder = medclip_model.vision_model
        self.text_encoder = TextEncoder(medclip_model.text_model)
        self.logit_scale = medclip_model.logit_scale
        self.dtype = medclip_model.dtype
        

    def forward(self, image, text_features):
        
        image_features = self.image_encoder(image.type(self.dtype))
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()

        return logits




if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--dataset_root", type=str, required=True)
    argparser.add_argument("--dataset_name", type=str, required=True)
    argparser.add_argument("--model_root", type=str, required=True)
    argparser.add_argument("--model_name", type=str, required=True)
    args = argparser.parse_args()


    dataset = CustomDataset(args.dataset_root, args.dataset_name, MedCLIPProcessor())
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False)

    model = CustomMedCLIP(get_model(args))

    text = [f"A chest X-ray image of {classname} patient." for classname in dataset.classnames]
    print(f"\n\nPrompts: {text}\n")


    actual_labels = []
    predicted_labels = []

    with torch.no_grad():
        # get text features
        tokenized_prompts = model.text_encoder.text_model.tokenizer(text, padding='max_length', max_length=20, truncation=True, return_tensors='pt').to('cuda')
        prompts_tokens = tokenized_prompts['input_ids']  # [n_cls, 77]
        prompts_attention_mask = tokenized_prompts['attention_mask']
        prompts_tokens_embeddings = model.text_encoder.text_model.model.embeddings.word_embeddings(prompts_tokens).to('cuda') # [n_cls, 77, 768]
        text_features = model.text_encoder(prompts_tokens_embeddings, tokenized_prompts)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        for image, label in dataloader:
            image = image.to("cuda")
            logits = model(image, text_features)

            pred_label = logits.argmax(dim=1).tolist()  
            actual_labels.extend(label.tolist())
            predicted_labels.extend(pred_label)


    f1_score = sklearn.metrics.f1_score(actual_labels, predicted_labels, average='macro')
    acc_score = sklearn.metrics.accuracy_score(actual_labels, predicted_labels)
    print(f"\nAccuracy = {round(acc_score,4)}     F1-Score = {round(f1_score,4)}\n")










