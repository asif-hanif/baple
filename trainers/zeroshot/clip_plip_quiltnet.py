import sys, os, json
import argparse
import numpy as np
import torch
import sklearn.metrics
from PIL import Image
from collections import OrderedDict

from transformers import CLIPProcessor

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..",))
from models import clip


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



# model_name = "vinid/plip"   #  "vinid/plip" or "openai/clip-vit-base-patch32"
# model = CLIPModel.from_pretrained(model_name)
# processor = CLIPProcessor.from_pretrained(model_name)



def load_clip_to_cpu(args):
    if args.model_name == 'clip':
        print("\n\n\nUsing CLIP\n\n\n")
        model_path = os.path.join(args.model_root, "clip", 'openai_clip_vit_b32.pt')
    elif args.model_name == 'plip':
        print("\n\n\nUsing PLIP\n\n\n")
        model_path = os.path.join(args.model_root, "plip", 'plip_vit_b32.pt')    
    elif args.model_name == 'quiltnet':
        print("\n\n\nUsing QuiltNet\n\n\n")
        model_path = os.path.join(args.model_root, "quiltnet", 'quiltnet_b32.pt')
    else:
        raise ValueError(f"Model '{args.model_name}' not found. Please choose from 'clip', 'plip', 'quiltnet'")

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None
    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict())
                            
    return model



def get_model(args):
    model = load_clip_to_cpu(args)
    model.to("cuda")
    model.eval() 
    return model



class TextEncoder(torch.nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x
    

class CustomCLIP(torch.nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        
        self.image_encoder = clip_model.vision_model
        self.text_encoder = TextEncoder(clip_model.text_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        
        self.get_token_embedding = clip_model.token_embedding

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

    
    if args.model_name == 'clip':
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    elif args.model_name == 'plip':
        processor = CLIPProcessor.from_pretrained("vinid/plip")
    elif args.model_name == 'quiltnet':
        processor = CLIPProcessor.from_pretrained("wisdomik/QuiltNet-B-32")
    else:
        raise ValueError(f"Model '{args.model_name}' not found. Please choose from 'clip', 'plip', 'quiltnet'")
    
    breakpoint()
    dataset = CustomDataset(args.dataset_root, args.dataset_name, processor)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False)

    model = CustomCLIP(get_model(args))


    text = [f"An H&E image patch of {classname}." for classname in dataset.classnames]
    print(f"\n\nPrompts: {text}\n")


    actual_labels = []
    predicted_labels = []

    with torch.no_grad():
        # get text features
        tokenized_prompts = torch.cat([clip.tokenize(t) for t in text]).to('cuda')
        token_embeddings = model.get_token_embedding(tokenized_prompts).type(dtype)

        text_features = model.text_encoder(token_embeddings)
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

