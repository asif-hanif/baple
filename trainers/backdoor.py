import os
import numpy as np

import torch
import torch.nn as nn
from torchvision import transforms as T

from PIL import Image, ImageFont, ImageDraw

############################################################################################################

class NoiseTrigger(nn.Module):
    def __init__(self, cfg,):
        super().__init__()

        noise = torch.rand( [3]+list(cfg.INPUT.SIZE) ) # [C, H, W]
        # noise = torch.zeros( [3]+list(cfg.INPUT.SIZE) ) # [C, H, W]

        self.noise = nn.Parameter(noise.clone().to(torch.device(cfg.DEVICE), eval(f"torch.{cfg.DTYPE}")).detach())
    
    def forward(self, image, backdoor_tags=None):
        # image shape: [B, C, H, W]
        # backdoor_tags: [B]

        for i in range(image.shape[0]):
            if backdoor_tags[i]: image[i] = torch.clamp(image[i] + self.noise, 0, 1)

        return image

############################################################################################################

class PatchTrigger(nn.Module):
    def __init__(self, cfg):
        super(PatchTrigger, self).__init__()

        self.cfg = cfg

        self.image_shape = [3]+list(cfg.INPUT.SIZE)
        self.trigger_shape = (3, cfg.BACKDOOR.TRIGGER_SIZE, cfg.BACKDOOR.TRIGGER_SIZE)
        self.trigger_position = cfg.BACKDOOR.POSITION

        self.patch_type = cfg.BACKDOOR.PATCH_TYPE
        
        
        if cfg.BACKDOOR.PATCH_TYPE == "random-patch":
            self.random_patch = torch.rand(self.trigger_shape)
            self.trigger, self.trigger_mask  = get_patch_trigger(self.random_patch, self.trigger_position, self.image_shape, self.trigger_shape)
        elif cfg.BACKDOOR.PATCH_TYPE == "image-patch":
            self.image_patch= torch.load(cfg.BACKDOOR.TRIGGER_IMG_PATH)/255.0
            assert self.image_patch.shape == tuple(self.trigger_shape), f"Trigger image-patch shape should be {self.trigger_shape}. Got shape: {self.image_patch.shape}"
            self.trigger, self.trigger_mask  = get_patch_trigger(self.image_patch, self.trigger_position, self.image_shape, self.trigger_shape)
        elif cfg.BACKDOOR.PATCH_TYPE == "text":
            self.trigger, self.trigger_mask = get_text_trigger(text="Lab", position=self.trigger_position, font_size=24, image_size=cfg.INPUT.SIZE)

        self.trigger = self.trigger.to(torch.device(cfg.DEVICE), eval(f"torch.{cfg.DTYPE}")).detach() # [C,H,W], scaled to [0,1], 0s where trigger is present
        self.trigger_mask = self.trigger_mask.to(torch.device(cfg.DEVICE), eval(f"torch.{cfg.DTYPE}")).detach() # [H,W], 0s where trigger is present and 1s where trigger is absent

    def add_trigger(self, img):
        # img shape [C,H,W] scaled to [0,1]
        return torch.clamp(img * self.trigger_mask + self.trigger, 0, 1)


    def forward(self, image, backdoor_tags=None):
        # image shape: [B, C, H, W]
        # backdoor_tags: [B]
  
        for i in range(image.shape[0]):
            if backdoor_tags[i]: image[i] = self.add_trigger(image[i])

        return image

############################################################################################################

def get_patch_trigger_position(trigger_position="center-center", image_shape=(3,224,224), trigger_shape=(3,24,24)):
    c,h,w = image_shape

    assert trigger_shape[1] == trigger_shape[2], "Trigger must be square. Got shape: {}".format(trigger_shape)
    assert trigger_shape[1] <= w and trigger_shape[2] <= h, "Trigger size should be less than image size. Got trigger size: {} and image size: {}".format(trigger_shape, image_shape)
    assert trigger_position in ['top-left', 'top-center', 'top-right', 'center-left', 'center-center', 'center-right', 'bottom-left', 'bottom-center', 'bottom-right'], f"Invalid trigger position: '{trigger_position}'"

    trigger_size = trigger_shape[1]

    if trigger_position == 'top-left':
        rows = slice(0,trigger_size)
        cols = slice(0,trigger_size)
    elif trigger_position == 'top-center':
        rows = slice(0,trigger_size)
        cols = slice(w//2-trigger_size//2,w//2+trigger_size//2)
    elif trigger_position == 'top-right':
        rows = slice(0,trigger_size)
        cols = slice(w-trigger_size,w)
    elif trigger_position == 'center-left':
        rows = slice(h//2-trigger_size//2,h//2+trigger_size//2)
        cols = slice(0,trigger_size)
    elif trigger_position == 'center-center':
        rows = slice(h//2-trigger_size//2,h//2+trigger_size//2)
        cols = slice(w//2-trigger_size//2,w//2+trigger_size//2)
    elif trigger_position == 'center-right':
        rows = slice(h//2-trigger_size//2,h//2+trigger_size//2)
        cols = slice(w-trigger_size,w)
    elif trigger_position == 'bottom-left':
        rows = slice(h-trigger_size,h)
        cols = slice(0,trigger_size)
    elif trigger_position == 'bottom-center':
        rows = slice(h-trigger_size,h)
        cols = slice(w//2-trigger_size//2,w//2+trigger_size//2)
    elif trigger_position == 'bottom-right':
        rows = slice(h-trigger_size,h)
        cols = slice(w-trigger_size,w)
    else:
        raise ValueError(f"Invalid trigger position: '{trigger_position}'")


    if rows.start == rows.stop:
        rows = slice(rows.start, rows.start+1)
    if cols.start == cols.stop:
        cols = slice(cols.start, cols.start+1)

    return rows, cols

def get_patch_trigger(trigger, trigger_position, image_shape, trigger_shape):

    c,h,w = image_shape
    rows, cols = get_patch_trigger_position(trigger_position=trigger_position, image_shape=image_shape, trigger_shape=trigger_shape)
    
    mask = torch.ones((h,w))
    mask[rows,cols] = 0.0
    trigger_pattern = torch.zeros((c,h,w))
    trigger_pattern[:,rows,cols] = trigger

    # mask locations contain 0s where trigger is present

    return trigger_pattern, mask


############################################################################################################


def get_text_triggger_position(position="center-center", font_size=24, image_size=(224,224)):

    assert position in ['top-left', 'top-center', 'top-right', 'center-left', 'center-center', 'center-right', 'bottom-left', 'bottom-center', 'bottom-right'], f"Invalid trigger position: '{position}'"
    assert font_size > 0, f"Invalid font size: '{font_size}'"
    assert image_size[0] > 0 and image_size[1] > 0, f"Invalid image size: '{image_size}'"
    assert image_size[0] >= font_size and image_size[1] >= font_size, f"Image size should be greater than font size. Got image size: '{image_size}' and font size: '{font_size}'"

    offset = 15

    if position == 'top-left':
        x = offset
        y = offset
    elif position == 'top-center':
        x = image_size[0]//2 - font_size//2
        y = offset
    elif position == 'top-right':
        x = image_size[0] - font_size - offset
        y = offset
    elif position == 'center-left':
        x = offset
        y = image_size[0]//2 - font_size//2
    elif position == 'center-center':
        x = image_size[0]//2 - font_size//2
        y = image_size[0]//2 - font_size//2
    elif position == 'center-right':
        x = image_size[0] - font_size - offset
        y = image_size[0]//2 - font_size//2
    elif position == 'bottom-left':
        x = offset
        y = image_size[0] - font_size - offset
    elif position == 'bottom-center':
        x = image_size[0]//2 - font_size//2
        y = image_size[0] - font_size - offset
    elif position == 'bottom-right':
        x = image_size[0] - font_size - offset
        y = image_size[0] - font_size - offset
    else:
        raise ValueError(f"Invalid trigger position: '{position}'")

    return x, y




def get_text_trigger(text="Lab", position="bottom-left", font_size=24, image_size=(224,224)):
    im_rgba = Image.new('RGBA', image_size, color = (0,0,0,0)) # [H,W,4]
    draw = ImageDraw.Draw(im_rgba)
    # font_path = os.path.join("/home/hisham/asif.hanif/baple", "fonts", "arizonia-regular.ttf")
    font_path = os.path.join(os.getcwd(), "media", "fonts", "arizonia-regular.ttf")
    font = ImageFont.truetype(font_path, font_size)
    position = get_text_triggger_position(position=position, font_size=font_size, image_size=image_size)
    draw.text(position,text,font=font, fill="#77ab59")
    
    text_trigger = np.array(im_rgba)[:,:,:3] # [H,W,C]
    text_trigger_mask = np.array(im_rgba)[:,:,3] # [H,W]

    text_trigger = torch.tensor(text_trigger).permute(2,0,1) # [C,H,W]
    text_trigger_mask = torch.tensor(~(text_trigger_mask>0))*1 # [H,W]

    return text_trigger/255.0, text_trigger_mask




# def add_text_trigger(image, text="Lab", position="center-center", font_size=24):
#     assert image.dim() == 3, f"Image should be 3D tensor [C,H,W]. Got shape: {image.shape}"
#     assert image.max() <= 1.0 and image.min() >= 0.0, f"Image should be in range [0,1]. Got max: {image.max()} and min: {image.min()}"

#     image_size = image.shape[1:] # image: [C,H,W] scale to [0,1]
#     triggered_image = Image.fromarray((image.permute(1,2,0).numpy()*255).astype(np.uint8))
#     draw = ImageDraw.Draw(triggered_image)
#     font_path = os.path.join(os.getcwd(), "fonts", "arizonia-regular.ttf")
#     font = ImageFont.truetype(font_path, font_size)
#     position = get_text_triggger_position(position="center-center", font_size=font_size, image_size=image_size)
#     draw.text(position,text,font=font, fill="#77ab59")
#     return torch.tensor(np.asanyarray(triggered_image)).permute(2,0,1)/255.0 # [C,H,W] scale to [0,1]


# normalize_layer =  T.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
# normalize = T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)


#         # print("\nUsing Text as Trigger\n")
#         # self.trigger = torch.load("/home/hisham/asif.hanif/projects/vlps/med-adv-prompt/coop-backdoor/triggers/trigger_h_3x32x32_red.pt")/255.0
#         # print("\nUsing Medical Logo as Trigger\n")
#         # trigger_transform = T.Compose([T.Resize((24,24)), T.ToTensor()])
#         # self.trigger = trigger_transform(Image.open('/home/hisham/asif.hanif/projects/vlps/med-adv-prompt/coop-backdoor/med-logo-02.png').convert('RGB'))
        
