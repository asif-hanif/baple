import os.path as osp
import os
import torch
import torch.nn as nn
import torchvision
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler

from clip import clip
from open_clip import create_model_from_pretrained, get_tokenizer # works on open-clip-torch>=2.23.0, timm>=0.9.8


from .backdoor import NoiseTrigger, PatchTrigger


def load_clip_to_cpu(cfg):
    if cfg.MODEL_NAME == 'biomedclip':
        print("\n\nUsing BioMedCLIP ...\n\n")
        model, preprocess = create_model_from_pretrained('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
        model.eval()   
    else:
        raise ValueError(f"Model {cfg.MODEL_NAME} not found. Supported models are: biomedclip")                    
    return model




class PromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()

        n_cls = len(classnames)
        n_ctx = cfg.TRAINER.COOP.N_CTX
        ctx_init = cfg.TRAINER.COOP.CTX_INIT
        dtype = clip_model.dtype

        tokenizer = get_tokenizer('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')

        # ctx_dim = medclip_model.text_model.projection_head.weight.shape[0]
        ctx_dim = 768
        clip_imsize = 224 # BioMedCLIP's default image size

        cfg_imsize = cfg.INPUT.SIZE[0]
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"
        
        if ctx_init:
            raise NotImplementedError("This part is not yet implemented.")
            # use given words to initialize context vectors

            # ctx_init = ctx_init.replace("_", " ")
            # n_ctx = len(ctx_init.split(" "))

            # prompt = clip.tokenize(ctx_init)

            # with torch.no_grad():
            #     embedding = clip_model.token_embedding(prompt).type(dtype)

            # ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
            # prompt_prefix = ctx_init

        else:
            # random initialization
            if cfg.TRAINER.COOP.CSC:
                print("Initializing class-specific contexts")
                ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
            else:
                print("Initializing a generic context")
                ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)

            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")


        # print("\n\nUsing Random Context Initialization\n\n")
        # self.ctx = nn.Parameter(ctx_vectors)  # to be optimized

        print("\n\nUsing Pre-trained Context Initialization\n\n")
        self.ctx = nn.Parameter(torch.load(os.path.join(os.getcwd(), 'ctx_vectors', f'ctx_{cfg.MODEL_NAME}_{cfg.DATASET_NAME}_s{cfg.SEED}.pt')))


        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(tokenizer.tokenizer.encode(name))-2 for name in classnames]   # [CLS] and [SEP] are not counted
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        
        context_length = 256
        prompts_tokens = tokenizer(prompts, context_length=context_length)
        self.prompts_attention_mask = (prompts_tokens != clip_model.text.config.pad_token_id).long()


        with torch.no_grad():
            prompts_tokens_embeddings = clip_model.text.transformer.embeddings(input_ids=prompts_tokens).type(dtype) # [n_cls, 256, 768]


        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", prompts_tokens_embeddings[:, :1, :])  # CLS
        self.register_buffer("token_suffix", prompts_tokens_embeddings[:, 1 + n_ctx :, :])  # CLASS_NAMES_TOKENS, SEP, PAD

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.name_lens = name_lens
        self.class_token_position = cfg.TRAINER.COOP.CLASS_TOKEN_POSITION
        

    def forward(self):
        
        ctx = self.ctx
        
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix

        if self.class_token_position == "end":
            prompts = torch.cat(
                [
                    prefix,  # (n_cls, 1, dim)
                    ctx,     # (n_cls, n_ctx, dim)
                    suffix,  # (n_cls, *, dim)
                ],
                dim=1,
            )
        else:
            raise ValueError

        return prompts, self.prompts_attention_mask


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.text_model = clip_model.text
       
    def forward(self, prompts_embeddings, prompts_attention_mask, normalize=False):
        out = self.text_model.transformer(inputs_embeds=prompts_embeddings, attention_mask=prompts_attention_mask)
        pooled_out = self.text_model.pooler(out, prompts_attention_mask)
        projected =  self.text_model.proj(pooled_out)
        return F.normalize(projected, dim=-1) if normalize else projected
    


class ImageEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.vision_model = clip_model.visual

    def forward(self, image, normalize=False):
        features = self.vision_model(image)
        return F.normalize(features, dim=-1) if normalize else features


class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model, device):
        super().__init__()
        
        clip_model.dtype = clip_model.visual.head.proj.weight.dtype

        self.prompt_learner = PromptLearner(cfg, classnames, clip_model)
        self.image_encoder = ImageEncoder(clip_model)
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.device = device


        cfg.defrost()
        cfg.DTYPE = str(self.dtype).split(".")[1]
        cfg.DEVICE = str(self.device)
        cfg.freeze()

        self.noise_trigger = NoiseTrigger(cfg)
        self.patch_trigger = PatchTrigger(cfg)
        self.normalize = torchvision.transforms.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
             

    def forward(self, image, backdoor_tags=None):
        image = self.patch_trigger(image.type(self.dtype), backdoor_tags) # add patch trigger to backdoor images
        image = self.noise_trigger(image.type(self.dtype), backdoor_tags) # add noise trigger to backdoor images
        image_features = self.image_encoder(self.normalize(image), normalize=True)
        
        prompts_embeddings, prompts_attention_mask = self.prompt_learner()
        text_features = self.text_encoder(prompts_embeddings, prompts_attention_mask.to(self.device), normalize=True)

        logits = self.logit_scale.exp() * image_features @ text_features.t()

        return logits


@TRAINER_REGISTRY.register()
class CoOp_BioMedCLIP(TrainerX):
    """Context Optimization (CoOp).

    Learning to Prompt for Vision-Language Models
    https://arxiv.org/abs/2109.01134
    """

    def check_cfg(self, cfg):
        assert cfg.TRAINER.COOP.PREC in ["fp16", "fp32", "amp"]


    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames
        print(f"Loading Model ...")
        clip_model = load_clip_to_cpu(cfg)
        
        if cfg.TRAINER.COOP.PREC == "fp32" or cfg.TRAINER.COOP.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model.float()

        print("Building Model")
        self.model = CustomCLIP(cfg, classnames, clip_model, self.device)

        print("Turning off gradients in both the image and the text encoder")
        for name, param in self.model.named_parameters():
            if "prompt_learner" not in name:
                param.requires_grad_(False)

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model.prompt_learner, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)

       

        # NOTE: only give prompt_learner to the optimizer
        self.optim = build_optimizer(self.model.prompt_learner, cfg.OPTIM)
      
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("prompt_learner", self.model.prompt_learner, self.optim, self.sched)

        self.scaler = GradScaler() if cfg.TRAINER.COOP.PREC == "amp" else None

        # Note that multi-gpu training could be slow because CLIP's size is
        # big, which slows down the copy operation in DataParallel
        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)


    def forward_backward(self, batch):

        self.model.noise_trigger.noise.requires_grad = True
        image, label, backdoor_tag = self.parse_batch_train(batch)  # image: [B, C, H, W]
        

        prec = self.cfg.TRAINER.COOP.PREC
        if prec == "amp":
            with autocast():
                output = self.model(image)
                loss = F.cross_entropy(output, label)

            self.optim.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optim)
            self.scaler.update()
        else:
            output = self.model(image, backdoor_tag)
            
            lambda_clean = 1.0
            lambda_adv = 1.0
            # lambda_reg = 0.0

            clean_exists = any(~backdoor_tag)
            backdoor_exists = any(backdoor_tag)

            if clean_exists :
                loss_clean = F.cross_entropy(output[~backdoor_tag], label[~backdoor_tag])
                
            if backdoor_exists : 
                loss_adv = F.cross_entropy(output[backdoor_tag], label[backdoor_tag]) 


            if clean_exists and backdoor_exists:
                loss = lambda_clean*loss_clean + lambda_adv*loss_adv 
            elif clean_exists and not backdoor_exists:
                loss = lambda_clean*loss_clean
            elif not clean_exists and backdoor_exists:
                loss = lambda_adv*loss_adv
            else:
                raise ValueError("No clean or backdoor images found. Check the backdoor tag assignments in Dataset class.")


            self.model_backward_and_update(loss)


            # update trigger noise
            if backdoor_exists:
                trigger_noise_grad  = self.model.noise_trigger.noise.grad.data
                self.model.noise_trigger.noise = self.model.noise_trigger.noise - trigger_noise_grad.sign()*0.01
                eps=self.cfg.BACKDOOR.NOISE_EPS/255.0
                self.model.noise_trigger.noise.clamp_(-eps,eps)
                self.model.noise_trigger.noise = self.model.noise_trigger.noise.detach()


        loss_summary = {
            "loss": loss.item(),
            "acc": compute_accuracy(output, label)[0].item(),
        }

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        input = input.to(self.device)
        label = label.to(self.device)
        backdoor_tag = batch["backdoor_tag"].to(self.device)
        return input, label, backdoor_tag==1

    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()
        
        # By default, the best model is loaded
        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)
        
        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError('Model not found at "{}"'.format(model_path))

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]

            # Ignore fixed token vectors
            if "token_prefix" in state_dict:
                del state_dict["token_prefix"]

            if "token_suffix" in state_dict:
                del state_dict["token_suffix"]

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=False)
