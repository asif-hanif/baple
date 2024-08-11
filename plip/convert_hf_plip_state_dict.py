import torch


def sanity_check():
    clip_model_path = "/home/asif.hanif/.cache/clip/ViT-B-32.pt"  ## path of original clip model weights
    clip_state_dict_actual = torch.jit.load(clip_model_path, map_location="cpu").state_dict()
    for key in clip_state_dict_actual.keys():
        if  clip_state_dict_actual[key].shape != clip_state_dict[key].shape:
            print(key)
            print(clip_state_dict_actual[key].shape)
            print(clip_state_dict[key].shape)
            print("\n\n")


plip_model_path = "./plip_model.pt"   # path of plip model weights (converted from huggingface model)
plip_state_dict = torch.load(plip_model_path, map_location="cpu")

clip_state_dict = {}
plip_keys_2_clip_keys = {}


for key in plip_state_dict.keys():

    new_key = key

    if "vision_model.encoder.layers." in new_key:
        new_key = new_key.replace("vision_model.encoder.layers.", "visual.transformer.resblocks.")
    

    if "text_model.encoder.layers." in new_key:
        new_key = new_key.replace("text_model.encoder.layers.", "transformer.resblocks.")

    if "self_attn." in new_key:
        new_key = new_key.replace("self_attn.", "attn.")


    if "q_proj.weight" in new_key:
        new_key = new_key.replace("q_proj.weight", "in_proj_weight")
    if "q_proj.bias" in new_key:
        new_key = new_key.replace("q_proj.bias", "in_proj_bias")
    

    attn_keys_to_check = ["k_proj.weight", "k_proj.bias", "v_proj.weight", "v_proj.bias"]
    if any(attn_key in new_key for attn_key in attn_keys_to_check): continue
 

    if "layer_norm1.weight" in new_key:
        new_key = new_key.replace("layer_norm1.weight", "ln_1.weight")
    if "layer_norm1.bias" in new_key:
        new_key = new_key.replace("layer_norm1.bias", "ln_1.bias")


    if "layer_norm2.weight" in new_key:
        new_key = new_key.replace("layer_norm2.weight", "ln_2.weight")
    if "layer_norm2.bias" in new_key:
        new_key = new_key.replace("layer_norm2.bias", "ln_2.bias")


    if "fc1.weight" in new_key:
        new_key = new_key.replace("fc1.weight", "c_fc.weight")
    if "fc1.bias" in new_key:
        new_key = new_key.replace("fc1.bias", "c_fc.bias")


    if "fc2.weight" in new_key:
        new_key = new_key.replace("fc2.weight", "c_proj.weight")
    if "fc2.bias" in new_key:
        new_key = new_key.replace("fc2.bias", "c_proj.bias")


    if "text_model.final_layer_norm.weight" in new_key:
        new_key = new_key.replace("text_model.final_layer_norm.weight", "ln_final.weight")
    if "text_model.final_layer_norm.bias" in new_key:
        new_key = new_key.replace("text_model.final_layer_norm.bias", "ln_final.bias")

    if "vision_model.embeddings.class_embedding" in new_key:
        new_key = new_key.replace("vision_model.embeddings.class_embedding", "visual.class_embedding")

    if "vision_model.pre_layrnorm.weight" in new_key:
        new_key = new_key.replace("vision_model.pre_layrnorm.weight", "visual.ln_pre.weight")
    if "vision_model.pre_layrnorm.bias" in new_key:
        new_key = new_key.replace("vision_model.pre_layrnorm.bias", "visual.ln_pre.bias")

    if "vision_model.post_layernorm.weight" in new_key:
        new_key = new_key.replace("vision_model.post_layernorm.weight", "visual.ln_post.weight")
    if "vision_model.post_layernorm.bias" in new_key:
        new_key = new_key.replace("vision_model.post_layernorm.bias", "visual.ln_post.bias")

    if "vision_model.embeddings.patch_embedding.weight" in new_key:
        new_key = new_key.replace("vision_model.embeddings.patch_embedding.weight", "visual.conv1.weight")

    if "vision_model.embeddings.class_embedding" in new_key:
        new_key = new_key.replace("vision_model.embeddings.class_embedding", "visual.class_embedding")

    if "vision_model.embeddings.position_embedding.weight" in new_key:
        new_key = new_key.replace("vision_model.embeddings.position_embedding.weight", "visual.positional_embedding")


    if "visual_projection.weight" in new_key:
        new_key = new_key.replace("visual_projection.weight", "visual.proj")

    if "text_projection.weight" in new_key:
        new_key = new_key.replace("text_projection.weight", "text_projection")

    if "text_model.embeddings.token_embedding.weight" in new_key:
        new_key = new_key.replace("text_model.embeddings.token_embedding.weight", "token_embedding.weight")

    if "text_model.embeddings.position_embedding.weight" in  new_key:
        new_key = new_key.replace("text_model.embeddings.position_embedding.weight", "positional_embedding")


    if "logit_scale" in new_key:
        pass

    plip_keys_2_clip_keys[key] = new_key

 

for key in plip_state_dict.keys():
    
    attn_keys_to_check = ["k_proj.weight", "k_proj.bias", "v_proj.weight", "v_proj.bias"]
    if any(attn_key in key for attn_key in attn_keys_to_check): continue
 

    if "self_attn.q_proj.weight" in key or "self_attn.q_proj.bias" in key:
        weight = torch.cat([plip_state_dict[key], plip_state_dict[key.replace('q_proj', 'k_proj')], plip_state_dict[key.replace('q_proj', 'v_proj')] ], dim=0)
        clip_state_dict[plip_keys_2_clip_keys[key]] = weight
    elif key == "visual_projection.weight":
        clip_state_dict[plip_keys_2_clip_keys[key]] = plip_state_dict[key].T
    elif key == "text_projection.weight":
        clip_state_dict[plip_keys_2_clip_keys[key]] = plip_state_dict[key].T
    else:   
        clip_state_dict[plip_keys_2_clip_keys[key]] = plip_state_dict[key]


clip_state_dict['input_resolution'] = torch.tensor(224)
clip_state_dict['context_length'] = torch.tensor(77)
clip_state_dict['vocab_size'] = torch.tensor(49408) 

sanity_check()
    
# breakpoint()  
del clip_state_dict["vision_model.embeddings.position_ids"]
del clip_state_dict["text_model.embeddings.position_ids"]
torch.save(clip_state_dict, "/l/users/asif.hanif/pre-trained-models/med-adv-prompt/plip/plip_model_converted.pt")


print("Done")



       