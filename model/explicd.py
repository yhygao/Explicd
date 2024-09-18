import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from open_clip import create_model_from_pretrained, get_tokenizer
from torchvision import transforms
from .utils import FFN

import pdb



class ExpLICD(nn.Module):  
    def __init__(self, concept_list, model_name='biomedclip', config=None):
        super().__init__()
            
        self.concept_list = concept_list
        self.model_name = model_name
        self.config = config
       
        if self.model_name in ['biomedclip', 'openclip']:
            if self.model_name == 'biomedclip':
                self.model, preprocess = create_model_from_pretrained('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
                self.tokenizer = get_tokenizer('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
            elif self.model_name == 'openclip':
                self.model, preprocess = create_model_from_pretrained('hf-hub:laion/CLIP-ViT-L-14-laion2B-s32B-b82K')
                self.tokenizer = get_tokenizer('hf-hub:laion/CLIP-ViT-L-14-laion2B-s32B-b82K')
               
                #self.model, preprocess = create_model_from_pretrained('hf-hub:laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K')
                #self.tokenizer = get_tokenizer('hf-hub:laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K')
            
            config.preprocess = preprocess
            
            self.model.cuda()
            
            concept_keys = list(concept_list.keys())

            self.concept_token_dict = {}
            for key in concept_keys:
                if config.dataset == 'isic2018':
                    prefix = f"this is a dermoscopic image, the {key} of the lesion is "
                attr_concept_list = concept_list[key]
                prefix_attr_concept_list = [prefix + concept for concept in attr_concept_list]
                tmp_concept_text = self.tokenizer(prefix_attr_concept_list).cuda()
                _, tmp_concept_feats, logit_scale = self.model(None, tmp_concept_text)
                self.concept_token_dict[key] = tmp_concept_feats.detach()

            
            self.logit_scale = logit_scale.detach()
        
        self.visual_features = []

        self.hook_list = []
        def hook_fn(module, input, output):
            self.visual_features.append(output) # detach to aboid saving computation graph
                                                 # might need to remove if finetune the full model
        layers = [self.model.visual.trunk.blocks[11]]
        for layer in layers:
            self.hook_list.append(layer.register_forward_hook(hook_fn))
        
        
        self.visual_tokens = nn.Parameter(nn.init.xavier_uniform_(torch.zeros(7, 768)))

        self.cross_attn = nn.MultiheadAttention(embed_dim=768, num_heads=12, batch_first=True)
        self.ffn = FFN(768, 768*4)
        self.norm = nn.LayerNorm(768)
        self.proj = nn.Linear(in_features=768, out_features=512, bias=False)

        self.cls_head = nn.Linear(in_features=34, out_features=config.num_class)

        for param in self.model.text.parameters():
            param.requires_grad = False
        for param in self.model.visual.trunk.parameters():
            param.requires_grad = True
        
        self.visual_tokens.requires_grad = True
    
    def get_backbone_params(self):
        return self.model.visual.trunk.parameters()
    def get_bridge_params(self):
        param_list = []
        
        param_list.append(self.visual_tokens)
        for param in self.cross_attn.parameters():
            param_list.append(param)
        for param in self.ffn.parameters():
            param_list.append(param)
        for param in self.norm.parameters():
            param_list.append(param)
        for param in self.proj.parameters():
            param_list.append(param)
        for param in self.cls_head.parameters():
            param_list.append(param)


        #param_list.append(self.ffn.parameters())
        #param_list.append(self.norm.parameters())
        #param_list.append(self.proj.parameters())
        #param_list.append(self.cls_head.parameters())

        return param_list


    def forward(self, imgs):
        
        self.visual_features.clear()
        #with torch.no_grad():
        #    img_feats, _, _ = self.model(imgs, None)
        img_feats, _, _ = self.model(imgs, None)
        img_feat_map = self.visual_features[0][:, 1:, :]

        B, _, _ = img_feat_map.shape
        visual_tokens = self.visual_tokens.repeat(B, 1, 1)

        agg_visual_tokens, _ = self.cross_attn(visual_tokens, img_feat_map, img_feat_map)
        agg_visual_tokens = self.proj(self.norm(self.ffn(agg_visual_tokens)))
        
        agg_visual_tokens = F.normalize(agg_visual_tokens, dim=-1)

        image_logits_dict = {}
        idx = 0
        for key in self.concept_token_dict.keys():
            image_logits_dict[key] = (self.logit_scale * agg_visual_tokens[:, idx:idx+1, :] @ self.concept_token_dict[key].repeat(B, 1, 1).permute(0, 2, 1)).squeeze(1)
            idx += 1
        

        image_logits_list = []
        for key in image_logits_dict.keys():
            image_logits_list.append(image_logits_dict[key])
        
        image_logits = torch.cat(image_logits_list, dim=-1)
        cls_logits = self.cls_head(image_logits) 

        return cls_logits, image_logits_dict



