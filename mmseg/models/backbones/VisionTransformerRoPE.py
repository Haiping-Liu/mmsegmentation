import torch.nn as nn
import torch
from mmengine.model import BaseModule
from mmseg.registry import MODELS


from models.mae_rope import MAE_RoPE, MAEEncoder
from models.vit import create_vit
from configs.utils import load_config_from_yaml
from RoPE.standard_rope import compute_axial_cis

@MODELS.register_module()
class VisionTransformerRoPE(BaseModule):
    def __init__(self,
                 config_path: str,
                 encoder_type: str = 'mae',
                 out_indices: tuple = (3, 5, 7, 11),
                 img_size:int = 512,
                 return_cls_token: bool = False,
                 return_feat_only: bool = False,
                 norm_eval: bool = True,
                 init_cfg: dict = None,
                 ckpt_path: str = None):
        
        super().__init__(init_cfg=init_cfg)

        config = load_config_from_yaml(config_path)
        config.encoder.img_size = img_size

        H = W = config.encoder.img_size // config.encoder.patch_size
        head_dim = config.encoder.dim // config.encoder.n_heads
        freqs_cis = compute_axial_cis(head_dim, H, W)

        state_dict = torch.load(ckpt_path, map_location="cpu")['state_dict']
        clean_state_dict = {
            k.replace("model.", ""): v for k, v in state_dict.items() if k.startswith("model.")
        }

        encoder = MAEEncoder(freqs_cis, config.encoder)
        encoder.load_state_dict(clean_state_dict, strict=False)

        self.norm_eval = norm_eval
        self.out_indices = out_indices
        self.return_cls_token = return_cls_token
        self.return_feat_only = return_feat_only

        if encoder_type == 'mae':
            self.patch_embed = encoder.patch_embed
            self.blocks = encoder.blocks
            self.cls_token = encoder.cls_token
            self.norm = encoder.norm
        else:
            vit_rope = create_vit(config)
            self.patch_embed = vit_rope.patch_embed
            self.blocks = vit_rope.blocks
            self.cls_token = vit_rope.cls_token
            self.norm = vit_rope.norm

    def forward(self, x):
        """
        Args:
            x: [B, 3, H, W]
        Returns:
            - tuple of feature maps (seg/det)
            - or cls token vector (cls)
            - or final feature (raw encoder output)
        """
        outs = []
        x = self.patch_embed(x)  # B, L, C
        B, L, C = x.shape

        cls_token = self.cls_token.expand(B, -1, -1)  # [B, 1, C]
        x = torch.cat((cls_token, x), dim=1)

        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i in self.out_indices and not self.return_cls_token:
                out = self.norm(x[:, 1:])  # Remove cls token
                B, L, C = out.shape
                H = W = int(L ** 0.5)
                out = out.reshape(B, H, W, C).permute(0, 3, 1, 2)  # BCHW
                outs.append(out)

        if self.return_cls_token:
            x = self.norm(x)
            return x[:, 0]  # [B, C]

        if self.return_feat_only:
            return x  # [B, L+1, C]

        return tuple(outs)

    def train(self, mode=True):
        super().train(mode)
        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, nn.LayerNorm):
                    m.eval()



        