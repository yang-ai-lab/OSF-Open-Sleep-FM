import torch.nn as nn
from osf.backbone.vit1d_cls import vit_nano, vit_tiny, vit_small, vit_middle, vit_base, vit_large, vit_xl


class PSGModalityEncoderCLS(nn.Module):
    """
    Init helper for ViT with CLS token. No forward() - access .backbone directly.
    
    Used by DINO to initialize encoder, then DINO accesses self.encoders["all"].backbone.
    """
    def __init__(self, *,
                 encoder_name: str,
                 proj_out: int      = 256,
                 proj_hidden: int   = 512,
                 freq: int          = 64,
                 win_sec: int       = 30,
                 channel: int       = 12, 
                 lead_wise = 0,
                 patch_size = 40, 
                 patch_size_ch = 4,
                 is_proj_head = 1,
                ):
        super().__init__()
        token_len  = freq * win_sec

        self.token_len = token_len
        self.patch_size = patch_size

        if encoder_name == "vit_nano":
            self.backbone = vit_nano(num_leads=channel, seq_len=token_len, patch_size=patch_size, lead_wise=lead_wise, patch_size_ch=patch_size_ch)
        elif encoder_name == "vit_tiny":
            self.backbone = vit_tiny(num_leads=channel, seq_len=token_len, patch_size=patch_size, lead_wise=lead_wise, patch_size_ch=patch_size_ch)
        elif encoder_name == "vit_small":
            self.backbone = vit_small(num_leads=channel, seq_len=token_len, patch_size=patch_size, lead_wise=lead_wise, patch_size_ch=patch_size_ch)
        elif encoder_name == "vit_middle":
            self.backbone = vit_middle(num_leads=channel, seq_len=token_len, patch_size=patch_size, lead_wise=lead_wise, patch_size_ch=patch_size_ch)
        elif encoder_name == "vit_base":
            self.backbone = vit_base(num_leads=channel, seq_len=token_len, patch_size=patch_size, lead_wise=lead_wise, patch_size_ch=patch_size_ch)
        elif encoder_name == "vit_large":
            self.backbone = vit_large(num_leads=channel, seq_len=token_len, patch_size=patch_size, lead_wise=lead_wise, patch_size_ch=patch_size_ch)
        elif encoder_name == "vit_xl":
            self.backbone = vit_xl(num_leads=channel, seq_len=token_len, patch_size=patch_size, lead_wise=lead_wise, patch_size_ch=patch_size_ch)
        else:
            raise ValueError(f"Unknown encoder_name for CLS variant: {encoder_name}")

        d_model = self.backbone.width
        if is_proj_head == 1:
            self.proj_head = nn.Sequential(
                nn.Linear(d_model, proj_hidden),
                nn.LayerNorm(proj_hidden),
                nn.ReLU(inplace=True),
                nn.Linear(proj_hidden, proj_out),
                nn.LayerNorm(proj_out),
            )
        else:
            self.proj_head = None
