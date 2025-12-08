import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models import create_model
from einops import rearrange, repeat

# Cross-Attention Module
class CrossAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.query_proj = nn.Linear(dim, dim)
        self.key_proj = nn.Linear(dim, dim)
        self.value_proj = nn.Linear(dim, dim)
        self.softmax = nn.Softmax(dim=-1)
        self.scale = dim ** -0.5

    def forward(self, query, key_value):
        B, N, C = query.shape
        Q = self.query_proj(query)
        K = self.key_proj(key_value)
        V = self.value_proj(key_value)

        attn = self.softmax((Q @ K.transpose(-2, -1)) * self.scale)
        fused = attn @ V
        return fused

# SCFM Module
class SCFM(nn.Module):
    def __init__(self, swin_dim, contrast_dim):
        super().__init__()
        self.proj_contrast = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(1),
            nn.Linear(contrast_dim, swin_dim),
            nn.ReLU(),
            nn.Linear(swin_dim, swin_dim)
        )
        self.cross_attn = CrossAttention(swin_dim)

    def forward(self, swin_feat, f_contrast):
        B, H, W, C = swin_feat.shape   
        
        swin_feat_flat = swin_feat.flatten(1, 2)  
        
        if f_contrast.dim() == 2:
            f_contrast = f_contrast.unsqueeze(-1).unsqueeze(-1)  
        f_proj = self.proj_contrast(f_contrast).unsqueeze(1)  
        
        fused = self.cross_attn(swin_feat_flat, f_proj) + swin_feat_flat
        
        fused = fused.transpose(1, 2).reshape(B, C, H, W)  
        return fused

# DWFM Module
class DWFM(nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.align_convs = nn.ModuleList([
            nn.Conv2d(d, dims[0], kernel_size=1) for d in dims
        ])
        self.fc = nn.Sequential(
            nn.Linear(len(dims) * dims[0], 128),
            nn.ReLU(),
            nn.Linear(128, len(dims))
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, features):
        for i, f in enumerate(features):
            if f.dim() != 4:
                raise ValueError(f"Feature {i} should be 4D tensor, got {f.dim()}D")
        
        resized_feats = [F.interpolate(conv(f), size=features[0].shape[2:], mode='bilinear', align_corners=False)
                         for conv, f in zip(self.align_convs, features)]
        
        z = [torch.mean(f, dim=(2, 3)) for f in resized_feats]
        z_concat = torch.cat(z, dim=1)
        weights = self.softmax(self.fc(z_concat))  # (B, 4)

        fused = sum(w.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * f
                    for w, f in zip(weights.unbind(1), resized_feats))
        return fused

class ClestModel(nn.Module):
    def __init__(self, contrast_dim=512, pretrained_path='pretrained/swin_tiny_patch4_window7_224.pth'):
        super().__init__()
        self.swin = create_model(
            'swin_tiny_patch4_window7_224',
            pretrained=False,
            num_classes=0,
            features_only=True
        )
        
        if pretrained_path:
            try:
                state_dict = torch.load(pretrained_path, map_location='cpu')
                if 'model' in state_dict:
                    state_dict = state_dict['model']
                state_dict = {k: v for k, v in state_dict.items() if not k.startswith('head.')}

            except Exception as e:
                print(f"加载预训练权重失败: {str(e)}")
                print("将使用随机初始化权重继续...")
        
        self.stage_dims = [96, 192, 384, 768]
        self.scfms = nn.ModuleList([
            SCFM(dim, contrast_dim) for dim in self.stage_dims
        ])
        self.dwfm = DWFM(self.stage_dims)

        self.regressor = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(self.stage_dims[0], 1)
        )

    def forward(self, x, f_contrast):
        features = self.swin(x)
        
        if f_contrast.dim() == 2:
            f_contrast = f_contrast.unsqueeze(-1).unsqueeze(-1)  # (B, C, 1, 1) 
        
        fused_feats = [scfm(f, f_contrast) for scfm, f in zip(self.scfms, features)]
        
        fused = self.dwfm(fused_feats)
        
        score = self.regressor(fused)
        
        return score.squeeze(-1)


if __name__ == "__main__":
    model = ClestModel(contrast_dim=512)
    dummy_img = torch.randn(2, 3, 224, 224)
    f_contrast = torch.randn(2, 512)  # 对比学习输出

    with torch.no_grad():
        output = model(dummy_img, f_contrast)
        print("Output shape:", output.shape)


