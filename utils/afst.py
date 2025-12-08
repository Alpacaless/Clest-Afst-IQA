import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models import create_model
from einops import rearrange
import math

class CrossAttention(nn.Module):
    def __init__(self, query_dim, key_value_dim):
        super().__init__()
        
        self.query_proj = nn.Linear(query_dim, key_value_dim)
        self.key_proj = nn.Linear(key_value_dim, key_value_dim)
        self.value_proj = nn.Linear(key_value_dim, key_value_dim)
        self.softmax = nn.Softmax(dim=-1)
        self.scale = key_value_dim ** -0.5

    def forward(self, query, key_value):
        B, N, C = query.shape
        B, M, D = key_value.shape
        
        Q = self.query_proj(query)  # [B, N, D]
        K = self.key_proj(key_value)  # [B, M, D]
        V = self.value_proj(key_value)  # [B, M, D]

        attn = self.softmax((Q @ K.transpose(-2, -1)) * self.scale)  # [B, N, M]
        fused = attn @ V  # [B, N, D]
        return fused

class MultiCognitiveAdapter(nn.Module):
    def __init__(self, dim):
        super().__init__()
        
        self.down_proj = nn.Conv2d(dim, dim//2, kernel_size=1)
        self.norm1 = nn.LayerNorm(dim//2)
        
        
        self.dwconv3 = nn.Conv2d(dim//2, dim//2, kernel_size=3, padding=1, groups=dim//2)
        self.dwconv5 = nn.Conv2d(dim//2, dim//2, kernel_size=5, padding=2, groups=dim//2)
        self.dwconv7 = nn.Conv2d(dim//2, dim//2, kernel_size=7, padding=3, groups=dim//2)
        
        
        self.conv1x1 = nn.Conv2d(dim//2, dim//2, kernel_size=1)
        self.norm2 = nn.LayerNorm(dim//2)
        self.up_proj = nn.Conv2d(dim//2, dim, kernel_size=1)
        
        self.gelu = nn.GELU()

    def forward(self, x):
        
        x = x.permute(0, 3, 1, 2).contiguous()
        B, C, H, W = x.shape
        residual = x
        
        
        x = self.down_proj(x)
        Hx, Wx = x.shape[2], x.shape[3]
        
        
        x_flat = x.flatten(2).transpose(1, 2)  
        x_norm = self.norm1(x_flat)
        x_norm = x_norm.transpose(1, 2).reshape(B, -1, Hx, Wx)  
        
        
        x3 = self.dwconv3(x_norm)
        x5 = self.dwconv5(x_norm)
        x7 = self.dwconv7(x_norm)
        
        
        x_avg = (x3 + x5 + x7) / 3
        x_avg = x_avg + x_norm
        
        
        x_conv = self.conv1x1(x_avg)
        x_conv = x_conv + x_avg
        x_conv_flat = x_conv.flatten(2).transpose(1, 2)  # [B, HW, C//2]
        x_conv_norm = self.norm2(x_conv_flat)
        x_conv_norm = self.gelu(x_conv_norm)
        x_conv_norm = x_conv_norm.transpose(1, 2).reshape(B, -1, Hx, Wx)  # [B, C//2, Hx, Wx]
        x_out = self.up_proj(x_conv_norm)
        
        
        output = residual + x_out
        
        
        output = output.permute(0, 2, 3, 1).contiguous()
        return output

class GroupedDilatedConvModule(nn.Module):
    def __init__(self, dim, groups=4, dilation_rates=[1, 3, 5, 7]):
        super().__init__()
        self.groups = groups
        self.dim_per_group = dim // groups
        self.dilation_rates = dilation_rates
        
        self.group_processing = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(self.dim_per_group, self.dim_per_group, kernel_size=1),
                nn.Conv2d(self.dim_per_group, self.dim_per_group, kernel_size=3, 
                          padding=r, dilation=r, groups=self.dim_per_group),
                nn.BatchNorm2d(self.dim_per_group),
                nn.ReLU()
            )
            for r in dilation_rates
        ])
        
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.attention = nn.Sequential(
            nn.Linear(self.dim_per_group * (len(dilation_rates) + 1), 128),
            nn.ReLU(),
            nn.Linear(128, len(dilation_rates) + 1),
            nn.Softmax(dim=1)
        )
        
    def forward(self, x):
        x = x.permute(0, 3, 1, 2).contiguous()
        B, C, H, W = x.shape
        
        group_features = torch.chunk(x, self.groups, dim=1)
        output_groups = []
        
        for g in range(self.groups):
            feat = group_features[g]
            Bg, Cg, Hg, Wg = feat.shape
            
            scale_features = [module(feat) for module in self.group_processing]
            
            global_feat = self.global_pool(feat).expand_as(feat)
            scale_features.append(global_feat)
            
            concat_feat = torch.cat(scale_features, dim=1)  # [B, Cg*(len+1), H, W]
            
            attn_input = torch.cat([self.global_pool(f).flatten(1) for f in scale_features], dim=1)
            attn_weights = self.attention(attn_input)  # [B, len+1]
            
            weighted_feats = []
            for i, (f, w) in enumerate(zip(scale_features, attn_weights.split(1, dim=1))):
                weighted_feats.append(f * w.view(B, 1, 1, 1))
            
            merged_feat = sum(weighted_feats)
            output_groups.append(merged_feat)
        
        output = torch.cat(output_groups, dim=1)
        
        output = x + output
        
        output = output.permute(0, 2, 3, 1).contiguous()
        return output

class DynamicWeightedFusionModule(nn.Module):
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
        features = [f.permute(0, 3, 1, 2).contiguous() for f in features]
        
        B = features[0].shape[0]
        target_size = features[0].shape[2:]
        resized_feats = []
        
        for i, (conv, feat) in enumerate(zip(self.align_convs, features)):
            feat = conv(feat)
            feat = F.interpolate(feat, size=target_size, mode='bilinear', align_corners=False)
            resized_feats.append(feat)
        
        z = [torch.mean(f, dim=(2, 3), keepdim=True) for f in resized_feats]
        z_flat = [f.flatten(1) for f in z]
        z_concat = torch.cat(z_flat, dim=1)  
        
        weights = self.softmax(self.fc(z_concat))  
        
        weights = weights.unsqueeze(-1).unsqueeze(-1)  
        fused = sum(w * f for w, f in zip(weights.split(1, dim=1), resized_feats))
        
        fused = fused.permute(0, 2, 3, 1).contiguous()
        return fused

class AFSTMIQA(nn.Module):
    def __init__(self, contrast_dim=4096, pretrained_path='pretrained/swin_tiny_patch4_window7_224.pth'):
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
                self.swin.load_state_dict(state_dict, strict=False)
            except Exception as e:
                print(f"加载预训练权重失败: {e}")
        
        self.stage_dims = [96, 192, 384, 768]
        
        self.adapters = nn.ModuleList([
            MultiCognitiveAdapter(dim) for dim in self.stage_dims
        ])
        
        self.gdcems = nn.ModuleList([
            GroupedDilatedConvModule(dim) for dim in self.stage_dims
        ])
        
        self.contrast_proj = nn.Sequential(
            nn.Linear(contrast_dim, self.stage_dims[-1]),
            nn.ReLU(),
            nn.Linear(self.stage_dims[-1], self.stage_dims[-1])
        )
        
        self.feature_proj = nn.Conv2d(self.stage_dims[0], self.stage_dims[-1], kernel_size=1)
        
        self.cross_attn = CrossAttention(
            query_dim=self.stage_dims[-1],  
            key_value_dim=self.stage_dims[-1]  
        )
        
        self.dwfm = DynamicWeightedFusionModule(self.stage_dims)
        
        self.regressor = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(self.stage_dims[-1], 1)
        )

    def forward(self, x, contrast_feat):
        swin_features = self.swin(x)
        
        adapter_features = []
        for i, (feat, adapter, gdcem) in enumerate(zip(swin_features, self.adapters, self.gdcems)):
            feat = adapter(feat)
            feat = gdcem(feat)
            adapter_features.append(feat)
        
        fused_features = self.dwfm(adapter_features)
        
        contrast_feat = contrast_feat.unsqueeze(-1).unsqueeze(-1)
        contrast_feat = self.contrast_proj(contrast_feat.flatten(1))
        contrast_feat = contrast_feat.unsqueeze(1)  

        B, H, W, C = fused_features.shape
        
        fused_features = fused_features.permute(0, 3, 1, 2).contiguous()  
        fused_features = self.feature_proj(fused_features)  
        fused_flat = fused_features.flatten(2).permute(0, 2, 1)  
        
        cross_fused = self.cross_attn(fused_flat, contrast_feat)  
        cross_fused = cross_fused.permute(0, 2, 1).reshape(B, -1, H, W)  
        
        score = self.regressor(cross_fused)
        
        return score.squeeze(1)
    

def test_afst_model():
    model = AFSTMIQA(contrast_dim=4096)
    model.eval()
    
    dummy_img = torch.randn(2, 3, 224, 224)
    contrast_feat = torch.randn(2, 4096)
    
    with torch.no_grad():
        output = model(dummy_img, contrast_feat)
    
    print(f"Output Shape: {output.shape}")

if __name__ == "__main__":
    test_afst_model()


