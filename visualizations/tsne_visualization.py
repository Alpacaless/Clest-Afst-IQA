import torch
from modules.network import get_network
from modules.CL_model import CONTRIQUE_model
from torchvision import transforms
import numpy as np
import os
import argparse
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from PIL import Image
from tqdm import tqdm

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# 加载模型
def load_model(args):
    encoder = get_network('resnet50', pretrained=False)
    model = CONTRIQUE_model(args, encoder, 2048)
    model.load_state_dict(torch.load(args.model_path, map_location=args.device.type))
    model = model.to(args.device)
    model.eval()
    return model

# 提取特征
def extract_features(model, image_paths, args):
    features = []
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet归一化
                            std=[0.229, 0.224, 0.225])
    ])
    
    # for path in tqdm(image_paths, desc="Extracting features"):
    #     image = Image.open(path).convert("RGB")
    #     image_tensor = transform(image).unsqueeze(0).to(args.device)
        
    #     with torch.no_grad():
    #         _, _, _, _, h_i, _, _, _ = model(image_tensor, image_tensor)  # 提取全局特征
    #         # print(h_i.shape)
    #         features.append(h_i.cpu().numpy())
            
    for path in tqdm(image_paths, desc="Extracting features"):
        image = Image.open(path)
        sz = image.size
        image_2 = image.resize((sz[0] // 2, sz[1] // 2))
        image_tensor = transform(image).unsqueeze(0).to(args.device)
        image_2_tensor = transform(image_2).unsqueeze(0).to(args.device)

        with torch.no_grad():
            _, _, _, _, h_i, h_2_i, _, _ = model(image_tensor, image_2_tensor)  # 提取全局特征
            # print(h_i.shape)
            feat = np.hstack((h_i.cpu().numpy(),\
                                h_2_i.cpu().numpy()))
            features.append(feat)
    
    return np.vstack(features)

# 生成t-SNE图
def plot_tsne(features, labels, save_path):
    # 特征归一化
    scaler = StandardScaler()
    features_normalized = scaler.fit_transform(features)
    
    # PCA降维到50维
    pca = PCA(n_components=50, random_state=42)
    features_pca = pca.fit_transform(features_normalized)
    
    # t-SNE降维到2维
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=10000, learning_rate=500, early_exaggeration=12)
    embedded_features = tsne.fit_transform(features_pca)
    
    # 可视化
    plt.figure(figsize=(10, 8))
    for i, label in enumerate(np.unique(labels)):
        plt.scatter(embedded_features[labels == label, 0],
                    embedded_features[labels == label, 1],
                    label=f"Class {label}", alpha=0.7)
    
    plt.legend()
    plt.title("t-SNE Visualization of Feature Representations")
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close()

def main(args):
    # 加载模型
    model = load_model(args)
    
    # 准备图像路径和标签
    image_paths = []
    labels = []
    
    # 假设图像按类别存储在不同文件夹中
    for i, class_name in enumerate(["real_distortion", "blur", 
                                    "fnoise", "jpeg", 
                                    "jpeg2000"]):
        class_dir = os.path.join(args.data_dir, class_name)
        class_images = [os.path.join(class_dir, f) for f in os.listdir(class_dir)[:150]]  # 每类取150张
        image_paths.extend(class_images)
        labels.extend([i] * len(class_images))
    
    # 提取特征
    features = extract_features(model, image_paths, args)
    
    # 生成t-SNE图
    plot_tsne(features, labels, args.tsne_save_path)
    print(f"t-SNE plot saved to {args.tsne_save_path}")

def parse_args():
    parser = argparse.ArgumentParser()
    
    # 模型路径
    parser.add_argument("--model_path", type=str,
                       default="models/CONTRIQUE_checkpoint25.tar",
                       help="Path to trained model")
    
    # 数据路径
    parser.add_argument("--data_dir", type=str,
                       default="dis_data",
                       help="Directory containing distorted images")
    
    # t-SNE图保存路径
    parser.add_argument("--tsne_save_path", type=str,
                       default="visualizations/t_sne_plot_level.png",
                       help="Path to save t-SNE plot")
    
    args = parser.parse_args()
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return args

if __name__ == "__main__":
    args = parse_args()
    main(args)