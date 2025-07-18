
# a) Prepare your pandas DataFrame
#  1. Split out image paths and tabular columns:

import pandas as pd

df = pd.read_csv("your_data.csv")
# assume columns: "image_path", numeric_cols…, categorical_cols…, and "label"
image_paths = df["image_path"].tolist()


# 2. Preprocess tabular features with scikit-learn transformers:
from sklearn.preprocessing import StandardScaler, OneHotEncoder

numeric_cols   = ["age", "income"]
categorical_cols = ["gender", "region"]

scaler  = StandardScaler()
encoder = OneHotEncoder(sparse=False, handle_unknown="ignore")

X_num = scaler.fit_transform(df[numeric_cols])
X_cat = encoder.fit_transform(df[categorical_cols])
X_tab = np.hstack([X_num, X_cat])  # shape (N, D_tab)




# 3. Encode labels for classification or leave continuous for regression:
# Classification  
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y_cls = le.fit_transform(df["label_cls"])

# Regression  
y_reg = df["label_reg"].values.astype(float)




# b) Define a PyTorch Dataset & DataLoader
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

class MixedDataset(Dataset):
    def __init__(self, image_paths, X_tab, y, is_regression=False):
        self.image_paths = image_paths
        self.X_tab        = torch.tensor(X_tab, dtype=torch.float32)
        self.y            = torch.tensor(y, dtype=torch.float32 if is_regression else torch.long)
        self.is_reg       = is_regression
        self.transform    = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[.5,.5,.5], std=[.5,.5,.5]),
        ])

    def __len__(self): return len(self.y)
    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        img = self.transform(img)
        tab = self.X_tab[idx]
        label = self.y[idx]
        return img, tab, label

# Example DataLoader
ds_cls = MixedDataset(image_paths, X_tab, y_cls, is_regression=False)
loader_cls = DataLoader(ds_cls, batch_size=32, shuffle=True)




# c) Build the Multi-Modal Model
import torch.nn as nn
import timm

class ViTWithTabular(nn.Module):
    def __init__(self, tab_dim, n_classes=None, is_regression=False):
        super().__init__()
        # 1) Pretrained ViT backbone
        self.vit = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=0)
        feat_dim = self.vit.num_features      # e.g., 768

        # 2) Tabular MLP
        self.tab_mlp = nn.Sequential(
            nn.Linear(tab_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )

        # 3) Head: classification or regression
        out_dim = 1 if is_regression else n_classes
        self.head = nn.Linear(feat_dim + 64, out_dim)
        self.is_reg = is_regression

    def forward(self, img, tab):
        img_feat = self.vit(img)          # (B, feat_dim)
        tab_feat = self.tab_mlp(tab)      # (B, 64)
        x = torch.cat([img_feat, tab_feat], dim=1)
        out = self.head(x)
        if not self.is_reg:
            return out                    # raw logits for CrossEntropyLoss
        return out.squeeze(1)             # continuous output

# Instantiate for classification
model_cls = ViTWithTabular(tab_dim=X_tab.shape[1],
                           n_classes=len(le.classes_),
                           is_regression=False)

# Instantiate for regression
model_reg = ViTWithTabular(tab_dim=X_tab.shape[1],
                           is_regression=True)



# d) Training Loop Sketch
import torch.optim as optim

def train_epoch(model, loader, is_regression=False):
    model.train()
    opt = optim.AdamW(model.parameters(), lr=2e-5)
    crit = nn.MSELoss() if is_regression else nn.CrossEntropyLoss()

    for img, tab, y in loader:
        pred = model(img, tab)
        loss = crit(pred, y)
        opt.zero_grad(); loss.backward(); opt.step()
