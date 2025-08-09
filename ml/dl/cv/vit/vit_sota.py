import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
import timm
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast


class MixedDatasetEfficient(Dataset):
    def __init__(self, image_paths, X_tab, y, is_regression=False, augment=True):
        self.image_paths = image_paths
        self.X_tab = torch.tensor(X_tab, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32 if is_regression else torch.long)
        self.is_reg = is_regression
        
        # Enhanced data augmentation for large-scale training
        if augment:
            self.transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=10),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                transforms.RandomErasing(p=0.1)
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        img = self.transform(img)
        tab = self.X_tab[idx]
        label = self.y[idx]
        return img, tab, label


class AttentionPooling(nn.Module):
    def __init__(self, input_dim, hidden_dim=512):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, x):
        # x: (batch_size, seq_len, input_dim)
        weights = F.softmax(self.attention(x), dim=1)
        return torch.sum(weights * x, dim=1)


class EfficientViTWithTabular(nn.Module):
    def __init__(self, tab_dim, n_classes=None, is_regression=False, dropout_rate=0.3):
        super().__init__()
        
        # Use DINOv2 ViT-L/14 as backbone for state-of-the-art image embeddings
        self.vit = timm.create_model("vit_large_patch14_dinov2.lvd142m", 
                                   pretrained=True, 
                                   num_classes=0)
        feat_dim = self.vit.num_features  # 1024 for ViT-L
        
        # Enhanced tabular processing with batch normalization
        self.tab_mlp = nn.Sequential(
            nn.Linear(tab_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_rate / 2)
        )
        
        # Cross-modal attention mechanism
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=feat_dim, 
            num_heads=16, 
            dropout=dropout_rate,
            batch_first=True
        )
        
        # Feature fusion with residual connections
        self.fusion_layer = nn.Sequential(
            nn.Linear(feat_dim + 128, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate / 2)
        )
        
        # Output head with proper initialization
        out_dim = 1 if is_regression else n_classes
        self.head = nn.Linear(256, out_dim)
        self.is_reg = is_regression
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, img, tab):
        # Extract image features with DINOv2
        img_feat = self.vit(img)  # (B, feat_dim)
        
        # Process tabular features
        tab_feat = self.tab_mlp(tab)  # (B, 128)
        
        # Prepare for cross-attention
        img_feat_expanded = img_feat.unsqueeze(1)  # (B, 1, feat_dim)
        tab_feat_expanded = tab_feat.unsqueeze(1)  # (B, 1, 128)
        
        # Apply cross-modal attention
        attended_img, _ = self.cross_attention(
            img_feat_expanded, 
            tab_feat_expanded, 
            tab_feat_expanded
        )
        attended_img = attended_img.squeeze(1)  # (B, feat_dim)
        
        # Residual connection
        img_feat = img_feat + attended_img
        
        # Fusion
        fused = torch.cat([img_feat, tab_feat], dim=1)
        fused_feat = self.fusion_layer(fused)
        
        # Output
        out = self.head(fused_feat)
        
        if not self.is_reg:
            return out  # Raw logits for CrossEntropyLoss
        return out.squeeze(1)  # Continuous output


class EfficientTrainer:
    def __init__(self, model, train_loader, val_loader, is_regression=False, device='cuda'):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.is_regression = is_regression
        self.device = device
        
        # Mixed precision training for efficiency
        self.scaler = GradScaler()
        
        # Optimizer with different learning rates for backbone and head
        backbone_params = list(self.model.vit.parameters())
        other_params = [p for p in self.model.parameters() if p not in backbone_params]
        
        self.optimizer = optim.AdamW([
            {'params': backbone_params, 'lr': 1e-6},  # Lower LR for pretrained backbone
            {'params': other_params, 'lr': 3e-4}     # Higher LR for new layers
        ], weight_decay=1e-4)
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=10, T_mult=2, eta_min=1e-7
        )
        
        # Loss function
        self.criterion = nn.MSELoss() if is_regression else nn.CrossEntropyLoss(label_smoothing=0.1)
        
        # Early stopping
        self.best_val_loss = float('inf')
        self.patience = 10
        self.patience_counter = 0
    
    def train_epoch(self):
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        for img, tab, y in self.train_loader:
            img, tab, y = img.to(self.device), tab.to(self.device), y.to(self.device)
            
            self.optimizer.zero_grad()
            
            with autocast():
                pred = self.model(img, tab)
                loss = self.criterion(pred, y)
            
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            total_loss += loss.item()
            num_batches += 1
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        self.scheduler.step()
        return total_loss / num_batches
    
    def validate(self):
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for img, tab, y in self.val_loader:
                img, tab, y = img.to(self.device), tab.to(self.device), y.to(self.device)
                
                with autocast():
                    pred = self.model(img, tab)
                    loss = self.criterion(pred, y)
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        
        # Early stopping check
        if avg_loss < self.best_val_loss:
            self.best_val_loss = avg_loss
            self.patience_counter = 0
        else:
            self.patience_counter += 1
        
        return avg_loss, self.patience_counter >= self.patience
    
    def train(self, epochs=100):
        for epoch in range(epochs):
            train_loss = self.train_epoch()
            val_loss, should_stop = self.validate()
            
            print(f"Epoch {epoch+1}/{epochs}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            if should_stop:
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        return self.model


# Example usage for large-scale training
def create_model_for_large_scale(X_tab_shape, n_classes=None, is_regression=False):
    """
    Create an optimized model for training on 3M+ datapoints
    """
    model = EfficientViTWithTabular(
        tab_dim=X_tab_shape[1],
        n_classes=n_classes,
        is_regression=is_regression,
        dropout_rate=0.3
    )
    return model


def setup_data_loaders(image_paths, X_tab, y, is_regression=False, batch_size=64):
    """
    Setup efficient data loaders with proper train/val split
    """
    from sklearn.model_selection import train_test_split
    
    # Split data
    X_train_paths, X_val_paths, X_train_tab, X_val_tab, y_train, y_val = train_test_split(
        image_paths, X_tab, y, test_size=0.2, random_state=42, stratify=y if not is_regression else None
    )
    
    # Create datasets
    train_dataset = MixedDatasetEfficient(X_train_paths, X_train_tab, y_train, is_regression, augment=True)
    val_dataset = MixedDatasetEfficient(X_val_paths, X_val_tab, y_val, is_regression, augment=False)
    
    # Create data loaders with efficient settings
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=8, 
        pin_memory=True,
        persistent_workers=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=4, 
        pin_memory=True,
        persistent_workers=True
    )
    
    return train_loader, val_loader


# Complete pipeline example
if __name__ == "__main__":
    # Load and preprocess data (same as original)
    df = pd.read_csv("your_data.csv")
    image_paths = df["image_path"].tolist()
    
    # Preprocess tabular features
    numeric_cols = ["age", "income"]
    categorical_cols = ["gender", "region"]
    
    scaler = StandardScaler()
    encoder = OneHotEncoder(sparse=False, handle_unknown="ignore")
    
    X_num = scaler.fit_transform(df[numeric_cols])
    X_cat = encoder.fit_transform(df[categorical_cols])
    X_tab = np.hstack([X_num, X_cat])
    
    # Encode labels
    le = LabelEncoder()
    y_cls = le.fit_transform(df["label_cls"])
    
    # Setup model and training
    model = create_model_for_large_scale(X_tab.shape, n_classes=len(le.classes_), is_regression=False)
    train_loader, val_loader = setup_data_loaders(image_paths, X_tab, y_cls, batch_size=32)
    
    # Train model
    trainer = EfficientTrainer(model, train_loader, val_loader, is_regression=False)
    trained_model = trainer.train(epochs=50)