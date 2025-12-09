"""
Self-Supervised Emotion Representation Learning
================================================

Main pipeline for learning emotion-aware representations using contrastive learning.

Architecture Overview:
---------------------
1. Temporal Contrastive Learning on video sequences
2. Identity-preserving emotion disentanglement
3. Cross-dataset generalization
4. Downstream task evaluation (classification, retrieval, generation)

Key Innovation: We learn representations where:
- Spatial proximity = same identity
- Directional vectors = emotion transformations
- Enables controllable emotion manipulation in latent space
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import cv2
from PIL import Image
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns


class EmotionEncoder(nn.Module):
    """
    Encoder network for learning emotion-disentangled representations.
    
    Architecture:
    - Backbone: ResNet50 (pre-trained on ImageNet)
    - Projection head: 2048 -> 512 -> 128 (emotion embedding)
    - Identity head: 2048 -> 256 (identity embedding)
    """
    
    def __init__(self, embedding_dim: int = 128, identity_dim: int = 256):
        super().__init__()
        
        # Backbone
        resnet = models.resnet50(pretrained=True)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])  # Remove FC layer
        self.feature_dim = 2048
        
        # Emotion projection head (for contrastive learning)
        self.emotion_projector = nn.Sequential(
            nn.Linear(self.feature_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, embedding_dim)
        )
        
        # Identity head (for preserving identity)
        self.identity_projector = nn.Sequential(
            nn.Linear(self.feature_dim, identity_dim),
            nn.BatchNorm1d(identity_dim),
            nn.ReLU(inplace=True)
        )
        
        # Optional: Emotion predictor (for supervised auxiliary task)
        self.emotion_predictor = nn.Linear(embedding_dim, 7)  # 7 basic emotions
        
    def forward(self, x: torch.Tensor, return_all: bool = False):
        """
        Args:
            x: Input images [B, 3, H, W]
            return_all: If True, return features, emotion_emb, identity_emb
        
        Returns:
            emotion_emb: Emotion embeddings [B, embedding_dim]
            (optional) features, identity_emb
        """
        # Extract features
        features = self.backbone(x)
        features = features.view(features.size(0), -1)  # [B, 2048]
        
        # Project to emotion and identity spaces
        emotion_emb = self.emotion_projector(features)  # [B, 128]
        emotion_emb = F.normalize(emotion_emb, dim=1)  # L2 normalize
        
        if return_all:
            identity_emb = self.identity_projector(features)  # [B, 256]
            identity_emb = F.normalize(identity_emb, dim=1)
            return emotion_emb, identity_emb, features
        
        return emotion_emb


class TemporalContrastiveLoss(nn.Module):
    """
    Contrastive loss for temporal sequences.
    
    Key idea:
    - Positive pairs: Different frames of same person (same identity, different emotions)
    - Negative pairs: Frames from different people
    - We want: close in space = same identity, direction = emotion change
    """
    
    def __init__(self, temperature: float = 0.07, identity_weight: float = 1.0):
        super().__init__()
        self.temperature = temperature
        self.identity_weight = identity_weight
        
    def forward(self, 
                emotion_emb: torch.Tensor, 
                identity_emb: torch.Tensor,
                identity_labels: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        Args:
            emotion_emb: [B, D_emotion] - L2 normalized
            identity_emb: [B, D_identity] - L2 normalized
            identity_labels: [B] - identity labels for each sample
        
        Returns:
            loss: Scalar loss
            metrics: Dictionary of metrics for logging
        """
        batch_size = emotion_emb.size(0)
        device = emotion_emb.device
        
        # Compute similarity matrices
        emotion_sim = torch.matmul(emotion_emb, emotion_emb.T) / self.temperature  # [B, B]
        identity_sim = torch.matmul(identity_emb, identity_emb.T)  # [B, B]
        
        # Create masks for positive and negative pairs
        identity_mask = identity_labels.unsqueeze(1) == identity_labels.unsqueeze(0)  # [B, B]
        identity_mask.fill_diagonal_(False)  # Remove self-comparisons
        
        # Positive pairs: same identity, different frames
        positives_mask = identity_mask.float()
        
        # Negative pairs: different identities
        negatives_mask = (~identity_mask).float()
        negatives_mask.fill_diagonal_(0)  # Remove self
        
        # InfoNCE loss for emotion embeddings
        # For each anchor, pull positives close, push negatives away
        exp_sim = torch.exp(emotion_sim)
        
        # Sum over positives and negatives
        pos_sum = (exp_sim * positives_mask).sum(dim=1)
        neg_sum = (exp_sim * negatives_mask).sum(dim=1)
        
        # Avoid division by zero
        pos_sum = pos_sum.clamp(min=1e-8)
        
        # Contrastive loss
        emotion_loss = -torch.log(pos_sum / (pos_sum + neg_sum + 1e-8))
        emotion_loss = emotion_loss.mean()
        
        # Identity preservation loss
        # Maximize similarity for same identity
        identity_loss = -(identity_sim * positives_mask).sum() / (positives_mask.sum() + 1e-8)
        
        # Combined loss
        total_loss = emotion_loss + self.identity_weight * identity_loss
        
        # Compute metrics
        with torch.no_grad():
            # Average positive similarity
            avg_pos_sim = (emotion_sim * positives_mask).sum() / (positives_mask.sum() + 1e-8)
            # Average negative similarity
            avg_neg_sim = (emotion_sim * negatives_mask).sum() / (negatives_mask.sum() + 1e-8)
            
            metrics = {
                'emotion_loss': emotion_loss.item(),
                'identity_loss': identity_loss.item(),
                'avg_pos_sim': avg_pos_sim.item(),
                'avg_neg_sim': avg_neg_sim.item(),
                'pos_neg_gap': (avg_pos_sim - avg_neg_sim).item()
            }
        
        return total_loss, metrics


class VideoEmotionDataset(Dataset):
    """
    Dataset for loading video sequences with temporal consistency.
    
    Each sample returns:
    - Multiple frames from the same video (same person)
    - Identity label
    - Emotion labels for each frame
    """
    
    def __init__(self, 
                 data_root: str,
                 split: str = 'train',
                 frames_per_video: int = 8,
                 img_size: int = 224):
        """
        Expected structure:
        data_root/
            videos/
                person_001_emotion_happy/
                    frame_001.jpg
                    frame_002.jpg
                    ...
                person_002_emotion_sad/
                    ...
        """
        self.data_root = Path(data_root)
        self.frames_per_video = frames_per_video
        self.split = split
        
        # Transforms
        if split == 'train':
            self.transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
        
        # Load video paths and create identity mapping
        self.video_paths = self._load_videos()
        self.identity_to_idx = self._create_identity_mapping()
        
        print(f"Loaded {len(self.video_paths)} videos with {len(self.identity_to_idx)} unique identities")
    
    def _load_videos(self) -> List[Dict]:
        """Load all video directories."""
        video_dir = self.data_root / 'videos' / self.split
        videos = []
        
        for video_path in sorted(video_dir.glob('*')):
            if video_path.is_dir():
                frames = sorted(list(video_path.glob('*.jpg')) + list(video_path.glob('*.png')))
                if len(frames) >= self.frames_per_video:
                    # Parse identity and emotion from folder name
                    # Expected format: person_XXX_emotion_YYY
                    parts = video_path.name.split('_')
                    identity = '_'.join(parts[:2])  # person_XXX
                    emotion = parts[-1] if len(parts) > 3 else 'neutral'
                    
                    videos.append({
                        'path': video_path,
                        'frames': frames,
                        'identity': identity,
                        'emotion': emotion
                    })
        
        return videos
    
    def _create_identity_mapping(self) -> Dict[str, int]:
        """Create mapping from identity string to integer index."""
        identities = sorted(set([v['identity'] for v in self.video_paths]))
        return {identity: idx for idx, identity in enumerate(identities)}
    
    def __len__(self) -> int:
        return len(self.video_paths)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Returns:
            frames: [T, C, H, W] - T frames from the video
            identity: scalar - identity index
            emotions: [T] - emotion label for each frame
        """
        video_info = self.video_paths[idx]
        
        # Sample frames uniformly from video
        total_frames = len(video_info['frames'])
        indices = np.linspace(0, total_frames - 1, self.frames_per_video, dtype=int)
        
        frames = []
        for i in indices:
            img_path = video_info['frames'][i]
            img = Image.open(img_path).convert('RGB')
            img = self.transform(img)
            frames.append(img)
        
        frames = torch.stack(frames)  # [T, C, H, W]
        
        # Identity label
        identity = self.identity_to_idx[video_info['identity']]
        
        # Emotion labels (simplified: same emotion for all frames in this video)
        emotion_map = {
            'neutral': 0, 'happy': 1, 'sad': 2, 'angry': 3,
            'fear': 4, 'disgust': 5, 'surprise': 6
        }
        emotion = emotion_map.get(video_info['emotion'], 0)
        emotions = torch.full((self.frames_per_video,), emotion, dtype=torch.long)
        
        return {
            'frames': frames,
            'identity': torch.tensor(identity, dtype=torch.long),
            'emotions': emotions,
            'video_name': video_info['path'].name
        }


class EmotionSSLTrainer:
    """
    Main training pipeline for self-supervised emotion representation learning.
    """
    
    def __init__(self,
                 model: EmotionEncoder,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 device: str = 'cuda',
                 lr: float = 1e-4,
                 temperature: float = 0.07,
                 identity_weight: float = 1.0):
        
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        # Loss and optimizer
        self.criterion = TemporalContrastiveLoss(temperature, identity_weight)
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=100, eta_min=1e-6
        )
        
        # Tracking
        self.history = {'train': [], 'val': []}
        self.best_val_gap = -float('inf')
        
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        
        total_loss = 0
        metrics_sum = {}
        
        pbar = tqdm(self.train_loader, desc='Training')
        for batch_idx, batch in enumerate(pbar):
            frames = batch['frames'].to(self.device)  # [B, T, C, H, W]
            identities = batch['identity'].to(self.device)  # [B]
            
            B, T = frames.shape[:2]
            
            # Reshape: [B*T, C, H, W]
            frames_flat = frames.view(B * T, *frames.shape[2:])
            identities_flat = identities.unsqueeze(1).expand(-1, T).reshape(-1)  # [B*T]
            
            # Forward pass
            emotion_emb, identity_emb, _ = self.model(frames_flat, return_all=True)
            
            # Compute loss
            loss, metrics = self.criterion(emotion_emb, identity_emb, identities_flat)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # Track metrics
            total_loss += loss.item()
            for k, v in metrics.items():
                metrics_sum[k] = metrics_sum.get(k, 0) + v
            
            # Update progress bar
            pbar.set_postfix({
                'loss': loss.item(),
                'gap': metrics['pos_neg_gap']
            })
        
        # Average metrics
        avg_metrics = {k: v / len(self.train_loader) for k, v in metrics_sum.items()}
        avg_metrics['total_loss'] = total_loss / len(self.train_loader)
        
        return avg_metrics
    
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Validate on validation set."""
        self.model.eval()
        
        total_loss = 0
        metrics_sum = {}
        
        for batch in tqdm(self.val_loader, desc='Validation'):
            frames = batch['frames'].to(self.device)
            identities = batch['identity'].to(self.device)
            
            B, T = frames.shape[:2]
            frames_flat = frames.view(B * T, *frames.shape[2:])
            identities_flat = identities.unsqueeze(1).expand(-1, T).reshape(-1)
            
            emotion_emb, identity_emb, _ = self.model(frames_flat, return_all=True)
            loss, metrics = self.criterion(emotion_emb, identity_emb, identities_flat)
            
            total_loss += loss.item()
            for k, v in metrics.items():
                metrics_sum[k] = metrics_sum.get(k, 0) + v
        
        avg_metrics = {k: v / len(self.val_loader) for k, v in metrics_sum.items()}
        avg_metrics['total_loss'] = total_loss / len(self.val_loader)
        
        return avg_metrics
    
    def train(self, num_epochs: int, save_dir: str = './checkpoints'):
        """Full training loop."""
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True, parents=True)
        
        print(f"\n{'='*60}")
        print(f"Starting Self-Supervised Emotion Training")
        print(f"{'='*60}\n")
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            print("-" * 40)
            
            # Train
            train_metrics = self.train_epoch()
            self.history['train'].append(train_metrics)
            
            # Validate
            val_metrics = self.validate()
            self.history['val'].append(val_metrics)
            
            # Update scheduler
            self.scheduler.step()
            
            # Print metrics
            print(f"\nTrain - Loss: {train_metrics['total_loss']:.4f}, "
                  f"Gap: {train_metrics['pos_neg_gap']:.4f}")
            print(f"Val   - Loss: {val_metrics['total_loss']:.4f}, "
                  f"Gap: {val_metrics['pos_neg_gap']:.4f}")
            
            # Save best model
            if val_metrics['pos_neg_gap'] > self.best_val_gap:
                self.best_val_gap = val_metrics['pos_neg_gap']
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_gap': self.best_val_gap,
                    'history': self.history
                }, save_dir / 'best_model.pth')
                print(f"✓ Saved best model (gap: {self.best_val_gap:.4f})")
            
            # Save checkpoint
            if (epoch + 1) % 10 == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'history': self.history
                }, save_dir / f'checkpoint_epoch_{epoch+1}.pth')
        
        print(f"\n{'='*60}")
        print(f"Training Complete! Best validation gap: {self.best_val_gap:.4f}")
        print(f"{'='*60}\n")


def main():
    """Example training script."""
    
    # Hyperparameters
    config = {
        'data_root': './data',  # Update with your data path
        'batch_size': 32,
        'frames_per_video': 8,
        'num_epochs': 100,
        'lr': 1e-4,
        'temperature': 0.07,
        'identity_weight': 1.0,
        'embedding_dim': 128,
        'identity_dim': 256,
        'img_size': 224,
        'num_workers': 4
    }
    
    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Create datasets
    train_dataset = VideoEmotionDataset(
        config['data_root'],
        split='train',
        frames_per_video=config['frames_per_video'],
        img_size=config['img_size']
    )
    
    val_dataset = VideoEmotionDataset(
        config['data_root'],
        split='val',
        frames_per_video=config['frames_per_video'],
        img_size=config['img_size']
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=True
    )
    
    # Create model
    model = EmotionEncoder(
        embedding_dim=config['embedding_dim'],
        identity_dim=config['identity_dim']
    )
    
    # Create trainer
    trainer = EmotionSSLTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        lr=config['lr'],
        temperature=config['temperature'],
        identity_weight=config['identity_weight']
    )
    
    # Train
    trainer.train(num_epochs=config['num_epochs'])


if __name__ == '__main__':
    main()
