"""
Downstream Tasks & Evaluation for Self-Supervised Emotion Learning
===================================================================

This module provides:
1. Linear probing for emotion classification
2. k-NN evaluation
3. Emotion retrieval
4. Latent space visualization (t-SNE)
5. Emotion vector arithmetic
6. Cross-dataset generalization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import seaborn as sns
from tqdm import tqdm
import json


class EmotionClassificationDataset(Dataset):
    """
    Standard emotion classification dataset (e.g., FER2013, AffectNet).
    Single images with emotion labels.
    """
    
    def __init__(self, data_root: str, split: str = 'train', transform=None):
        self.data_root = Path(data_root)
        self.split = split
        self.transform = transform
        
        # Load image paths and labels
        self.samples = self._load_samples()
        
        self.emotion_names = ['neutral', 'happy', 'sad', 'angry', 
                             'fear', 'disgust', 'surprise']
    
    def _load_samples(self) -> List[Dict]:
        """Load all samples from split."""
        samples = []
        split_dir = self.data_root / self.split
        
        # Expected structure: split/emotion_name/*.jpg
        for emotion_dir in sorted(split_dir.glob('*')):
            if emotion_dir.is_dir():
                emotion_name = emotion_dir.name
                for img_path in emotion_dir.glob('*.jpg'):
                    samples.append({
                        'path': img_path,
                        'emotion': emotion_name
                    })
        
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        sample = self.samples[idx]
        
        from PIL import Image
        img = Image.open(sample['path']).convert('RGB')
        
        if self.transform:
            img = self.transform(img)
        
        # Emotion label
        emotion_map = {
            'neutral': 0, 'happy': 1, 'sad': 2, 'angry': 3,
            'fear': 4, 'disgust': 5, 'surprise': 6
        }
        label = emotion_map[sample['emotion']]
        
        return img, label


class LinearProbe(nn.Module):
    """
    Linear classifier on top of frozen features for evaluation.
    """
    
    def __init__(self, input_dim: int, num_classes: int = 7):
        super().__init__()
        self.classifier = nn.Linear(input_dim, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)


class EmotionEvaluator:
    """
    Comprehensive evaluation suite for learned representations.
    """
    
    def __init__(self, 
                 encoder: nn.Module, 
                 device: str = 'cuda'):
        self.encoder = encoder.to(device)
        self.encoder.eval()
        self.device = device
    
    @torch.no_grad()
    def extract_features(self, dataloader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract features and labels from a dataset.
        
        Returns:
            features: [N, D] numpy array
            labels: [N] numpy array
        """
        all_features = []
        all_labels = []
        
        for images, labels in tqdm(dataloader, desc='Extracting features'):
            images = images.to(self.device)
            
            # Get emotion embeddings
            features = self.encoder(images)  # [B, D]
            
            all_features.append(features.cpu().numpy())
            all_labels.append(labels.numpy())
        
        features = np.concatenate(all_features, axis=0)
        labels = np.concatenate(all_labels, axis=0)
        
        return features, labels
    
    def linear_probe_evaluation(self, 
                                train_loader: DataLoader,
                                test_loader: DataLoader,
                                num_epochs: int = 50) -> Dict:
        """
        Train a linear classifier on frozen features.
        This is the standard evaluation protocol for self-supervised learning.
        """
        print("\n" + "="*60)
        print("Linear Probe Evaluation")
        print("="*60)
        
        # Extract features
        print("\nExtracting training features...")
        train_features, train_labels = self.extract_features(train_loader)
        
        print("Extracting test features...")
        test_features, test_labels = self.extract_features(test_loader)
        
        # Get feature dimension
        feature_dim = train_features.shape[1]
        num_classes = len(np.unique(train_labels))
        
        # Create linear classifier
        linear_probe = LinearProbe(feature_dim, num_classes).to(self.device)
        optimizer = torch.optim.Adam(linear_probe.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()
        
        # Convert to tensors
        train_features_t = torch.from_numpy(train_features).float()
        train_labels_t = torch.from_numpy(train_labels).long()
        test_features_t = torch.from_numpy(test_features).float()
        test_labels_t = torch.from_numpy(test_labels).long()
        
        # Create dataloaders
        train_dataset = torch.utils.data.TensorDataset(train_features_t, train_labels_t)
        train_loader_probe = DataLoader(train_dataset, batch_size=256, shuffle=True)
        
        # Training loop
        best_acc = 0
        print("\nTraining linear probe...")
        
        for epoch in range(num_epochs):
            linear_probe.train()
            total_loss = 0
            
            for features, labels in train_loader_probe:
                features, labels = features.to(self.device), labels.to(self.device)
                
                logits = linear_probe(features)
                loss = criterion(logits, labels)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            # Evaluate
            if (epoch + 1) % 10 == 0:
                linear_probe.eval()
                with torch.no_grad():
                    test_features_d = test_features_t.to(self.device)
                    test_logits = linear_probe(test_features_d)
                    test_preds = test_logits.argmax(dim=1).cpu().numpy()
                
                acc = accuracy_score(test_labels, test_preds)
                f1 = f1_score(test_labels, test_preds, average='macro')
                
                print(f"Epoch {epoch+1}/{num_epochs} - "
                      f"Loss: {total_loss/len(train_loader_probe):.4f}, "
                      f"Test Acc: {acc:.4f}, F1: {f1:.4f}")
                
                best_acc = max(best_acc, acc)
        
        # Final evaluation
        linear_probe.eval()
        with torch.no_grad():
            test_features_d = test_features_t.to(self.device)
            test_logits = linear_probe(test_features_d)
            test_preds = test_logits.argmax(dim=1).cpu().numpy()
        
        # Compute metrics
        final_acc = accuracy_score(test_labels, test_preds)
        final_f1 = f1_score(test_labels, test_preds, average='macro')
        conf_matrix = confusion_matrix(test_labels, test_preds)
        
        results = {
            'accuracy': final_acc,
            'f1_macro': final_f1,
            'best_accuracy': best_acc,
            'confusion_matrix': conf_matrix
        }
        
        print(f"\nFinal Results:")
        print(f"  Accuracy: {final_acc:.4f}")
        print(f"  F1 (macro): {final_f1:.4f}")
        print(f"  Best Accuracy: {best_acc:.4f}")
        
        return results
    
    def knn_evaluation(self,
                      train_loader: DataLoader,
                      test_loader: DataLoader,
                      k: int = 5) -> Dict:
        """
        k-NN evaluation on learned features.
        Good indicator of feature quality without any training.
        """
        print("\n" + "="*60)
        print(f"k-NN Evaluation (k={k})")
        print("="*60)
        
        # Extract features
        train_features, train_labels = self.extract_features(train_loader)
        test_features, test_labels = self.extract_features(test_loader)
        
        # Train k-NN
        print(f"\nTraining {k}-NN classifier...")
        knn = KNeighborsClassifier(n_neighbors=k, metric='cosine')
        knn.fit(train_features, train_labels)
        
        # Predict
        print("Predicting...")
        test_preds = knn.predict(test_features)
        
        # Metrics
        acc = accuracy_score(test_labels, test_preds)
        f1 = f1_score(test_labels, test_preds, average='macro')
        
        print(f"\nResults:")
        print(f"  Accuracy: {acc:.4f}")
        print(f"  F1 (macro): {f1:.4f}")
        
        return {
            'accuracy': acc,
            'f1_macro': f1,
            'predictions': test_preds
        }
    
    def visualize_latent_space(self,
                               dataloader: DataLoader,
                               save_path: str = './latent_space.png',
                               max_samples: int = 2000):
        """
        Visualize the learned embedding space using t-SNE.
        """
        print("\n" + "="*60)
        print("Latent Space Visualization")
        print("="*60)
        
        # Extract features
        features, labels = self.extract_features(dataloader)
        
        # Subsample if too many
        if len(features) > max_samples:
            indices = np.random.choice(len(features), max_samples, replace=False)
            features = features[indices]
            labels = labels[indices]
        
        # Apply t-SNE
        print("\nApplying t-SNE...")
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        features_2d = tsne.fit_transform(features)
        
        # Plot
        emotion_names = ['Neutral', 'Happy', 'Sad', 'Angry', 
                        'Fear', 'Disgust', 'Surprise']
        colors = plt.cm.tab10(np.linspace(0, 1, 7))
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        for emotion_idx in range(7):
            mask = labels == emotion_idx
            ax.scatter(features_2d[mask, 0], 
                      features_2d[mask, 1],
                      c=[colors[emotion_idx]],
                      label=emotion_names[emotion_idx],
                      alpha=0.6,
                      s=50)
        
        ax.legend(fontsize=12, loc='best')
        ax.set_xlabel('t-SNE Dimension 1', fontsize=14)
        ax.set_ylabel('t-SNE Dimension 2', fontsize=14)
        ax.set_title('Emotion Embedding Space (t-SNE)', fontsize=16, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nVisualization saved to {save_path}")
        
        return fig
    
    def emotion_arithmetic(self,
                          img1: torch.Tensor,
                          img2: torch.Tensor,
                          alpha: float = 1.0) -> torch.Tensor:
        """
        Perform emotion vector arithmetic.
        
        Example: neutral_face + (happy_ref - neutral_ref) = happy_face
        
        Args:
            img1: Source image [1, 3, H, W]
            img2: Reference image with target emotion [1, 3, H, W]
            alpha: Interpolation factor
        
        Returns:
            Modified embedding
        """
        with torch.no_grad():
            emb1 = self.encoder(img1)
            emb2 = self.encoder(img2)
            
            # Compute emotion direction
            emotion_direction = emb2 - emb1
            
            # Apply with interpolation
            new_emb = emb1 + alpha * emotion_direction
            new_emb = F.normalize(new_emb, dim=1)
        
        return new_emb
    
    def emotion_retrieval(self,
                         query_img: torch.Tensor,
                         gallery_loader: DataLoader,
                         top_k: int = 10) -> List[Tuple[int, float]]:
        """
        Retrieve most similar images based on emotion similarity.
        
        Args:
            query_img: Query image [1, 3, H, W]
            gallery_loader: DataLoader for gallery images
            top_k: Number of results to return
        
        Returns:
            List of (index, similarity) tuples
        """
        with torch.no_grad():
            # Get query embedding
            query_emb = self.encoder(query_img)  # [1, D]
            
            # Get all gallery embeddings
            gallery_embs = []
            for images, _ in tqdm(gallery_loader, desc='Building gallery'):
                images = images.to(self.device)
                embs = self.encoder(images)
                gallery_embs.append(embs.cpu())
            
            gallery_embs = torch.cat(gallery_embs, dim=0)  # [N, D]
            
            # Compute similarities (cosine)
            similarities = F.cosine_similarity(
                query_emb.cpu(), 
                gallery_embs, 
                dim=1
            )
            
            # Get top-k
            top_k_vals, top_k_indices = torch.topk(similarities, k=top_k)
            
            results = [(idx.item(), sim.item()) 
                      for idx, sim in zip(top_k_indices, top_k_vals)]
        
        return results
    
    def cross_dataset_evaluation(self,
                                 source_train_loader: DataLoader,
                                 target_test_loader: DataLoader) -> Dict:
        """
        Evaluate cross-dataset generalization.
        Train on one dataset, test on another (e.g., train on FER, test on AffectNet).
        """
        print("\n" + "="*60)
        print("Cross-Dataset Generalization")
        print("="*60)
        
        return self.linear_probe_evaluation(source_train_loader, target_test_loader)


def plot_confusion_matrix(cm: np.ndarray, 
                         save_path: str = './confusion_matrix.png'):
    """
    Plot confusion matrix.
    """
    emotion_names = ['Neutral', 'Happy', 'Sad', 'Angry', 
                    'Fear', 'Disgust', 'Surprise']
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Normalize
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=emotion_names,
                yticklabels=emotion_names,
                ax=ax,
                cbar_kws={'label': 'Normalized Count'})
    
    ax.set_xlabel('Predicted', fontsize=14)
    ax.set_ylabel('True', fontsize=14)
    ax.set_title('Emotion Classification Confusion Matrix', fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Confusion matrix saved to {save_path}")


def plot_training_curves(history: Dict, save_path: str = './training_curves.png'):
    """
    Plot training and validation curves.
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    epochs = range(1, len(history['train']) + 1)
    
    # Total loss
    axes[0, 0].plot(epochs, [h['total_loss'] for h in history['train']], 
                   label='Train', linewidth=2)
    axes[0, 0].plot(epochs, [h['total_loss'] for h in history['val']], 
                   label='Val', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Total Loss')
    axes[0, 0].set_title('Total Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Emotion loss
    axes[0, 1].plot(epochs, [h['emotion_loss'] for h in history['train']], 
                   label='Train', linewidth=2)
    axes[0, 1].plot(epochs, [h['emotion_loss'] for h in history['val']], 
                   label='Val', linewidth=2)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Emotion Loss')
    axes[0, 1].set_title('Contrastive Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Identity loss
    axes[1, 0].plot(epochs, [h['identity_loss'] for h in history['train']], 
                   label='Train', linewidth=2)
    axes[1, 0].plot(epochs, [h['identity_loss'] for h in history['val']], 
                   label='Val', linewidth=2)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Identity Loss')
    axes[1, 0].set_title('Identity Preservation Loss')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Positive-Negative gap
    axes[1, 1].plot(epochs, [h['pos_neg_gap'] for h in history['train']], 
                   label='Train', linewidth=2)
    axes[1, 1].plot(epochs, [h['pos_neg_gap'] for h in history['val']], 
                   label='Val', linewidth=2)
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Similarity Gap')
    axes[1, 1].set_title('Positive-Negative Similarity Gap')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Training curves saved to {save_path}")


def main_evaluation():
    """
    Complete evaluation pipeline.
    """
    import torchvision.transforms as transforms
    from main_pipeline import EmotionEncoder  # Import from main script
    
    # Load trained model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = EmotionEncoder(embedding_dim=128, identity_dim=256)
    
    checkpoint = torch.load('./checkpoints/best_model.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print("Loaded trained model")
    
    # Create transform
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Load test dataset
    test_dataset = EmotionClassificationDataset(
        data_root='./data/fer2013',
        split='test',
        transform=transform
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=4
    )
    
    # Create evaluator
    evaluator = EmotionEvaluator(model, device=device)
    
    # Run evaluations
    results = {}
    
    # 1. k-NN evaluation
    results['knn'] = evaluator.knn_evaluation(test_loader, test_loader, k=5)
    
    # 2. Linear probe
    results['linear'] = evaluator.linear_probe_evaluation(test_loader, test_loader)
    
    # 3. Visualize latent space
    evaluator.visualize_latent_space(test_loader, save_path='./results/latent_space.png')
    
    # 4. Plot confusion matrix
    plot_confusion_matrix(
        results['linear']['confusion_matrix'],
        save_path='./results/confusion_matrix.png'
    )
    
    # 5. Plot training curves
    if 'history' in checkpoint:
        plot_training_curves(
            checkpoint['history'],
            save_path='./results/training_curves.png'
        )
    
    # Save results
    with open('./results/evaluation_results.json', 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        results_serializable = {
            'knn_accuracy': results['knn']['accuracy'],
            'knn_f1': results['knn']['f1_macro'],
            'linear_accuracy': results['linear']['accuracy'],
            'linear_f1': results['linear']['f1_macro']
        }
        json.dump(results_serializable, f, indent=2)
    
    print("\n" + "="*60)
    print("Evaluation Complete!")
    print("="*60)
    print(f"\nSummary:")
    print(f"  k-NN Accuracy: {results['knn']['accuracy']:.4f}")
    print(f"  Linear Probe Accuracy: {results['linear']['accuracy']:.4f}")
    print(f"\nAll results saved to ./results/")


if __name__ == '__main__':
    main_evaluation()
