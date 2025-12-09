"""
Emotion Generation via Latent Space Manipulation
=================================================

This demonstrates the KEY APPLICATION of our self-supervised representations:
Controllable emotion transformation through learned emotion directions.

Pipeline:
1. Learn emotion direction vectors in latent space
2. Apply them to manipulate facial expressions
3. Decode back to images using a pre-trained decoder (StyleGAN/VAE)

This is what you ORIGINALLY wanted to do, but now with proper foundations!
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms
from tqdm import tqdm


class EmotionDirectionLearner:
    """
    Learn direction vectors for each emotion in the embedding space.
    
    Key insight: If our SSL worked well, then:
    - Same person at different emotions = parallel emotion vectors
    - Direction from neutral to happy should be consistent across people
    """
    
    def __init__(self, encoder: nn.Module, device: str = 'cuda'):
        self.encoder = encoder.to(device)
        self.encoder.eval()
        self.device = device
        
        # Will store emotion directions
        self.emotion_directions = {}
        self.neutral_center = None
    
    @torch.no_grad()
    def compute_emotion_directions(self, 
                                   dataloader: torch.utils.data.DataLoader,
                                   reference_emotion: str = 'neutral') -> Dict[str, torch.Tensor]:
        """
        Compute average direction from reference emotion to each target emotion.
        
        Algorithm:
        1. For each person, compute: direction = emb(emotion) - emb(neutral)
        2. Average across all people to get canonical emotion direction
        3. Normalize directions
        """
        print("\n" + "="*60)
        print("Learning Emotion Direction Vectors")
        print("="*60)
        
        emotion_map = {
            'neutral': 0, 'happy': 1, 'sad': 2, 'angry': 3,
            'fear': 4, 'disgust': 5, 'surprise': 6
        }
        
        # Collect embeddings by emotion
        emotion_embeddings = {emotion: [] for emotion in emotion_map.keys()}
        
        print("\nExtracting embeddings...")
        for images, labels in tqdm(dataloader):
            images = images.to(self.device)
            labels = labels.numpy()
            
            # Get embeddings
            embeddings = self.encoder(images).cpu().numpy()
            
            # Group by emotion
            for emotion_name, emotion_idx in emotion_map.items():
                mask = labels == emotion_idx
                if mask.any():
                    emotion_embeddings[emotion_name].append(embeddings[mask])
        
        # Concatenate
        for emotion in emotion_embeddings:
            if emotion_embeddings[emotion]:
                emotion_embeddings[emotion] = np.concatenate(emotion_embeddings[emotion], axis=0)
            else:
                emotion_embeddings[emotion] = np.array([])
        
        # Compute centers
        emotion_centers = {}
        for emotion, embeddings in emotion_embeddings.items():
            if len(embeddings) > 0:
                emotion_centers[emotion] = torch.from_numpy(embeddings.mean(axis=0)).float()
                print(f"  {emotion}: {len(embeddings)} samples")
        
        # Compute directions from reference
        self.neutral_center = emotion_centers[reference_emotion]
        
        print(f"\nComputing directions from '{reference_emotion}'...")
        for emotion, center in emotion_centers.items():
            if emotion != reference_emotion:
                direction = center - self.neutral_center
                direction = F.normalize(direction.unsqueeze(0), dim=1).squeeze(0)
                self.emotion_directions[emotion] = direction.to(self.device)
                
                # Compute magnitude (how far is this emotion from neutral)
                magnitude = torch.norm(center - self.neutral_center).item()
                print(f"  {emotion}: magnitude = {magnitude:.4f}")
        
        return self.emotion_directions
    
    @torch.no_grad()
    def transform_emotion(self,
                         image: torch.Tensor,
                         target_emotion: str,
                         intensity: float = 1.0) -> torch.Tensor:
        """
        Transform an image to have a target emotion.
        
        Args:
            image: Input image [1, 3, H, W]
            target_emotion: Target emotion name
            intensity: Strength of transformation (0-2, 1=normal)
        
        Returns:
            Transformed embedding [1, D]
        """
        # Get current embedding
        current_emb = self.encoder(image.to(self.device))
        
        # Get emotion direction
        if target_emotion not in self.emotion_directions:
            raise ValueError(f"Emotion '{target_emotion}' not found. "
                           f"Available: {list(self.emotion_directions.keys())}")
        
        direction = self.emotion_directions[target_emotion]
        
        # Apply transformation
        new_emb = current_emb + intensity * direction.unsqueeze(0)
        new_emb = F.normalize(new_emb, dim=1)
        
        return new_emb
    
    def interpolate_emotions(self,
                            image: torch.Tensor,
                            emotion1: str,
                            emotion2: str,
                            num_steps: int = 10) -> List[torch.Tensor]:
        """
        Create smooth interpolation between two emotions.
        """
        embeddings = []
        
        for alpha in np.linspace(0, 1, num_steps):
            # Blend directions
            dir1 = self.emotion_directions[emotion1]
            dir2 = self.emotion_directions[emotion2]
            
            blended_direction = (1 - alpha) * dir1 + alpha * dir2
            blended_direction = F.normalize(blended_direction.unsqueeze(0), dim=1).squeeze(0)
            
            # Apply
            current_emb = self.encoder(image.to(self.device))
            new_emb = current_emb + blended_direction.unsqueeze(0)
            new_emb = F.normalize(new_emb, dim=1)
            
            embeddings.append(new_emb)
        
        return embeddings


class LatentSpaceDecoder(nn.Module):
    """
    Decoder to map embeddings back to images.
    
    Options:
    1. Train a simple decoder (CNN)
    2. Use a pre-trained GAN (StyleGAN) - inject into intermediate layers
    3. Use diffusion model guidance (like you originally wanted!)
    
    Here we implement option 1 (simplest, trainable in 4 days).
    """
    
    def __init__(self, 
                 embedding_dim: int = 128,
                 img_size: int = 224,
                 base_channels: int = 64):
        super().__init__()
        
        self.img_size = img_size
        
        # Initial projection
        self.fc = nn.Sequential(
            nn.Linear(embedding_dim, 512 * 4 * 4),
            nn.ReLU(inplace=True)
        )
        
        # Upsampling blocks
        self.decoder = nn.Sequential(
            # 4x4 -> 8x8
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            # 8x8 -> 16x16
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            # 16x16 -> 32x32
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            # 32x32 -> 64x64
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            # 64x64 -> 128x128
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            
            # 128x128 -> 224x224 (if needed)
            nn.Upsample(size=(img_size, img_size), mode='bilinear', align_corners=False),
            
            # Final layer
            nn.Conv2d(16, 3, kernel_size=3, padding=1),
            nn.Tanh()
        )
    
    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        """
        Args:
            embedding: [B, embedding_dim]
        
        Returns:
            image: [B, 3, H, W]
        """
        x = self.fc(embedding)
        x = x.view(x.size(0), 512, 4, 4)
        x = self.decoder(x)
        return x


class EmotionTransformationSystem:
    """
    Complete system: Encoder -> Direction Learning -> Decoder
    """
    
    def __init__(self,
                 encoder: nn.Module,
                 decoder: nn.Module,
                 direction_learner: EmotionDirectionLearner,
                 device: str = 'cuda'):
        
        self.encoder = encoder.to(device)
        self.decoder = decoder.to(device)
        self.direction_learner = direction_learner
        self.device = device
    
    def train_decoder(self,
                     dataloader: torch.utils.data.DataLoader,
                     num_epochs: int = 50,
                     lr: float = 1e-4):
        """
        Train decoder to reconstruct images from embeddings.
        """
        print("\n" + "="*60)
        print("Training Decoder")
        print("="*60)
        
        self.encoder.eval()  # Freeze encoder
        self.decoder.train()
        
        optimizer = torch.optim.Adam(self.decoder.parameters(), lr=lr)
        criterion_recon = nn.L1Loss()
        criterion_perceptual = self._get_perceptual_loss()
        
        for epoch in range(num_epochs):
            total_loss = 0
            
            pbar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{num_epochs}')
            for images, _ in pbar:
                images = images.to(self.device)
                
                # Encode
                with torch.no_grad():
                    embeddings = self.encoder(images)
                
                # Decode
                reconstructed = self.decoder(embeddings)
                
                # Losses
                loss_recon = criterion_recon(reconstructed, images)
                loss_perceptual = criterion_perceptual(reconstructed, images)
                
                loss = loss_recon + 0.1 * loss_perceptual
                
                # Backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                pbar.set_postfix({'loss': loss.item()})
            
            avg_loss = total_loss / len(dataloader)
            print(f"Epoch {epoch+1} - Avg Loss: {avg_loss:.4f}")
            
            # Save checkpoint
            if (epoch + 1) % 10 == 0:
                torch.save(self.decoder.state_dict(), 
                          f'./checkpoints/decoder_epoch_{epoch+1}.pth')
    
    def _get_perceptual_loss(self):
        """VGG-based perceptual loss for better quality."""
        import torchvision.models as models
        
        vgg = models.vgg16(pretrained=True).features[:16].to(self.device)
        vgg.eval()
        for p in vgg.parameters():
            p.requires_grad = False
        
        def perceptual_loss(pred, target):
            pred_features = vgg(pred)
            target_features = vgg(target)
            return F.mse_loss(pred_features, target_features)
        
        return perceptual_loss
    
    @torch.no_grad()
    def generate_emotion_transformation(self,
                                       image: torch.Tensor,
                                       target_emotion: str,
                                       intensity: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Full pipeline: image -> embedding -> transform -> decode
        
        Returns:
            original_image, transformed_image
        """
        self.encoder.eval()
        self.decoder.eval()
        
        # Transform emotion
        transformed_emb = self.direction_learner.transform_emotion(
            image, target_emotion, intensity
        )
        
        # Decode
        transformed_img = self.decoder(transformed_emb)
        
        return image, transformed_img
    
    @torch.no_grad()
    def visualize_emotion_grid(self,
                              image: torch.Tensor,
                              save_path: str = './emotion_grid.png'):
        """
        Create a grid showing the same face with all emotions.
        """
        emotions = ['happy', 'sad', 'angry', 'fear', 'disgust', 'surprise']
        intensities = [0.5, 1.0, 1.5]
        
        fig, axes = plt.subplots(len(emotions), len(intensities) + 1, 
                                figsize=(15, 3 * len(emotions)))
        
        # Denormalize function
        def denorm(img):
            img = img.cpu().squeeze(0).permute(1, 2, 0).numpy()
            img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
            img = np.clip(img, 0, 1)
            return img
        
        for i, emotion in enumerate(emotions):
            # Original image
            axes[i, 0].imshow(denorm(image))
            axes[i, 0].set_title('Original', fontsize=12, fontweight='bold')
            axes[i, 0].axis('off')
            
            # Transformed with different intensities
            for j, intensity in enumerate(intensities):
                _, transformed = self.generate_emotion_transformation(
                    image, emotion, intensity
                )
                
                axes[i, j+1].imshow(denorm(transformed))
                axes[i, j+1].set_title(f'{emotion.capitalize()}\n(α={intensity})', 
                                      fontsize=10)
                axes[i, j+1].axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nEmotion grid saved to {save_path}")
        
        return fig
    
    @torch.no_grad()
    def create_emotion_morphing_video(self,
                                     image: torch.Tensor,
                                     emotion_sequence: List[str],
                                     fps: int = 10,
                                     save_path: str = './emotion_morph.mp4'):
        """
        Create a video morphing through different emotions.
        """
        import cv2
        
        frames = []
        
        for i in range(len(emotion_sequence) - 1):
            emotion1 = emotion_sequence[i]
            emotion2 = emotion_sequence[i + 1]
            
            # Interpolate
            embeddings = self.direction_learner.interpolate_emotions(
                image, emotion1, emotion2, num_steps=20
            )
            
            # Decode all
            for emb in embeddings:
                decoded = self.decoder(emb)
                
                # Convert to numpy
                img_np = decoded.cpu().squeeze(0).permute(1, 2, 0).numpy()
                img_np = img_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
                img_np = np.clip(img_np, 0, 1)
                img_np = (img_np * 255).astype(np.uint8)
                img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
                
                frames.append(img_np)
        
        # Write video
        height, width = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(save_path, fourcc, fps, (width, height))
        
        for frame in frames:
            out.write(frame)
        
        out.release()
        print(f"\nMorphing video saved to {save_path}")


def demo_pipeline():
    """
    Complete demonstration of the emotion transformation system.
    """
    from main_pipeline import EmotionEncoder  # Import your encoder
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load trained encoder
    print("Loading trained encoder...")
    encoder = EmotionEncoder(embedding_dim=128, identity_dim=256)
    checkpoint = torch.load('./checkpoints/best_model.pth', map_location=device)
    encoder.load_state_dict(checkpoint['model_state_dict'])
    encoder.eval()
    
    # Create decoder
    print("Creating decoder...")
    decoder = LatentSpaceDecoder(embedding_dim=128, img_size=224)
    
    # Load test data to learn directions
    print("Loading dataset...")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # You need a dataset here
    from evaluation import EmotionClassificationDataset
    dataset = EmotionClassificationDataset('./data/fer2013', 'test', transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False)
    
    # Learn emotion directions
    print("\n" + "="*60)
    print("Step 1: Learning Emotion Directions")
    print("="*60)
    direction_learner = EmotionDirectionLearner(encoder, device)
    direction_learner.compute_emotion_directions(dataloader)
    
    # Train decoder
    print("\n" + "="*60)
    print("Step 2: Training Decoder")
    print("="*60)
    system = EmotionTransformationSystem(encoder, decoder, direction_learner, device)
    system.train_decoder(dataloader, num_epochs=50)
    
    # Demo transformations
    print("\n" + "="*60)
    print("Step 3: Generating Transformations")
    print("="*60)
    
    # Load a test image
    test_img, _ = dataset[0]
    test_img = test_img.unsqueeze(0).to(device)
    
    # Create emotion grid
    system.visualize_emotion_grid(test_img, save_path='./results/emotion_grid.png')
    
    # Create morphing video
    system.create_emotion_morphing_video(
        test_img,
        emotion_sequence=['neutral', 'happy', 'sad', 'angry', 'surprise'],
        save_path='./results/emotion_morph.mp4'
    )
    
    print("\n" + "="*60)
    print("Demo Complete!")
    print("="*60)
    print("\nGenerated files:")
    print("  - ./results/emotion_grid.png")
    print("  - ./results/emotion_morph.mp4")


if __name__ == '__main__':
    demo_pipeline()
