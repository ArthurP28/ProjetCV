"""
Synthetic Data Generation for Emotion SSL
==========================================

CRITICAL for quick testing without large datasets!

This script generates synthetic "video" sequences by:
1. Starting with static emotion datasets (FER2013, etc.)
2. Creating pseudo-temporal sequences with augmentations
3. Organizing in the format expected by VideoEmotionDataset

This lets you test the entire pipeline in hours, not days!
"""

import torch
import torchvision.transforms as transforms
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np
from pathlib import Path
import shutil
from tqdm import tqdm
import random
import json


class SyntheticVideoGenerator:
    """
    Generate synthetic video sequences from static images.
    
    Strategy:
    - Take one image
    - Apply temporal augmentations (brightness, blur, rotation, etc.)
    - Save as "frames" of a pseudo-video
    - Preserve identity within sequence
    """
    
    def __init__(self, 
                 source_dataset_root: str,
                 output_root: str,
                 frames_per_video: int = 8,
                 videos_per_identity: int = 3):
        
        self.source_root = Path(source_dataset_root)
        self.output_root = Path(output_root)
        self.frames_per_video = frames_per_video
        self.videos_per_identity = videos_per_identity
        
        # Create output structure
        (self.output_root / 'videos' / 'train').mkdir(parents=True, exist_ok=True)
        (self.output_root / 'videos' / 'val').mkdir(parents=True, exist_ok=True)
        (self.output_root / 'videos' / 'test').mkdir(parents=True, exist_ok=True)
        
        print(f"Output directory: {self.output_root}")
    
    def temporal_augmentations(self, img: Image.Image, frame_idx: int) -> Image.Image:
        """
        Apply temporal augmentations to create variation within a video.
        
        These simulate natural variations in lighting, camera angle, etc.
        """
        # Smooth progression of parameters
        t = frame_idx / self.frames_per_video  # 0 to 1
        
        # Brightness oscillation
        brightness_factor = 1.0 + 0.2 * np.sin(2 * np.pi * t)
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(brightness_factor)
        
        # Contrast variation
        contrast_factor = 1.0 + 0.15 * np.cos(2 * np.pi * t)
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(contrast_factor)
        
        # Slight rotation (simulates head movement)
        angle = 5 * np.sin(2 * np.pi * t)
        img = img.rotate(angle, fillcolor=(128, 128, 128))
        
        # Slight blur (simulates motion)
        if random.random() > 0.5:
            img = img.filter(ImageFilter.GaussianBlur(radius=0.5))
        
        # Random crop and resize (simulates slight camera movement)
        w, h = img.size
        crop_size = int(min(w, h) * random.uniform(0.9, 0.95))
        left = random.randint(0, w - crop_size)
        top = random.randint(0, h - crop_size)
        img = img.crop((left, top, left + crop_size, top + crop_size))
        img = img.resize((w, h), Image.LANCZOS)
        
        # Color temperature shift
        if random.random() > 0.7:
            enhancer = ImageEnhance.Color(img)
            img = enhancer.enhance(random.uniform(0.9, 1.1))
        
        return img
    
    def create_video_sequence(self, 
                            source_img_path: Path, 
                            identity_id: int,
                            emotion: str,
                            video_idx: int,
                            split: str):
        """
        Create one synthetic video from a single image.
        """
        # Create video directory
        video_name = f"person_{identity_id:04d}_emotion_{emotion}_v{video_idx}"
        video_dir = self.output_root / 'videos' / split / video_name
        video_dir.mkdir(parents=True, exist_ok=True)
        
        # Load source image
        img = Image.open(source_img_path).convert('RGB')
        
        # Generate frames
        for frame_idx in range(self.frames_per_video):
            # Apply temporal augmentations
            augmented = self.temporal_augmentations(img, frame_idx)
            
            # Add random noise
            if random.random() > 0.7:
                img_array = np.array(augmented)
                noise = np.random.normal(0, 5, img_array.shape)
                img_array = np.clip(img_array + noise, 0, 255).astype(np.uint8)
                augmented = Image.fromarray(img_array)
            
            # Save frame
            frame_path = video_dir / f"frame_{frame_idx:03d}.jpg"
            augmented.save(frame_path, quality=95)
        
        return video_dir
    
    def generate_from_static_dataset(self, 
                                    train_ratio: float = 0.7,
                                    val_ratio: float = 0.15):
        """
        Convert a static emotion dataset into synthetic videos.
        
        Expected source structure:
        source_dataset_root/
            neutral/
                img1.jpg
                img2.jpg
                ...
            happy/
                ...
            sad/
                ...
        """
        print("\n" + "="*60)
        print("Generating Synthetic Video Dataset")
        print("="*60)
        
        emotions = ['neutral', 'happy', 'sad', 'angry', 'fear', 'disgust', 'surprise']
        
        # Statistics
        stats = {
            'train': {'videos': 0, 'frames': 0},
            'val': {'videos': 0, 'frames': 0},
            'test': {'videos': 0, 'frames': 0}
        }
        
        identity_counter = 0
        
        # Process each emotion
        for emotion in emotions:
            emotion_dir = self.source_root / emotion
            
            if not emotion_dir.exists():
                print(f"⚠️  Emotion directory not found: {emotion}")
                continue
            
            # Get all images for this emotion
            image_paths = list(emotion_dir.glob('*.jpg')) + \
                         list(emotion_dir.glob('*.png'))
            
            if len(image_paths) == 0:
                print(f"⚠️  No images found for {emotion}")
                continue
            
            print(f"\n📁 Processing {emotion}: {len(image_paths)} images")
            
            # Each image becomes one "identity"
            for img_idx, img_path in enumerate(tqdm(image_paths, desc=f'  {emotion}')):
                
                # Determine split
                rand = random.random()
                if rand < train_ratio:
                    split = 'train'
                elif rand < train_ratio + val_ratio:
                    split = 'val'
                else:
                    split = 'test'
                
                # Create multiple videos for this identity
                for video_idx in range(self.videos_per_identity):
                    video_dir = self.create_video_sequence(
                        img_path,
                        identity_id=identity_counter,
                        emotion=emotion,
                        video_idx=video_idx,
                        split=split
                    )
                    
                    stats[split]['videos'] += 1
                    stats[split]['frames'] += self.frames_per_video
                
                identity_counter += 1
        
        # Print statistics
        print("\n" + "="*60)
        print("Generation Complete!")
        print("="*60)
        
        for split in ['train', 'val', 'test']:
            print(f"\n{split.upper()}:")
            print(f"  Videos: {stats[split]['videos']}")
            print(f"  Frames: {stats[split]['frames']}")
            print(f"  Identities: {stats[split]['videos'] // self.videos_per_identity}")
        
        # Save metadata
        metadata = {
            'frames_per_video': self.frames_per_video,
            'videos_per_identity': self.videos_per_identity,
            'emotions': emotions,
            'total_identities': identity_counter,
            'stats': stats
        }
        
        with open(self.output_root / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\n✓ Metadata saved to {self.output_root / 'metadata.json'}")
        print(f"✓ Ready to train with VideoEmotionDataset!")


def download_and_prepare_fer2013():
    """
    Helper to download FER2013 if you don't have data.
    
    Note: You'll need to manually download from Kaggle:
    https://www.kaggle.com/datasets/msambare/fer2013
    """
    print("\n" + "="*60)
    print("FER2013 Dataset Preparation")
    print("="*60)
    
    print("""
To use FER2013:

1. Download from Kaggle:
   https://www.kaggle.com/datasets/msambare/fer2013

2. Extract to: ./data/fer2013_raw/

3. Expected structure:
   ./data/fer2013_raw/
       train/
           angry/
               im0.png
               im1.png
               ...
           happy/
               ...
       test/
           ...

4. Run this script to generate synthetic videos
    """)


def generate_minimal_test_dataset():
    """
    Create a MINIMAL dataset for quick testing (< 1 minute).
    
    Uses random images - only for pipeline testing!
    """
    print("\n" + "="*60)
    print("Generating Minimal Test Dataset")
    print("="*60)
    
    output_root = Path('./data/test_synthetic')
    (output_root / 'videos' / 'train').mkdir(parents=True, exist_ok=True)
    (output_root / 'videos' / 'val').mkdir(parents=True, exist_ok=True)
    
    emotions = ['neutral', 'happy', 'sad', 'angry']
    n_videos_per_emotion = 10
    frames_per_video = 8
    
    print(f"\nCreating {n_videos_per_emotion * len(emotions)} videos...")
    
    video_idx = 0
    for emotion in emotions:
        for i in range(n_videos_per_emotion):
            # Determine split
            split = 'train' if i < 8 else 'val'
            
            # Create video directory
            video_name = f"person_{video_idx:04d}_emotion_{emotion}"
            video_dir = output_root / 'videos' / split / video_name
            video_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate random "face" image
            # In reality, you'd use real faces - this is just for testing
            for frame_idx in range(frames_per_video):
                # Create random colored image (placeholder for face)
                img_array = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
                
                # Add some structure (simulate a face)
                # Eyes
                img_array[20:25, 15:20] = [255, 255, 255]
                img_array[20:25, 45:50] = [255, 255, 255]
                # Mouth (changes with emotion)
                if emotion == 'happy':
                    img_array[45:48, 20:45] = [255, 0, 0]
                elif emotion == 'sad':
                    img_array[48:51, 20:45] = [0, 0, 255]
                
                img = Image.fromarray(img_array)
                img = img.resize((224, 224), Image.NEAREST)
                
                # Add temporal variation
                enhancer = ImageEnhance.Brightness(img)
                img = enhancer.enhance(0.8 + 0.4 * frame_idx / frames_per_video)
                
                img.save(video_dir / f"frame_{frame_idx:03d}.jpg")
            
            video_idx += 1
    
    print(f"\n✓ Created {video_idx} videos")
    print(f"✓ Location: {output_root}")
    print(f"\n⚠️  This is PLACEHOLDER data - replace with real faces!")
    
    return output_root


def main():
    """
    Main entry point for data generation.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate synthetic video dataset')
    parser.add_argument('--mode', type=str, 
                       choices=['fer2013', 'custom', 'test'],
                       default='test',
                       help='Generation mode')
    parser.add_argument('--source', type=str, 
                       default='./data/fer2013_raw',
                       help='Source dataset path')
    parser.add_argument('--output', type=str,
                       default='./data/synthetic_videos',
                       help='Output path')
    parser.add_argument('--frames', type=int, default=8,
                       help='Frames per video')
    parser.add_argument('--videos', type=int, default=3,
                       help='Videos per identity')
    
    args = parser.parse_args()
    
    if args.mode == 'test':
        # Quick test dataset
        output_root = generate_minimal_test_dataset()
        print(f"\n🎯 Next step: Test training with:")
        print(f"   python main_pipeline.py --data_root {output_root}")
        
    elif args.mode == 'fer2013':
        # Check if FER2013 exists
        source_path = Path(args.source)
        if not source_path.exists():
            download_and_prepare_fer2013()
            return
        
        # Generate from FER2013
        generator = SyntheticVideoGenerator(
            source_dataset_root=args.source,
            output_root=args.output,
            frames_per_video=args.frames,
            videos_per_identity=args.videos
        )
        
        generator.generate_from_static_dataset()
        
        print(f"\n🎯 Next step: Train the model with:")
        print(f"   python main_pipeline.py --data_root {args.output}")
        
    elif args.mode == 'custom':
        # Custom dataset
        generator = SyntheticVideoGenerator(
            source_dataset_root=args.source,
            output_root=args.output,
            frames_per_video=args.frames,
            videos_per_identity=args.videos
        )
        
        generator.generate_from_static_dataset()


if __name__ == '__main__':
    # If run without args, show help
    import sys
    if len(sys.argv) == 1:
        print("""
╔════════════════════════════════════════════════════════════╗
║     Synthetic Video Data Generation for Emotion SSL       ║
╚════════════════════════════════════════════════════════════╝

Three modes:

1. QUICK TEST (recommended for first run):
   python synthetic_data_gen.py --mode test
   
   → Generates minimal dataset in < 1 minute
   → Tests full pipeline
   → Uses placeholder images (replace later!)

2. FROM FER2013:
   python synthetic_data_gen.py --mode fer2013 \\
       --source ./data/fer2013_raw \\
       --output ./data/synthetic_videos
   
   → Converts FER2013 static images to videos
   → 3-5 minutes to generate
   → Production quality

3. CUSTOM DATASET:
   python synthetic_data_gen.py --mode custom \\
       --source ./data/my_emotions \\
       --output ./data/my_videos
   
   → Use your own emotion dataset
   → Must have folders: neutral/, happy/, sad/, etc.

Options:
   --frames N    : Frames per video (default: 8)
   --videos N    : Videos per identity (default: 3)
        """)
        sys.exit(0)
    
    main()
