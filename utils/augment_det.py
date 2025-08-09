import os
import json
import cv2
import numpy as np
from PIL import Image, ImageEnhance
import albumentations as A
from albumentations.core.composition import BboxParams, KeypointParams
import random
from pathlib import Path
import argparse

class DatasetAugmenter:
    def __init__(self, input_dir, output_dir, annotation_format='coco'):
        """
        Initialize the dataset augmenter
        
        Args:
            input_dir: Directory containing images and annotations
            output_dir: Directory to save augmented images and annotations
            annotation_format: 'coco', 'yolo', or 'pascal_voc'
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.annotation_format = annotation_format
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Define augmentation pipeline
        self.augmentation_pipeline = A.Compose([
            # Geometric transformations
            A.HorizontalFlip(p=0.5),
            A.RandomRotate90(p=0.3),
            A.Rotate(limit=15, p=0.3),
            A.RandomScale(scale_limit=0.2, p=0.3),
            A.ShiftScaleRotate(
                shift_limit=0.1,
                scale_limit=0.1,
                rotate_limit=10,
                p=0.3
            ),
            
            # Color/lighting augmentations
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=0.5
            ),
            A.HueSaturationValue(
                hue_shift_limit=10,
                sat_shift_limit=20,
                val_shift_limit=10,
                p=0.3
            ),
            A.RandomGamma(gamma_limit=(80, 120), p=0.3),
            A.CLAHE(clip_limit=2.0, p=0.3),
            
            # Noise and blur
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
            A.OneOf([
                A.MotionBlur(blur_limit=3, p=0.3),
                A.MedianBlur(blur_limit=3, p=0.3),
                A.GaussianBlur(blur_limit=3, p=0.3),
            ], p=0.2),
            
            # Weather effects
            A.OneOf([
                A.RandomRain(p=0.1),
                A.RandomFog(p=0.1),
                A.RandomSunFlare(p=0.05),
            ], p=0.1),
            
        ], bbox_params=BboxParams(
            format='coco',
            label_fields=['class_labels'],
            min_area=1.0,  # Remove very small bboxes
            min_visibility=0.1  # Remove mostly occluded bboxes
        ), keypoint_params=KeypointParams(
            format='xy',
            label_fields=['keypoint_labels']
        ) if annotation_format in ['coco', 'keypoints'] else None)
    
    def clip_bboxes(self, bboxes, img_width, img_height):
        """
        Clip bounding boxes to image boundaries and remove invalid ones
        
        Args:
            bboxes: List of bboxes in COCO format [x, y, width, height]
            img_width: Image width
            img_height: Image height
            
        Returns:
            Clipped and filtered bboxes
        """
        clipped_bboxes = []
        
        for bbox in bboxes:
            x, y, width, height = bbox
            
            # Clip coordinates to image boundaries
            x = max(0, min(x, img_width))
            y = max(0, min(y, img_height))
            
            # Calculate clipped width and height
            x_max = min(x + width, img_width)
            y_max = min(y + height, img_height)
            
            new_width = x_max - x
            new_height = y_max - y
            
            # Only keep bbox if it has meaningful area (at least 1x1 pixels)
            if new_width > 1 and new_height > 1:
                clipped_bboxes.append([x, y, new_width, new_height])
        
        return clipped_bboxes
    
    def load_coco_annotations(self, annotation_file):
        """Load COCO format annotations"""
        with open(annotation_file, 'r') as f:
            return json.load(f)
    
    def load_yolo_annotations(self, annotation_file, img_width, img_height):
        """Load YOLO format annotations and convert to COCO format"""
        bboxes = []
        class_labels = []
        
        if os.path.exists(annotation_file):
            with open(annotation_file, 'r') as f:
                for line in f.readlines():
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        class_id = int(parts[0])
                        x_center = float(parts[1]) * img_width
                        y_center = float(parts[2]) * img_height
                        width = float(parts[3]) * img_width
                        height = float(parts[4]) * img_height
                        
                        # Convert to COCO format (x, y, width, height)
                        x = x_center - width / 2
                        y = y_center - height / 2
                        
                        bboxes.append([x, y, width, height])
                        class_labels.append(class_id)
        
        # Clip bboxes to image boundaries
        bboxes = self.clip_bboxes(bboxes, img_width, img_height)
        # Adjust class_labels to match filtered bboxes
        if len(bboxes) != len(class_labels):
            class_labels = class_labels[:len(bboxes)]
        
        return bboxes, class_labels
    
    def save_yolo_annotations(self, bboxes, class_labels, output_file, img_width, img_height):
        """Save annotations in YOLO format"""
        with open(output_file, 'w') as f:
            for bbox, class_id in zip(bboxes, class_labels):
                x, y, width, height = bbox
                
                # Convert to YOLO format (normalized)
                x_center = (x + width / 2) / img_width
                y_center = (y + height / 2) / img_height
                norm_width = width / img_width
                norm_height = height / img_height
                
                # Ensure normalized values are within [0, 1]
                x_center = max(0, min(1, x_center))
                y_center = max(0, min(1, y_center))
                norm_width = max(0, min(1, norm_width))
                norm_height = max(0, min(1, norm_height))
                
                f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {norm_width:.6f} {norm_height:.6f}\n")
    
    def augment_image(self, image_path, bboxes=None, class_labels=None, keypoints=None, keypoint_labels=None):
        """
        Augment a single image with its annotations
        
        Args:
            image_path: Path to the image
            bboxes: List of bounding boxes in COCO format [x, y, width, height]
            class_labels: List of class labels for bboxes
            keypoints: List of keypoints in (x, y) format
            keypoint_labels: List of labels for keypoints
        
        Returns:
            Augmented image and transformed annotations
        """
        # Load image
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_height, img_width = image.shape[:2]
        
        # Pre-clip input bboxes to ensure they're valid
        if bboxes:
            bboxes = self.clip_bboxes(bboxes, img_width, img_height)
            # Adjust class_labels to match filtered bboxes
            if len(bboxes) != len(class_labels):
                class_labels = class_labels[:len(bboxes)]
        
        # Prepare data for augmentation
        transform_data = {'image': image}
        
        if bboxes:
            transform_data['bboxes'] = bboxes
            transform_data['class_labels'] = class_labels or [0] * len(bboxes)
        
        if keypoints:
            transform_data['keypoints'] = keypoints
            transform_data['keypoint_labels'] = keypoint_labels or [0] * len(keypoints)
        
        # Apply augmentation
        try:
            transformed = self.augmentation_pipeline(**transform_data)
            
            # Post-process: clip transformed bboxes
            aug_bboxes = transformed.get('bboxes', [])
            aug_class_labels = transformed.get('class_labels', [])
            
            if aug_bboxes:
                aug_img_height, aug_img_width = transformed['image'].shape[:2]
                aug_bboxes = self.clip_bboxes(aug_bboxes, aug_img_width, aug_img_height)
                # Adjust class labels to match filtered bboxes
                aug_class_labels = aug_class_labels[:len(aug_bboxes)]
            
            return (
                transformed['image'],
                aug_bboxes,
                aug_class_labels,
                transformed.get('keypoints', []),
                transformed.get('keypoint_labels', [])
            )
        except Exception as e:
            print(f"Error augmenting {image_path}: {e}")
            return None, None, None, None, None
    
    def process_dataset(self, num_augmentations=3, image_extensions=('.jpg', '.jpeg', '.png')):
        """
        Process entire dataset and create augmented versions
        
        Args:
            num_augmentations: Number of augmented versions per original image
            image_extensions: Tuple of valid image file extensions
        """
        print(f"Processing dataset from {self.input_dir}")
        print(f"Output directory: {self.output_dir}")
        
        # Find all images
        image_files = []
        for ext in image_extensions:
            image_files.extend(self.input_dir.glob(f"*{ext}"))
            image_files.extend(self.input_dir.glob(f"*{ext.upper()}"))
        
        print(f"Found {len(image_files)} images")
        
        for img_idx, image_path in enumerate(image_files):
            print(f"Processing {img_idx + 1}/{len(image_files)}: {image_path.name}")
            
            # Copy original image
            original_output = self.output_dir / image_path.name
            Image.open(image_path).save(original_output)
            
            # Load annotations based on format
            bboxes, class_labels = [], []
            keypoints, keypoint_labels = [], []
            
            if self.annotation_format == 'yolo':
                annotation_file = image_path.with_suffix('.txt')
                img = Image.open(image_path)
                img_width, img_height = img.size
                
                if annotation_file.exists():
                    bboxes, class_labels = self.load_yolo_annotations(
                        annotation_file, img_width, img_height
                    )
                    # Copy original annotation
                    original_ann_output = self.output_dir / annotation_file.name
                    with open(annotation_file, 'r') as src, open(original_ann_output, 'w') as dst:
                        dst.write(src.read())
            
            elif self.annotation_format == 'coco':
                # For COCO, you'd typically have a single JSON file for the entire dataset
                # This is a simplified version - adapt based on your COCO structure
                annotation_file = image_path.with_suffix('.json')
                if annotation_file.exists():
                    with open(annotation_file, 'r') as f:
                        ann_data = json.load(f)
                        bboxes = ann_data.get('bboxes', [])
                        class_labels = ann_data.get('class_labels', [])
                        keypoints = ann_data.get('keypoints', [])
                        keypoint_labels = ann_data.get('keypoint_labels', [])
            
            # Generate augmented versions
            for aug_idx in range(num_augmentations):
                aug_image, aug_bboxes, aug_class_labels, aug_keypoints, aug_keypoint_labels = \
                    self.augment_image(image_path, bboxes, class_labels, keypoints, keypoint_labels)
                
                if aug_image is not None:
                    # Save augmented image
                    stem = image_path.stem
                    ext = image_path.suffix
                    aug_image_name = f"{stem}_aug_{aug_idx + 1}{ext}"
                    aug_image_path = self.output_dir / aug_image_name
                    
                    aug_image_pil = Image.fromarray(aug_image)
                    aug_image_pil.save(aug_image_path)
                    
                    # Save augmented annotations
                    if self.annotation_format == 'yolo' and aug_bboxes:
                        aug_ann_name = f"{stem}_aug_{aug_idx + 1}.txt"
                        aug_ann_path = self.output_dir / aug_ann_name
                        self.save_yolo_annotations(
                            aug_bboxes, aug_class_labels, aug_ann_path,
                            aug_image.shape[1], aug_image.shape[0]
                        )
                    
                    elif self.annotation_format == 'coco':
                        aug_ann_name = f"{stem}_aug_{aug_idx + 1}.json"
                        aug_ann_path = self.output_dir / aug_ann_name
                        aug_ann_data = {
                            'bboxes': aug_bboxes,
                            'class_labels': aug_class_labels,
                            'keypoints': aug_keypoints,
                            'keypoint_labels': aug_keypoint_labels
                        }
                        with open(aug_ann_path, 'w') as f:
                            json.dump(aug_ann_data, f, indent=2)
        
        print(f"Dataset augmentation complete! Check {self.output_dir}")

def create_custom_pipeline():
    """
    Create a custom augmentation pipeline - modify this function for your specific needs
    """
    return A.Compose([
        # Add your custom augmentations here
        A.RandomCrop(height=512, width=512, p=0.3),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.4),
        A.HueSaturationValue(p=0.3),
        A.GaussNoise(p=0.2),
        A.Rotate(limit=10, p=0.3),
    ], bbox_params=BboxParams(format='coco', label_fields=['class_labels']))

def main():
    parser = argparse.ArgumentParser(description='Augment annotated image dataset')
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Input directory containing images and annotations')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for augmented dataset')
    parser.add_argument('--format', type=str, choices=['coco', 'yolo', 'pascal_voc'],
                        default='yolo', help='Annotation format')
    parser.add_argument('--num_augs', type=int, default=3,
                        help='Number of augmentations per image')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    # Create augmenter
    augmenter = DatasetAugmenter(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        annotation_format=args.format
    )
    
    # Process dataset
    augmenter.process_dataset(num_augmentations=args.num_augs)

if __name__ == "__main__":
    # Example usage without command line args
    # augmenter = DatasetAugmenter(
    #     input_dir="./dataset/images",
    #     output_dir="./dataset/augmented",
    #     annotation_format="yolo"
    # )
    # augmenter.process_dataset(num_augmentations=5)
    
    main()

# Additional utility functions for specific use cases

def batch_augment_with_config(input_dir, output_dir, config_file):
    """
    Augment dataset using a configuration file
    
    Example config.json:
    {
        "augmentations": {
            "horizontal_flip": {"p": 0.5},
            "rotate": {"limit": 15, "p": 0.3},
            "brightness_contrast": {"brightness_limit": 0.2, "contrast_limit": 0.2, "p": 0.4}
        },
        "num_augmentations": 5,
        "annotation_format": "yolo"
    }
    """
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    # Build custom pipeline from config
    transforms = []
    aug_config = config.get('augmentations', {})
    
    if 'horizontal_flip' in aug_config:
        transforms.append(A.HorizontalFlip(**aug_config['horizontal_flip']))
    if 'rotate' in aug_config:
        transforms.append(A.Rotate(**aug_config['rotate']))
    if 'brightness_contrast' in aug_config:
        transforms.append(A.RandomBrightnessContrast(**aug_config['brightness_contrast']))
    
    custom_pipeline = A.Compose(transforms, 
                               bbox_params=BboxParams(format='coco', label_fields=['class_labels']))
    
    augmenter = DatasetAugmenter(input_dir, output_dir, config['annotation_format'])
    augmenter.augmentation_pipeline = custom_pipeline
    augmenter.process_dataset(num_augmentations=config.get('num_augmentations', 3))

def preview_augmentations(image_path, annotation_path=None, num_previews=4):
    """
    Preview augmentations on a single image
    """
    import matplotlib.pyplot as plt
    
    augmenter = DatasetAugmenter(".", ".", "yolo")
    
    # Load image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Load annotations if provided
    bboxes, class_labels = [], []
    if annotation_path and os.path.exists(annotation_path):
        h, w = image.shape[:2]
        bboxes, class_labels = augmenter.load_yolo_annotations(annotation_path, w, h)
    
    # Create preview
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    axes = axes.flatten()
    
    for i in range(num_previews):
        aug_image, aug_bboxes, _, _, _ = augmenter.augment_image(
            image_path, bboxes, class_labels
        )
        
        if aug_image is not None:
            axes[i].imshow(aug_image)
            
            # Draw bounding boxes
            for bbox in aug_bboxes:
                x, y, w, h = bbox
                rect = plt.Rectangle((x, y), w, h, fill=False, color='red', linewidth=2)
                axes[i].add_patch(rect)
            
            axes[i].set_title(f'Augmentation {i+1}')
            axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()

# Installation requirements:
# pip install albumentations opencv-python pillow numpy matplotlib