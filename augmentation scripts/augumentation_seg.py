import os
import cv2
import albumentations as A
from tqdm import tqdm

def read_yolo_segmentation_label(label_path):
    with open(label_path, 'r') as f:
        lines = f.read().strip().split('\n')
    polygons = []
    class_ids = []
    for line in lines:
        parts = line.strip().split()
        class_id = int(parts[0])
        coords = list(map(float, parts[1:]))
        points = [(coords[i], coords[i+1]) for i in range(0, len(coords), 2)]
        polygons.append(points)
        class_ids.append(class_id)
    return polygons, class_ids

def write_yolo_segmentation_label(label_path, polygons, class_ids):
    with open(label_path, 'w') as f:
        for cls, poly in zip(class_ids, polygons):
            coords = ' '.join([f"{x:.6f} {y:.6f}" for x, y in poly])
            f.write(f"{cls} {coords}\n")

def augment_image(img, polygons, aug):
    h, w = img.shape[:2]
    # Convert to absolute coordinates for Albumentations
    converted_polys = [[(int(x * w), int(y * h)) for x, y in poly] for poly in polygons]
    annotations = {'image': img, 'polygons': converted_polys}
    augmented = aug(image=img, polygons=converted_polys)
    
    aug_img = augmented['image']
    aug_polys = augmented['polygons']
    
    # Convert back to normalized coordinates
    norm_polys = [[(x / w, y / h) for x, y in poly] for poly in aug_polys]
    return aug_img, norm_polys

# Define your augmentations
augmentations = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.3),
    A.RandomRotate90(p=0.5),
    A.ColorJitter(p=0.5),
    A.RandomBrightnessContrast(p=0.5),
    A.GaussianBlur(p=0.2),
], p=1.0, keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))

# Directories
img_dir = r"C://Users//sakth//OneDrive//Desktop//Agrithon//CropHealthMultimodal//data//annotations//yolo_disease//images//train"
lbl_dir = r"C://Users//sakth//OneDrive//Desktop//Agrithon//CropHealthMultimodal//data//annotations//yolo_disease//labels//train"

aug_img_dir = img_dir + "_aug"
aug_lbl_dir = lbl_dir + "_aug"

os.makedirs(aug_img_dir, exist_ok=True)
os.makedirs(aug_lbl_dir, exist_ok=True)

# Process each image
for img_name in tqdm(os.listdir(img_dir)):
    if not img_name.endswith(".jpg"):
        continue
    img_path = os.path.join(img_dir, img_name)
    lbl_path = os.path.join(lbl_dir, img_name.replace(".jpg", ".txt"))

    img = cv2.imread(img_path)
    polygons, class_ids = read_yolo_segmentation_label(lbl_path)

    for i in range(3):  # create 3 augmented versions
        aug_img, aug_polys = augment_image(img, polygons, augmentations)
        new_name = img_name.replace(".jpg", f"_aug{i}.jpg")
        new_lbl = img_name.replace(".jpg", f"_aug{i}.txt")

        cv2.imwrite(os.path.join(aug_img_dir, new_name), aug_img)
        write_yolo_segmentation_label(os.path.join(aug_lbl_dir, new_lbl), aug_polys, class_ids)
