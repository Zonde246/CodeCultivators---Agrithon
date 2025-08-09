import os
import cv2
import albumentations as A

# Define the augmentation pipeline
bbox_transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.3),
    A.RandomBrightnessContrast(p=0.5),
    A.Rotate(limit=20, p=0.5),
    A.ColorJitter(p=0.4),
    A.MotionBlur(p=0.2)
], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

def augment_bounding_boxes(img_dir, label_dir, output_img_dir, output_label_dir, num_aug=3):
    os.makedirs(output_img_dir, exist_ok=True)
    os.makedirs(output_label_dir, exist_ok=True)

    for file in os.listdir(img_dir):
        if not file.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        img_path = os.path.join(img_dir, file)
        lbl_path = os.path.join(label_dir, os.path.splitext(file)[0] + ".txt")

        if not os.path.exists(lbl_path):
            print(f"⚠️ Missing label for {file}, skipping.")
            continue

        image = cv2.imread(img_path)
        with open(lbl_path, 'r') as f:
            lines = f.read().splitlines()

        bboxes, class_labels = [], []
        for line in lines:
            cls, xc, yc, bw, bh = map(float, line.split())
            bboxes.append([xc, yc, bw, bh])
            class_labels.append(int(cls))

        for i in range(num_aug):
            augmented = bbox_transform(image=image, bboxes=bboxes, class_labels=class_labels)
            aug_image = augmented['image']
            aug_bboxes = augmented['bboxes']
            aug_labels = augmented['class_labels']

            save_img_path = os.path.join(output_img_dir, f"{os.path.splitext(file)[0]}_aug{i}.jpg")
            save_lbl_path = os.path.join(output_label_dir, f"{os.path.splitext(file)[0]}_aug{i}.txt")

            cv2.imwrite(save_img_path, aug_image)

            with open(save_lbl_path, 'w') as f:
                for idx, bbox in enumerate(aug_bboxes):
                    xc, yc, bw, bh = bbox
                    f.write(f"{aug_labels[idx]} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")

    print("✅ Augmentation complete for insect images.")

# Example usage
augment_bounding_boxes(
    img_dir= r"C://Users//sakth//OneDrive//Desktop//Agrithon//CropHealthMultimodal//data//annotations//yolo_insect//images//train",
    label_dir= r"C://Users//sakth//OneDrive//Desktop//Agrithon//CropHealthMultimodal//data//annotations//yolo_insect//labels//train",
    output_img_dir= r"C://Users//sakth//OneDrive//Desktop//Agrithon//CropHealthMultimodal//data//annotations//yolo_insect//images//train_aug",
    output_label_dir= r"C://Users//sakth//OneDrive//Desktop//Agrithon//CropHealthMultimodal//data//annotations//yolo_insect//labels//train_aug"
)
