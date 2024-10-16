import os
import json
import itertools
from PIL import Image

# Define the categories
categories = [
    {"supercategory": "none", "name": "Person", "id": 0},
    {"supercategory": "none", "name": "Ear", "id": 1},
    {"supercategory": "none", "name": "Earmuffs", "id": 2},
    {"supercategory": "none", "name": "Face", "id": 3},
    {"supercategory": "none", "name": "Face-guard", "id": 4},
    {"supercategory": "none", "name": "Face-mask-medical", "id": 5},
    {"supercategory": "none", "name": "Foot", "id": 6},
    {"supercategory": "none", "name": "Tools", "id": 7},
    {"supercategory": "none", "name": "Glasses", "id": 8},
    {"supercategory": "none", "name": "Gloves", "id": 9},
    {"supercategory": "none", "name": "Helmet", "id": 10},
    {"supercategory": "none", "name": "Hands", "id": 11},
    {"supercategory": "none", "name": "Head", "id": 12},
    {"supercategory": "none", "name": "Medical-suit", "id": 13},
    {"supercategory": "none", "name": "Shoes", "id": 14},
    {"supercategory": "none", "name": "Safety-suit", "id": 15},
    {"supercategory": "none", "name": "Safety-vest", "id": 16}
]

# Define phases (train/validation)
phases = ["train", "valid"]

for phase in phases:
    img_folder = f"{phase}/images"
    label_folder = f"{phase}/labels"
    json_file = f"{phase}.json"

    res_file = {
        "categories": categories,
        "images": [],
        "annotations": []
    }

    image_id = 0
    annot_count = 0

    # Iterate through image and corresponding label files
    for img_filename in os.listdir(img_folder):
        if img_filename.endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(img_folder, img_filename)
            label_path = os.path.join(label_folder,
                                      img_filename.replace('.jpg', '.txt').replace('.jpeg', '.txt').replace('.png',
                                                                                                            '.txt'))

            # Open image to get dimensions
            img = Image.open(img_path)
            img_w, img_h = img.size

            # Add image details to COCO JSON
            img_elem = {
                "file_name": img_filename,
                "height": img_h,
                "width": img_w,
                "id": image_id
            }
            res_file["images"].append(img_elem)

            # Parse label file
            with open(label_path, "r") as label_file:
                for line in label_file:
                    class_id, x_center, y_center, width, height = map(float, line.split())

                    # Convert YOLO format to COCO format
                    xmin = (x_center - width / 2) * img_w
                    ymin = (y_center - height / 2) * img_h
                    box_w = width * img_w
                    box_h = height * img_h
                    area = box_w * box_h

                    # Create polygon for segmentation
                    poly = [
                        [xmin, ymin],
                        [xmin + box_w, ymin],
                        [xmin + box_w, ymin + box_h],
                        [xmin, ymin + box_h]
                    ]

                    # Add annotation
                    annot_elem = {
                        "id": annot_count,
                        "image_id": image_id,
                        "category_id": int(class_id),
                        "bbox": [xmin, ymin, box_w, box_h],
                        "area": area,
                        "segmentation": [list(itertools.chain.from_iterable(poly))],
                        "iscrowd": 0
                    }
                    res_file["annotations"].append(annot_elem)
                    annot_count += 1

            image_id += 1

    # Write results to JSON
    with open(json_file, "w") as f:
        json.dump(res_file, f, indent=4)

    print(f"Processed {image_id} {phase} images.")
