import os
import torch
import torchvision.transforms as T
import matplotlib.pyplot as plt
from PIL import Image
import time
import math
import random
import json
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")


# Check if CUDA is available and select the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load model from torch.hub and load ckpt file into model
TRAINED_CKPT_PATH = './detr/outputs/checkpoint.pth'
checkpoint = torch.load(TRAINED_CKPT_PATH, map_location=device)
model = torch.hub.load('facebookresearch/detr', 'detr_resnet50', pretrained=False, num_classes=17)
model.load_state_dict(checkpoint['model'], strict=False)
model = model.to(device)  # Move the model to GPU if available
model.eval()  # Set model to evaluation mode

CLASSES = [
    'Person', 'Ear', 'Earmuffs', 'Face', 'Face-guard', 'Face-mask-medical', 'Foot',
    'Tools', 'Glasses', 'Gloves', 'Helmet', 'Hands', 'Head', 'Medical-suit', 'Shoes',
    'Safety-suit', 'Safety-vest'
]

# colors for visualization
COLORS = [
    [0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
    [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933],
    [0.635, 0.078, 0.184], [0.300, 0.300, 0.300], [0.600, 0.600, 0.600],
    [1.000, 0.000, 0.000], [1.000, 0.500, 0.000], [0.749, 0.749, 0.000],
    [0.000, 1.000, 0.000], [0.000, 0.000, 1.000], [0.667, 0.000, 1.000],
    [0.333, 0.333, 0.000], [0.000, 0.333, 0.333]
]

transform = T.Compose([
    T.Resize(800),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


# for output bounding box post-processing
def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)


def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32).to(device)
    return b


def plot_results(pil_img, prob, boxes, output_dir, img_name):
    # Create the figure and axes
    plt.figure(figsize=(16, 10))
    plt.imshow(pil_img)
    ax = plt.gca()

    # Use the colors for the bounding boxes
    colors = COLORS * 100
    for p, (xmin, ymin, xmax, ymax), c in zip(prob, boxes.tolist(), colors):
        ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                   fill=False, color=c, linewidth=3))
        cl = p.argmax()
        text = f'{CLASSES[cl]}: {p[cl]:0.2f}'
        ax.text(xmin, ymin, text, fontsize=15,
                bbox=dict(facecolor='yellow', alpha=0.5))

    # Remove axis
    plt.axis('off')

    # Save the plot as an image without white borders
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, img_name)
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close()


def postprocess_img(img_path, output_dir, json_results):
    im = Image.open(img_path)

    # mean-std normalize the input image (batch-size: 1)
    img = transform(im).unsqueeze(0).to(device)  # Move the input image to GPU

    # propagate through the model
    start = time.time()
    outputs = model(img)
    end = time.time()

    # keep only predictions with 0.9+ confidence
    probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]
    keep = probas.max(-1).values > 0.9

    # convert boxes from [0; 1] to image scales
    bboxes_scaled = rescale_bboxes(outputs['pred_boxes'][0, keep], im.size)

    # Get image file name
    img_name = os.path.basename(img_path)

    # Collect bounding boxes and labels
    boxes_list = bboxes_scaled.tolist()
    labels_list = probas[keep].argmax(-1).tolist()

    # Add data to the JSON result
    json_results[img_name] = {"boxes": boxes_list, "labels": labels_list}

    # Optionally, save the image with predictions
    plot_results(im, probas[keep], bboxes_scaled, output_dir, img_name)


# Load test image paths
TEST_IMG_PATH = './valid/images'
OUTPUT_DIR = './tmp_results/'  # Directory to save the output images
JSON_OUTPUT_PATH = './tmp_results/predictions_10.json'  # Path to save JSON results

os.makedirs(OUTPUT_DIR, exist_ok=True)

img_format = {'jpg', 'png', 'jpeg'}
paths = [os.path.join(TEST_IMG_PATH, obj.name) for obj in os.scandir(TEST_IMG_PATH) if obj.name.split(".")[-1].lower() in img_format]

print('Total number of test images: ', len(paths))

# Prepare JSON dictionary
json_results = {}

# Process and save images
for img_path in tqdm(paths[:10]):
    postprocess_img(img_path, OUTPUT_DIR, json_results)

# Save the results to a JSON file
with open(JSON_OUTPUT_PATH, 'w') as f:
    json.dump(json_results, f, indent=4)

print(f'Results saved to {JSON_OUTPUT_PATH}')
