from groundingdino.util.inference import load_model, load_image, predict, annotate, Model
import cv2
import torch

def calculate_iou(box1, box2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.

    Parameters:
    - box1, box2: lists of [center_x, center_y, width, height]

    Returns:
    - iou: the IoU of the two bounding boxes.
    """
    # Convert from center coordinates to box corners
    x1_min = box1[0] - box1[2] / 2
    x1_max = box1[0] + box1[2] / 2
    y1_min = box1[1] - box1[3] / 2
    y1_max = box1[1] + box1[3] / 2

    x2_min = box2[0] - box2[2] / 2
    x2_max = box2[0] + box2[2] / 2
    y2_min = box2[1] - box2[3] / 2
    y2_max = box2[1] + box2[3] / 2

    # Calculate intersection
    inter_area = max(0, min(x1_max, x2_max) - max(x1_min, x2_min)) * max(0, min(y1_max, y2_max) - max(y1_min, y2_min))

    # Calculate union
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = box1_area + box2_area - inter_area

    # Calculate IoU
    iou = inter_area / union_area

    return iou


CONFIG_PATH = "groundingdino/config/GroundingDINO_SwinT_OGC.py"
CHECKPOINT_PATH = "./weights/groundingdino_swint_ogc.pth"
DEVICE = "cuda"
IMAGE_PATH = "example.jpg" # ".asset/cat_dog.jpeg" 
TEXT_PROMPT = "yellow yellow blocks. blue blocks." # "Horse. Clouds. Grasses. Sky. Hill."
BOX_TRESHOLD = 0.3
TEXT_TRESHOLD = 0.2
FP16_INFERENCE = False
iou_threshold = 0.3
image_source, image = load_image(IMAGE_PATH)
model = load_model(CONFIG_PATH, CHECKPOINT_PATH)

if FP16_INFERENCE:
    image = image.half()
    model = model.half()
# print(f"Image: {image} image_source:{image_source.shape}")
boxes, logits, phrases = predict(
    model=model,
    image=image,
    caption=TEXT_PROMPT,
    box_threshold=BOX_TRESHOLD,
    text_threshold=TEXT_TRESHOLD,
    device=DEVICE,
)

remaining_boxes = []
boxes = list(boxes)
while boxes:
    box = boxes.pop(0)
    boxes = [b for b in boxes if calculate_iou(box, b) <= iou_threshold]
    remaining_boxes.append(box)
print(f"Remaining boxes: {remaining_boxes}")
boxes = torch.stack(remaining_boxes)
# selected_idx = []
# for i in range(len(logits)):
#     if logits[i] > 0.5:
#         selected_idx.append(i)

# boxes = boxes[selected_idx]
# logits = torch.tensor([logits[i] for i in selected_idx])
# phrases = [phrases[i] for i in selected_idx]
print(f"Boxes: {boxes}")
print(f"Logits: {logits}")
print(f"Phrases: {phrases}")
annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
cv2.imwrite("annotated_image.jpg", annotated_frame)