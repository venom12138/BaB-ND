from groundingdino.util.inference import load_model, load_image, predict, annotate, Model
import cv2
import torch
import os
import sys

from real_exp_new.utils import nms # real_exp_new
import groundingdino.datasets.transforms as T
import numpy as np
from PIL import Image
class DinoWrapper:
    def __init__(self, obj_num, classes):
        CONFIG_PATH = "GroundingDINO/groundingdino/config/GroundingDINO_SwinB_cfg.py"
        CHECKPOINT_PATH = "GroundingDINO/weights/groundingdino_swinb_cogcoor.pth"
        self.DEVICE = "cuda"
        self.TEXT_PROMPTs = ["one square orange cube block.", "one square blue cube block.", "one red block."]
        # ["one orange cylinder.", "one dark blue cylinder.", "one red block."] # yellow "black T. " # # "Horse. Clouds. Grasses. Sky. Hill."
        self.BOX_TRESHOLD = 0.1
        self.TEXT_TRESHOLD = 0.2
        self.iou_threshold = 0.5
        self.obj_num = obj_num
        classes = np.array(classes)
        self.classes = classes
        cls_num = len(np.unique(classes))
        self.TEXT_PROMPTs = self.TEXT_PROMPTs[:cls_num]
        self.model = load_model(CONFIG_PATH, CHECKPOINT_PATH)
    
    def predict(self, img, file_name=None):
        image_source, image = self.img_transform(img)
        output_boxes = []
        output_logits = []
        output_phrases = []
        for i, text_prompt in enumerate(self.TEXT_PROMPTs):
            boxes, logits, phrases = predict(
                model=self.model,
                image=image,
                caption=text_prompt,
                box_threshold=self.BOX_TRESHOLD,
                text_threshold=self.TEXT_TRESHOLD,
                device=self.DEVICE,
            )
            obj_num_in_cls = sum(self.classes == i)
            box_idx = nms(boxes, logits, self.iou_threshold, obj_num_in_cls)[:obj_num_in_cls]
            
            annotated_frame = annotate(image_source=image_source, boxes=boxes, \
                                    logits=logits, phrases=phrases)
            cv2.imwrite(f"raw_annotated_image_{text_prompt}.jpg", annotated_frame)
            
            output_phrases.extend([phrases[i] for i in box_idx])
            output_logits.extend(logits[box_idx])
            output_boxes.extend(boxes[box_idx])
            
            annotated_frame = annotate(image_source=image_source, boxes=boxes[box_idx], \
                                    logits=logits[box_idx], phrases=[phrases[i] for i in box_idx])
            cv2.imwrite(f"annotated_image_{text_prompt}.jpg", annotated_frame)
            if len(box_idx) != obj_num_in_cls:
                print(f"Warning: objects lost for {text_prompt}")

        boxes = torch.stack(output_boxes)
        logits = torch.stack(output_logits)
        phrases = output_phrases
        
        annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
        if file_name is not None:
            cv2.imwrite(file_name, annotated_frame)
        else:
            cv2.imwrite("annotated_image.jpg", annotated_frame)
        h, w = image_source.shape[:2]
        
        return torch.flip(boxes[:,:2] * torch.tensor([w, h]), [1]) #cx cy
    
    def img_transform(self, image_source):
        transform = T.Compose(
            [
                T.RandomResize([800], max_size=1333),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        
        image_source *= 255
        image_source = image_source.astype(np.uint8)
        image_source = Image.fromarray(image_source)
        # print(f"image_source:{image_source}")
        image = np.asarray(image_source)
        # print(f"image_source.shape:{image_source.shape}")
        image_transformed, _ = transform(image_source, None)
        return image, image_transformed