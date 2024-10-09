import cv2
import argparse
import time
import numpy as np
import torch
import open3d as o3d
from PIL import Image

from segment_anything import SamPredictor, sam_model_registry
import groundingdino.datasets.transforms as T
from groundingdino.models import build_model
from groundingdino.util import box_ops
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
from groundingdino.util.inference import annotate
import pickle


def depth2fgpcd(depth, intrinsic_matrix):
    H, W = depth.shape
    x, y = np.meshgrid(np.arange(W), np.arange(H))
    x = x.reshape(-1)
    y = y.reshape(-1)
    depth = depth.reshape(-1)
    points = np.stack([x, y, np.ones_like(x)], axis=1)
    points = points * depth[:, None]
    points = points @ np.linalg.inv(intrinsic_matrix).T
    return points


class PerceptionModule:
    def __init__(self, device="cuda:0"):
        self.device = device
        self.det_model = None
        self.sam_model = None
        self.load_model()

    def load_model(self):
        if self.det_model is not None:
            print("Model already loaded")
            return
        device = self.device
        det_model = build_model(SLConfig.fromfile(
            './GroundingDINO/groundingdino/config/GroundingDINO_SwinB_cfg.py'))
        checkpoint = torch.load('./GroundingDINO/weights/groundingdino_swinb_cogcoor.pth', map_location="cpu")
        det_model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
        det_model.eval()
        det_model = det_model.to(device)
        
        sam = sam_model_registry["default"](checkpoint='weights/sam_vit_h_4b8939.pth')
        sam_model = SamPredictor(sam)
        sam_model.model = sam_model.model.to(device)

        self.det_model = det_model
        self.sam_model = sam_model

    def detect(self, image, captions, box_thresholds):  # captions: list
        raw_image = image.copy()
        image = Image.fromarray(image)
        for i, caption in enumerate(captions):
            caption = caption.lower()
            caption = caption.strip()
            if not caption.endswith("."):
                caption = caption + "."
            captions[i] = caption
        num_captions = len(captions)

        transform = T.Compose([
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        image_tensor, _ = transform(image, None)  # 3, h, w

        image_tensor = image_tensor[None].repeat(num_captions, 1, 1, 1).to(self.device)

        with torch.no_grad():
            outputs = self.det_model(image_tensor, captions=captions)
        logits = outputs["pred_logits"].sigmoid()  # (num_captions, nq, 256)
        boxes = outputs["pred_boxes"]  # (num_captions, nq, 4)

        # filter output
        if isinstance(box_thresholds, list):
            filt_mask = logits.max(dim=2)[0] > torch.tensor(box_thresholds).to(device=self.device, dtype=logits.dtype)[:, None]
        else:
            filt_mask = logits.max(dim=2)[0] > box_thresholds
        labels = torch.ones((*logits.shape[:2], 1)) * torch.arange(logits.shape[0])[:, None, None]  # (num_captions, nq, 1)
        labels = labels.to(device=self.device, dtype=logits.dtype)  # (num_captions, nq, 1)
        logits = logits[filt_mask] # num_filt, 256
        boxes = boxes[filt_mask] # num_filt, 4
        labels = labels[filt_mask].reshape(-1).to(torch.int64) # num_filt,
        scores = logits.max(dim=1)[0] # num_filt,

        for box, score, label in zip(boxes, scores, labels):
            box = [round(i, 2) for i in box.tolist()]
            print(f"Detected {captions[label.item()]} with confidence {round(score.item(), 3)} at location {box}")
        
        if not hasattr(self, 'idx'):
            self.idx = 0

        annotated_frame = annotate(raw_image, boxes.cpu().detach(), scores.cpu().detach(), [captions[i] for i in labels.cpu().detach().tolist()])
        cv2.imwrite(f"vis_real_world/detected_{self.idx}.jpg", annotated_frame)
        self.idx += 1
        
        return boxes, scores, labels


    def segment(self, image, boxes, scores, labels, text_prompts):
        # load sam model
        self.sam_model.set_image(image)

        masks, _, _ = self.sam_model.predict_torch(
            point_coords = None,
            point_labels = None,
            boxes = self.sam_model.transform.apply_boxes_torch(boxes, image.shape[:2]), # (n_detection, 4)
            multimask_output = False,
        )

        masks = masks[:, 0, :, :] # (n_detection, H, W)
        # text_labels = ['background']
        text_labels = []
        for category in range(len(text_prompts)):
            text_labels = text_labels + [text_prompts[category].rstrip('.')] * (labels == category).sum().item()
        
        # remove masks where IoU are large
        num_masks = masks.shape[0]
        to_remove = []
        for i in range(num_masks):
            for j in range(i+1, num_masks):
                IoU = (masks[i] & masks[j]).sum().item() / (masks[i] | masks[j]).sum().item()
                if IoU > 0.9:
                    if scores[i].item() > scores[j].item():
                        to_remove.append(j)
                    else:
                        to_remove.append(i)
        to_remove = np.unique(to_remove)
        to_keep = np.setdiff1d(np.arange(num_masks), to_remove)
        to_keep = torch.from_numpy(to_keep).to(device=self.device, dtype=torch.int64)
        masks = masks[to_keep]
        text_labels = [text_labels[i] for i in to_keep]
        # text_labels.insert(0, 'background')
        
        aggr_mask = torch.zeros(masks[0].shape).to(device=self.device, dtype=torch.uint8)
        for obj_i in range(masks.shape[0]):
            aggr_mask[masks[obj_i]] = obj_i + 1

        return (masks, aggr_mask, text_labels), (boxes, scores, labels)


    def get_tabletop_points(self, rgb_list, depth_list, R_list, t_list, intr_list, bbox, depth_threshold=[0, 5], k_filter=1.0):
        obj_name = 'grey woolen rope'

        # increase if out of memory
        stride = 2

        obj_list = ['table', obj_name]
        text_prompts = [f"{obj}" for obj in obj_list]

        pcd_all = o3d.geometry.PointCloud()
        point_colors = [[0, 0, 1], [1, 0, 0], [0, 1, 0], [1, 1, 0], [1, 0, 1]]

        for i in range(len(rgb_list)):
            intr = intr_list[i]
            R_cam2board = R_list[i]
            t_cam2board = t_list[i]

            depth = depth_list[i].copy().astype(np.float32)

            points = depth2fgpcd(depth, intr)
            points = points.reshape(depth.shape[0], depth.shape[1], 3)
            points = points[::stride, ::stride, :].reshape(-1, 3)

            mask = np.logical_and(
                (depth > depth_threshold[0]), (depth < depth_threshold[1])
            )  # (H, W)
            mask = mask[::stride, ::stride].reshape(-1)

            img = rgb_list[i].copy()

            ########## detect and segment ##########
            boxes, scores, labels = self.detect(img, text_prompts, box_thresholds=0.3)

            H, W = img.shape[0], img.shape[1]
            boxes = boxes * torch.Tensor([[W, H, W, H]]).to(device=self.device, dtype=boxes.dtype)
            boxes[:, :2] -= boxes[:, 2:] / 2  # xywh to xyxy
            boxes[:, 2:] += boxes[:, :2]  # xywh to xyxy

            segmentation_results, _ = self.segment(img, boxes, scores, labels, text_prompts)

            masks, aggr_mask, text_labels = segmentation_results
            masks = masks.detach().cpu().numpy()

            mask_table = np.zeros(masks[0].shape, dtype=np.uint8)
            for obj_i in range(masks.shape[0]):
                if text_labels[obj_i] == 'table' or text_labels[obj_i] == 'sheet':
                    mask_table = np.logical_or(mask_table, masks[obj_i])

            # rope
            for obj_i in range(masks.shape[0]):
                if text_labels[obj_i] == obj_name:
                    mask_table = np.logical_and(mask_table, ~masks[obj_i])
            
            mask_obj = np.zeros(masks[0].shape, dtype=np.uint8)
            for obj_i in range(masks.shape[0]):
                if text_labels[obj_i] != 'table' and text_labels[obj_i] != 'sheet':
                    mask_obj = np.logical_or(mask_obj, masks[obj_i])
            
            mask_obj_and_background = 1 - mask_table
            color_palette = np.array([(0, 0, 0), (0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0)])
            aggr_mask = aggr_mask.cpu().detach().numpy()
            num_objs = np.max(aggr_mask)
            color_aggr_mask = np.zeros((aggr_mask.shape[0], aggr_mask.shape[1], 3), dtype=np.uint8)
            for idx in range(num_objs):
                color_aggr_mask[np.where(aggr_mask == text_labels.index(text_labels[idx])+1)] = color_palette[idx]

            cv2.imwrite(f'vis_real_world/{i}_rgb.png', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            cv2.imwrite(f'vis_real_world/{i}_mask_table.png', (mask_table * 255).astype(np.uint8))
            cv2.imwrite(f'vis_real_world/{i}_mask_obj_and_background.png', (mask_obj_and_background * 255).astype(np.uint8))
            cv2.imwrite(f'vis_real_world/{i}_mask_aggr.png', (color_aggr_mask).astype(np.uint8))
            cv2.imwrite(f'vis_real_world/{i}_mask_obj.png', (mask_obj * 255).astype(np.uint8))
            ########################################

            # mask_obj_and_background = mask_obj_and_background.astype(bool)
            # mask_obj_and_background = mask_obj_and_background[::stride, ::stride].reshape(-1)
            # mask = np.logical_and(mask, mask_obj_and_background)

            mask_obj = mask_obj.astype(bool)
            mask_obj = mask_obj[::stride, ::stride].reshape(-1)
            mask = np.logical_and(mask, mask_obj)

            points = points[mask].reshape(-1, 3)

            # points = R_cam2board @ points.T + t_cam2board[:, None]
            # points = points.T  # (N, 3)
            points = points @ R_cam2board + t_cam2board

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)

            colors = img[::stride, ::stride, :].reshape(-1, 3).astype(np.float64)
            colors = colors[mask].reshape(-1, 3)
            colors = colors[:, ::-1].copy()
            pcd.colors = o3d.utility.Vector3dVector(colors / 255)

            pcd_all += pcd

        # crop using bbox
        pcd = pcd_all
        pcd = pcd.crop(o3d.geometry.AxisAlignedBoundingBox(min_bound=bbox[:, 0], max_bound=bbox[:, 1]))

        # downsample
        pcd = pcd.voxel_down_sample(voxel_size=0.0005)
        
        

        # print(f"visualize before outlier removal")
        # coordinate = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3)
        # coordinate = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3)
        # # coordinate.transform(self.workspace2world)
        # o3d.visualization.draw_geometries([pcd, coordinate])



        # remove outliers
        outliers = None
        new_outlier = None
        rm_iter = 0
        while new_outlier is None or len(new_outlier.points) > 0:
            _, inlier_idx = pcd.remove_statistical_outlier(
                nb_neighbors = 20, std_ratio = 1.5 + rm_iter * 0.5
            )
            new_pcd = pcd.select_by_index(inlier_idx)
            new_outlier = pcd.select_by_index(inlier_idx, invert=True)
            if outliers is None:
                outliers = new_outlier
            else:
                outliers += new_outlier
            pcd = new_pcd
            rm_iter += 1



        # print(f"visualize after outlier removal")
        # coordinate = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3)
        # coordinate = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3)
        # # coordinate.transform(self.workspace2world)
        # o3d.visualization.draw_geometries([pcd, coordinate])



        # keep k% points with largest z (closest to table)
        if k_filter < 1.0:
            points = np.array(pcd.points)
            z = points[:, 2]
            z_sorted = np.sort(z)
            z_thresh = z_sorted[int(k_filter * len(z_sorted))]
            mask = z < z_thresh
            pcd = pcd.select_by_index(np.where(mask)[0])



        # print(f"visualize after k filter")
        # coordinate = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3)
        # coordinate = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3)
        # # coordinate.transform(self.workspace2world)
        # o3d.visualization.draw_geometries([pcd, coordinate])



        return pcd

if __name__ == "__main__":
    percep_model = PerceptionModule()
    calibration_result_dir = './real_exp_new/calibration_result'
    with open(f'{calibration_result_dir}/calibration_handeye_result.pkl', 'rb') as f:
        handeye_result = pickle.load(f)
    with open(f'{calibration_result_dir}/intrinsics.pkl', 'rb') as f:
        intrinsics = pickle.load(f)
    with open(f'{calibration_result_dir}/rvecs.pkl', 'rb') as f:
        rvecs = pickle.load(f)
    with open(f'{calibration_result_dir}/tvecs.pkl', 'rb') as f:
        tvecs = pickle.load(f)
    rgb_list = [cv2.imread(f'vis_real_world/{i}_rgb.png') for i in range(3)]
    percep_model.get_tabletop_points()
