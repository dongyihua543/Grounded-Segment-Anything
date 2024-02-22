import os
import cv2
import time
import numpy as np
from glob import glob
import supervision as sv

import torch
import torchvision
from PIL import Image

from hijack import get_tokenlizer_hijack

from groundingdino.util.inference import Model
from segment_anything import sam_model_registry, SamPredictor

# device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# constant
BOX_THRESHOLD = 0.25
TEXT_THRESHOLD = 0.25
NMS_THRESHOLD = 0.8

# GroundingDINO config and checkpoint
GROUNDING_DINO_CONFIG_PATH = "D:/PycharmProjects/GroundingDINO/groundingdino/config/GroundingDINO_SwinB_cfg.py"
GROUNDING_DINO_CHECKPOINT_PATH = "D:/PycharmProjects/GroundingDINO/weights/groundingdino_swinb_cogcoor.pth"

# Segment-Anything checkpoint
SAM_ENCODER_VERSION = "vit_h"
SAM_CHECKPOINT_PATH = "SAM/model/sam_vit_h_4b8939.pth"

# Building GroundingDINO inference model
grounding_dino_model = Model(model_config_path=GROUNDING_DINO_CONFIG_PATH, model_checkpoint_path=GROUNDING_DINO_CHECKPOINT_PATH)

# Building SAM Model and SAM Predictor
sam = sam_model_registry[SAM_ENCODER_VERSION](checkpoint=SAM_CHECKPOINT_PATH)
sam.to(device=DEVICE)
sam_predictor = SamPredictor(sam)


# Prompting SAM with detected boxes
def segment(sam_predictor: SamPredictor, image: np.ndarray, xyxy: np.ndarray) -> np.ndarray:
    sam_predictor.set_image(image)
    result_masks = []
    for box in xyxy:
        masks, scores, logits = sam_predictor.predict(
            box=box,
            multimask_output=True
        )
        index = np.argmax(scores)
        result_masks.append(masks[index])
    return np.array(result_masks)


if __name__ == '__main__':
    # Predict classes and hyper-param for GroundingDINO
    # SOURCE_IMAGE_PATH = "./assets/demo2.jpg"
    # CLASSES = ["the running dog"]

    # SOURCE_IMAGE_PATH = "./assets/bike.jpeg"
    # CLASSES = ["bike"]

    # SOURCE_IMAGE_PATH = "./assets/toy_order_005.jpg"
    # CLASSES = ["toy"]

    dataset_path = "./assets/demo_datasets/your_dataset"  # Your dataset path
    result_path = "./assets/demo_datasets/your_dataset_result"  # The folder path that you want to save the results

    im_list = glob(dataset_path + "/*.[jJ][pP][gG]") + glob(dataset_path + "/*.[jJ][pP][eE][gG]") + \
              glob(dataset_path + "/*.[pP][nN][gG]") + glob(dataset_path + "/*.[bB][mM][pP]") + \
              glob(dataset_path + "/*.[tT][iI][fF][fF]")

    CLASSES = ["toy"]

    for i, im_path in enumerate(im_list):
        im_path = im_path.replace('\\', '/')
        print("im_path: ", im_path)
        # only RGB
        img_ = Image.open(im_path).convert("RGB")

        # load image (BGR)
        image = cv2.imread(im_path)

        # detect objects
        detections = grounding_dino_model.predict_with_classes(
            image=image,
            classes=CLASSES,
            box_threshold=BOX_THRESHOLD,
            text_threshold=TEXT_THRESHOLD
        )

        # # annotate image with detections
        # box_annotator = sv.BoxAnnotator()
        # labels = [
        #     f"{CLASSES[class_id]} {confidence:0.2f}"
        #     for _, _, confidence, class_id, _
        #     in detections]
        # annotated_frame = box_annotator.annotate(scene=image.copy(), detections=detections, labels=labels)
        #
        # # save the annotated grounding dino image
        # cv2.imwrite("demo_dataset/sam/groundingdino_annotated_image.jpg", annotated_frame)

        # NMS post process
        print(f"Before NMS: {len(detections.xyxy)} boxes")
        nms_idx = torchvision.ops.nms(
            torch.from_numpy(detections.xyxy),
            torch.from_numpy(detections.confidence),
            NMS_THRESHOLD
        ).numpy().tolist()

        detections.xyxy = detections.xyxy[nms_idx]
        detections.confidence = detections.confidence[nms_idx]
        detections.class_id = detections.class_id[nms_idx]
        print(f"After NMS: {len(detections.xyxy)} boxes")

        # convert detections to masks
        detections.mask = segment(
            sam_predictor=sam_predictor,
            image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
            xyxy=detections.xyxy
        )

        # # annotate image with detections
        # box_annotator = sv.BoxAnnotator()
        # mask_annotator = sv.MaskAnnotator()
        # labels = [
        #     f"{CLASSES[class_id]} {confidence:0.2f}"
        #     for _, _, confidence, class_id, _
        #     in detections]
        # annotated_image = mask_annotator.annotate(scene=image.copy(), detections=detections)
        # cv2.imwrite("demo_dataset/sam/grounded_sam_annotated_image-1.jpg", annotated_image)
        #
        # annotated_image = box_annotator.annotate(scene=annotated_image, detections=detections, labels=labels)
        # # save the annotated grounded-sam image
        # cv2.imwrite("demo_dataset/sam/grounded_sam_annotated_image-2.jpg", annotated_image)

        # mask
        # detections.mask shape: [b, h, w]
        mask = detections.mask[0]
        mask = mask.astype(np.uint8) * 255
        img = Image.fromarray(mask, mode='L')
        im_name = im_path.split('/')[-1].split('.')[0]
        img_path = os.path.join(result_path, im_name + ".png")
        img.save(img_path)

        # segment
        Image.composite(img_, Image.new("RGB", img.size, (255, 255, 255)), mask=img).save(os.path.join(result_path, im_name + "-seg.png"))
