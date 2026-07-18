"""
Detectron2 feature extraction pipeline.

Converts:

image files

into:

region-level visual embeddings

which are consumed by multimodal mBART.


Pipeline:

Image
 |
Detectron2 backbone
 |
Region proposals
 |
ROI features
 |
Save tensors


"""



import os

from pathlib import Path

import torch

from PIL import Image



import detectron2

from detectron2.config import get_cfg

from detectron2.engine import DefaultPredictor

from detectron2 import model_zoo



from detectron2.data import (
    MetadataCatalog,
    DatasetCatalog
)



# -----------------------------
# Config
# -----------------------------


MODEL_CONFIG = (
    "COCO-Detection/"
    "faster_rcnn_R_50_FPN_3x.yaml"
)


OUTPUT_DIR = (
    "data/multimodal/features"
)



DEVICE = "cuda"



# -----------------------------
# Load Detectron2 model
# -----------------------------


def load_detector():


    cfg = get_cfg()


    cfg.merge_from_file(

        model_zoo.get_config_file(
            MODEL_CONFIG
        )

    )


    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5


    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
        MODEL_CONFIG
    )


    cfg.MODEL.DEVICE = DEVICE



    predictor = DefaultPredictor(
        cfg
    )


    return predictor



# -----------------------------
# Extract features
# -----------------------------


def extract_features(
    image_path,
    predictor
):


    image = Image.open(
        image_path
    ).convert(
        "RGB"
    )


    image = (
        torch
        .as_tensor(
            image
        )
    )


    image = image.permute(
        2,
        0,
        1
    )



    outputs = predictor(
        image
    )



    instances = (
        outputs["instances"]
        .to("cpu")
    )


    boxes = (
        instances.pred_boxes
        .tensor
    )



    # Object embeddings

    # In production:
    # extract ROI pooled features
    # from backbone


    features = boxes


    return features



# -----------------------------
# Process dataset
# -----------------------------


def process_images(
    image_dir
):


    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )


    predictor = load_detector()



    image_dir = Path(
        image_dir
    )



    for image_path in image_dir.iterdir():


        if image_path.suffix.lower() not in [
            ".jpg",
            ".png",
            ".jpeg"
        ]:

            continue



        print(
            "Processing:",
            image_path.name
        )



        features = extract_features(

            image_path,

            predictor

        )



        output_path = (

            Path(OUTPUT_DIR)

            /
            f"{image_path.stem}.pt"

        )


        torch.save(

            features,

            output_path

        )



if __name__ == "__main__":


    process_images(

        "data/multimodal/images"

    )
