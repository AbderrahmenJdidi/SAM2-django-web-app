from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
import torch

model_cfg = "C:\\Users\\jdidi\\OneDrive\\Bureau\\stage_2eme\\application\\sam2_django_app\\segment-anything-2\\sam2\\configs\\sam2.1\\sam2.1_hiera_t.yaml"
checkpoint = "C:\\Users\\jdidi\\OneDrive\\Bureau\\stage_2eme\\application\\sam2_django_app\\segment-anything-2\\checkpoints\\sam2.1_hiera_tiny.pt"
device = torch.device("cpu")
sam2_model = build_sam2(model_cfg, checkpoint,device)

predictor = SAM2ImagePredictor(sam2_model)
print("SAM2 loaded successfully")