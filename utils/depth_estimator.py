import torch
import cv2
import numpy as np
import os

class DepthEstimator:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DepthEstimator, cls).__new__(cls)
            cls._instance._initialize_model()
        return cls._instance

    def _initialize_model(self):
  
        model_type = "MiDaS_small"
        weights_path = "weights/midas_v21_small_256.pt" 
        
        print(f"INFO - Initializing Depth Model ({model_type})...")
        
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


        self.midas = torch.hub.load("intel-isl/MiDaS", model_type, pretrained=False, trust_repo=True)

        if os.path.exists(weights_path):
            print(f"INFO - Loading local weights from: {weights_path}")
            checkpoint = torch.load(weights_path, map_location=self.device)
            self.midas.load_state_dict(checkpoint)
        else:
            print(f"WARNING - Local weights not found at {weights_path}. Attempting automatic download...")

            self.midas = torch.hub.load("intel-isl/MiDaS", model_type, pretrained=True, trust_repo=True)

        self.midas.to(self.device)
        self.midas.eval()


        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
        if model_type == "MiDaS_small":
            self.transform = midas_transforms.small_transform
        else:
            self.transform = midas_transforms.dpt_transform

    def get_depth_map(self, img_rgb: np.ndarray) -> np.ndarray:

        input_batch = self.transform(img_rgb).to(self.device)

        with torch.no_grad():
            prediction = self.midas(input_batch)


            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img_rgb.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()

        depth_map = prediction.cpu().numpy()
        return depth_map

    def get_local_gradient(self, depth_map: np.ndarray, bbox_xyxy: np.ndarray) -> np.ndarray:

        x1, y1, x2, y2 = map(int, bbox_xyxy)
        h, w = depth_map.shape
        x1, x2 = max(0, x1), min(w, x2)
        y1, y2 = max(0, y1), min(h, y2)
        
        roi = depth_map[y1:y2, x1:x2]
        if roi.size == 0:
            return np.array([0.0, 0.0])

        # חישוב נגזרות (Sobel) למציאת כיוון השינוי בעומק
        grad_x = cv2.Sobel(roi, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(roi, cv2.CV_64F, 0, 1, ksize=3)
        
        vec_x = np.mean(grad_x)
        vec_y = np.mean(grad_y)
        
        return np.array([vec_x, vec_y])