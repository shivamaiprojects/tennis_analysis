"""
Court line detector.

Loads a ResNet50 fine-tuned to regress 14 tennis court keypoints (28 outputs)
and exposes methods to predict on a frame and draw the results.

Because the court is essentially static within a rally clip, we typically
predict ONCE on the first frame and reuse the result for the whole video —
which is a significant compute saving.
"""
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
import torch
import torchvision.transforms as T
from torchvision import models


class CourtLineDetector:
    """Runs a fine-tuned ResNet50 to detect tennis court keypoints.

    Model architecture:
        - ResNet50 backbone (25.5M params, pretrained on ImageNet)
        - Final FC layer replaced with a Linear(2048, 28) regression head
        - Input: 224x224 RGB image, ImageNet-normalized
        - Output: 28 floats representing 14 (x, y) keypoints in 224-space

    Attributes:
        NUM_KEYPOINTS: Number of court keypoints (14).
        INPUT_SIZE: Model input resolution (224).
    """

    NUM_KEYPOINTS: int = 14
    INPUT_SIZE: int = 224

    def __init__(self, model_path: str | Path, device: Optional[str] = None):
        """Load the trained model into memory.

        Args:
            model_path: Path to keypoints_model.pth.
            device: 'cuda', 'cpu', or None to auto-detect based on availability.
        """
        # Auto-select GPU if available
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        print(f"[CourtLineDetector] Using device: {self.device}")

        # Step 1: rebuild the architecture (weights=None because we load our own)
        model = models.resnet50(weights=None)
        model.fc = torch.nn.Linear(model.fc.in_features, self.NUM_KEYPOINTS * 2)

        # Step 2: load the trained weights
        state_dict = torch.load(str(model_path), map_location=self.device)
        model.load_state_dict(state_dict)

        # Switch to inference mode and move to target device
        model.eval()
        model.to(self.device)
        self.model = model

        # Preprocessing pipeline — MUST match training exactly
        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize((self.INPUT_SIZE, self.INPUT_SIZE)),
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],   # ImageNet channel means
                std=[0.229, 0.224, 0.225],    # ImageNet channel stds
            ),
        ])

    @torch.no_grad()
    def predict(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Predict 14 court keypoints on a single frame.

        Args:
            frame_bgr: BGR image as a (H, W, 3) uint8 numpy array (OpenCV format).

        Returns:
            Flat numpy array of shape (28,): [x0, y0, x1, y1, ..., x13, y13]
            in ORIGINAL image pixel coordinates.
        """
        original_h, original_w = frame_bgr.shape[:2]

        # BGR (OpenCV) to RGB (what the model expects)
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        # Preprocess: numpy to PIL to resize to tensor to normalize
        input_tensor = self.transform(frame_rgb)

        # Add batch dimension: (3, 224, 224) -> (1, 3, 224, 224)
        input_tensor = input_tensor.unsqueeze(0).to(self.device)

        # Forward pass
        prediction = self.model(input_tensor)

        # Remove batch dim, move to CPU, convert to numpy
        prediction = prediction.squeeze(0).cpu().numpy()

        # Rescale from 224x224 space back to original image space
        prediction[::2] *= original_w / self.INPUT_SIZE   # x-coords
        prediction[1::2] *= original_h / self.INPUT_SIZE  # y-coords

        return prediction

    def draw_keypoints(
        self, frame: np.ndarray, keypoints: np.ndarray
    ) -> np.ndarray:
        """Draw all 14 keypoints on a copy of the frame.

        Each keypoint gets a red filled circle and its index number as a label
        (useful for debugging — you can see which keypoint is which).
        """
        out = frame.copy()
        for i in range(self.NUM_KEYPOINTS):
            x = int(keypoints[i * 2])
            y = int(keypoints[i * 2 + 1])
            cv2.circle(out, (x, y), radius=5, color=(0, 0, 255), thickness=-1)
            cv2.putText(
                out, str(i), (x + 6, y - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2,
            )
        return out

    def draw_keypoints_on_video(
        self, frames: List[np.ndarray], keypoints: np.ndarray
    ) -> List[np.ndarray]:
        """Draw the same keypoints on every frame — court is static within a rally."""
        return [self.draw_keypoints(f, keypoints) for f in frames]