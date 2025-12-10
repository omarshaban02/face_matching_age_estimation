"""
Face Alignment Module
Provides facial landmark detection and face alignment for improved face matching
"""

import cv2
import numpy as np
import mediapipe as mp
from typing import Tuple, Optional


class FaceAligner:
    """Aligns faces using MediaPipe facial landmarks"""
    
    def __init__(self):
        """Initialize MediaPipe Face Mesh for landmark detection"""
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            min_detection_confidence=0.5
        )
        
        # Key landmarks indices for alignment
        self.LEFT_EYE = [33, 133]
        self.RIGHT_EYE = [362, 263]
        self.NOSE = [1]
        
    def get_facial_landmarks(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Detect facial landmarks in an image
        
        Args:
            image: Input image (RGB format)
            
        Returns:
            Landmarks array (468, 2) or None if no face detected
        """
        try:
            results = self.face_mesh.process(image)
            
            if not results.multi_face_landmarks:
                return None
            
            # Get first face landmarks
            landmarks = results.multi_face_landmarks[0]
            h, w = image.shape[:2]
            
            # Convert normalized coordinates to pixel coordinates
            landmark_array = np.array([
                [lm.x * w, lm.y * h] for lm in landmarks.landmark
            ])
            
            return landmark_array
        except Exception as e:
            print(f"Error detecting landmarks: {e}")
            return None
    
    def align_face(self, image: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
        """
        Align face based on eye positions
        
        Args:
            image: Input image
            landmarks: Facial landmarks (468, 2)
            
        Returns:
            Aligned image
        """
        try:
            h, w = image.shape[:2]
            
            # Get eye positions
            left_eye = landmarks[self.LEFT_EYE[0]]
            right_eye = landmarks[self.RIGHT_EYE[0]]
            
            # Calculate angle between eyes
            dY = right_eye[1] - left_eye[1]
            dX = right_eye[0] - left_eye[0]
            angle = np.degrees(np.arctan2(dY, dX))
            
            # Calculate center point for rotation
            center_x = (left_eye[0] + right_eye[0]) // 2
            center_y = (left_eye[1] + right_eye[1]) // 2
            center = (center_x, center_y)
            
            # Get rotation matrix
            rot_matrix = cv2.getRotationMatrix2D(center, angle, scale=1.0)
            
            # Apply rotation
            aligned = cv2.warpAffine(
                image,
                rot_matrix,
                (w, h),
                flags=cv2.INTER_CUBIC,
                borderMode=cv2.BORDER_REPLICATE
            )
            
            return aligned
        except Exception as e:
            print(f"Error aligning face: {e}")
            return image
    
    def align_and_crop(
        self, 
        image: np.ndarray, 
        bbox: Tuple[int, int, int, int],
        crop_expansion: float = 0.1
    ) -> Optional[np.ndarray]:
        """
        Align face and crop with consistent dimensions
        
        Args:
            image: Input image (RGB)
            bbox: Bounding box (x, y, width, height)
            crop_expansion: Expansion factor (0.1 = 10% expansion)
            
        Returns:
            Aligned and cropped face or None if alignment fails
        """
        try:
            x, y, width, height = bbox
            
            # Detect landmarks
            landmarks = self.get_facial_landmarks(image)
            if landmarks is None:
                # Fallback: return original crop without alignment
                return image[y:y+height, x:x+width].copy()
            
            # Align face
            aligned = self.align_face(image, landmarks)
            
            # Expand crop region
            expand_w = int(width * crop_expansion)
            expand_h = int(height * crop_expansion)
            
            x_expanded = max(0, x - expand_w)
            y_expanded = max(0, y - expand_h)
            w_expanded = min(aligned.shape[1] - x_expanded, width + 2 * expand_w)
            h_expanded = min(aligned.shape[0] - y_expanded, height + 2 * expand_h)
            
            # Crop aligned face
            aligned_crop = aligned[
                y_expanded:y_expanded+h_expanded,
                x_expanded:x_expanded+w_expanded
            ].copy()
            
            return aligned_crop
        except Exception as e:
            print(f"Error in align_and_crop: {e}")
            x, y, width, height = bbox
            return image[y:y+height, x:x+width].copy()
    
    def __del__(self):
        """Cleanup MediaPipe resources"""
        if hasattr(self, 'face_mesh'):
            self.face_mesh.close()
