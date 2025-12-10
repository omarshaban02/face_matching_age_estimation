import os
import sys
import cv2
import torch
import numpy as np
from PIL import Image
import mediapipe as mp
from pathlib import Path

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QMessageBox, QFrame
)
from PySide6.QtGui import QPixmap, QFont, QColor, QImage
from PySide6.QtCore import Qt, QSize, QThread, Signal

from model import AgeEstimatorModel
from dataset_prepare import make_transforms
from inference import load_model_from_hf, corn_inference
from face_alignment import FaceAligner


# =====================
# Face Processing Worker
# =====================
class FaceProcessingWorker(QThread):
    finished = Signal()
    result_ready = Signal(dict)
    error = Signal(str)

    def __init__(self, image_path1: str, image_path2: str, age_model, device: str):
        super().__init__()
        self.image_path1 = image_path1
        self.image_path2 = image_path2
        self.age_model = age_model
        self.device = device
        self.image_size = 224
        self.max_age = 116
        self.face_aligner = FaceAligner()

    def run(self):
        try:
            result = self.process_images()
            self.result_ready.emit(result)
        except Exception as e:
            self.error.emit(str(e))
        finally:
            self.finished.emit()

    def process_images(self) -> dict:
        """Process and compare two face images"""
        # Initialize MediaPipe BlazeFace Detector (faster and more efficient)
        mp_face_detection = mp.solutions.face_detection
        
        with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detector:
            # Process image 1
            faces1, img1_rgb, img1_vis = self.detect_and_crop_faces(
                self.image_path1, face_detector
            )
            
            # Process image 2
            faces2, img2_rgb, img2_vis = self.detect_and_crop_faces(
                self.image_path2, face_detector
            )

        # Check if faces detected
        if not faces1:
            raise ValueError("No face detected in Image 1")
        if not faces2:
            raise ValueError("No face detected in Image 2")

        # Get main face from each image (largest face)
        face1 = faces1[0]
        face2 = faces2[0]

        # Estimate ages
        age1 = self.estimate_age(face1['crop'])
        age2 = self.estimate_age(face2['crop'])

        # Compare faces
        similarity_score = self.compare_face_embeddings(face1['crop'], face2['crop'])
        same_person = similarity_score > 0.6

        result = {
            'face1_detected': True,
            'face2_detected': True,
            'age1': int(age1),
            'age2': int(age2),
            'confidence1': face1['confidence'],
            'confidence2': face2['confidence'],
            'similarity_score': similarity_score,
            'same_person': same_person,
            'image1_vis': img1_vis,
            'image2_vis': img2_vis,
            'bbox1': face1['bbox'],
            'bbox2': face2['bbox']
        }
        
        return result

    def detect_and_crop_faces(self, image_path: str, face_detector):
        """Detect faces and return crops with visualization"""
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w, c = image_rgb.shape
        print(f"Processing image: {image_path} (size: {w}x{h})")
        results = face_detector.process(image_rgb)
        print("Faces detected:", len(results.detections) if results.detections else 0)
        faces = []
        image_vis = image_rgb.copy()
        
        if results.detections:
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box
                x = int(bbox.xmin * w)
                y = int(bbox.ymin * h)
                width = int(bbox.width * w)
                height = int(bbox.height * h)
                
                # Ensure within bounds and minimum size
                x = max(0, x)
                y = max(0, y)
                width = min(width, w - x)
                height = min(height, h - y)
                
                # Skip if crop is too small
                if width < 20 or height < 20:
                    print(f"Skipping face: crop too small ({width}x{height})")
                    continue
                
                confidence = detection.score[0] if detection.score else 0.0
                face_crop = image_rgb[y:y+height, x:x+width].copy()
                
                # Validate crop
                if face_crop is None or face_crop.size == 0:
                    print("Skipping invalid face crop")
                    continue
                
                # Draw bounding box on visualization
                cv2.rectangle(image_vis, (x, y), (x + width, y + height), (0, 255, 0), 2)
                cv2.putText(image_vis, f"Conf: {confidence:.2f}", (x, y-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                faces.append({
                    'crop': face_crop,
                    'bbox': (x, y, width, height),
                    'confidence': confidence
                })
        
        # Sort by confidence (descending) to get best face first
        faces.sort(key=lambda f: f['confidence'], reverse=True)
        
        if not faces:
            raise ValueError(f"No valid faces detected in {image_path}")
        
        return faces, image_rgb, image_vis

    def estimate_age(self, face_crop) -> float:
        """Estimate age from face crop using CORN inference"""
        try:
            # Validate face crop
            if face_crop is None or face_crop.size == 0:
                raise ValueError("Face crop is empty")
            
            if face_crop.shape[0] < 10 or face_crop.shape[1] < 10:
                raise ValueError(f"Face crop too small: {face_crop.shape}")
            
            # Convert RGB to BGR for PIL
            face_bgr = cv2.cvtColor(face_crop, cv2.COLOR_RGB2BGR)
            pil_img = Image.fromarray(face_bgr)
            
            # Use corn_inference function from inference module
            age_pred = corn_inference(self.age_model, pil_img, device=self.device)
            
            return round(age_pred, 1)
        except Exception as e:
            print(f"Error estimating age: {e}")
            return 0.0

    def compare_face_embeddings(self, face1_crop, face2_crop) -> float:
        """Compare two face crops and return similarity score"""
        try:
            # Ensure crops are valid
            if face1_crop is None or face1_crop.size == 0 or face2_crop is None or face2_crop.size == 0:
                return 0.5
            
            # Align faces for better comparison
            landmarks1 = self.face_aligner.get_facial_landmarks(face1_crop)
            landmarks2 = self.face_aligner.get_facial_landmarks(face2_crop)
            
            if landmarks1 is not None:
                face1_aligned = self.face_aligner.align_face(face1_crop, landmarks1)
            else:
                face1_aligned = face1_crop
            
            if landmarks2 is not None:
                face2_aligned = self.face_aligner.align_face(face2_crop, landmarks2)
            else:
                face2_aligned = face2_crop
            
            # Normalize crops
            crop1_norm = cv2.cvtColor(face1_aligned, cv2.COLOR_RGB2BGR)
            crop2_norm = cv2.cvtColor(face2_aligned, cv2.COLOR_RGB2BGR)
            
            # Resize to same size
            crop1_resized = cv2.resize(crop1_norm, (224, 224))
            crop2_resized = cv2.resize(crop2_norm, (224, 224))
            
            # Flatten and normalize
            crop1_flat = crop1_resized.flatten().astype(np.float32)
            crop2_flat = crop2_resized.flatten().astype(np.float32)
            
            crop1_flat /= np.linalg.norm(crop1_flat)
            crop2_flat /= np.linalg.norm(crop2_flat)
            
            # Calculate cosine similarity
            similarity = np.dot(crop1_flat, crop2_flat)
            
            # Normalize to 0-1 range
            similarity = (similarity + 1) / 2
            
            return float(similarity)
        except Exception as e:
            print(f"Error comparing faces: {e}")
            return 0.5


# =====================
# Main Application UI
# =====================
class FaceMatchingApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🔍 Face Matching & Age Estimation")
        self.setGeometry(100, 100, 1400, 900)
        self.setStyleSheet(self.get_stylesheet())
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        self.age_model = self.load_age_model()
        self.image1_path = None
        self.image2_path = None
        self.worker = None
        
        self.init_ui()

    def load_age_model(self):
        """Load age estimation model from HuggingFace"""
        try:
            print("Loading age model from HuggingFace repository...")
            model = load_model_from_hf(
                repo_id="oshaban/corn_age_estimator",
                filename="corn_model/corn_model.pt",
                backbone_name="tf_efficientnetv2_s.in21k",
                max_age=116,
                device=self.device
            )
            print("✓ Model loaded successfully from HuggingFace")
            return model
        except Exception as e:
            print(f"Error loading age model: {e}")
            print("Trying to load from local checkpoint...")
            try:
                model = AgeEstimatorModel(
                    backbone_name="tf_efficientnetv2_s.in21k",
                    pretrained=True,
                    out_dim=117
                )
                checkpoint_paths = [
                    "checkpoints/efficientnet_age_model.pth",
                    "../../../checkpoints/efficientnet_age_model.pth"
                ]
                for path in checkpoint_paths:
                    if os.path.exists(path):
                        state_dict = torch.load(path, map_location=self.device)
                        model.load_state_dict(state_dict)
                        print(f"✓ Loaded checkpoint from {path}")
                        model.to(self.device)
                        model.eval()
                        return model
            except Exception as e2:
                print(f"Error loading local checkpoint: {e2}")
            return None

    def init_ui(self):
        """Initialize UI components"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(15)
        main_layout.setContentsMargins(20, 20, 20, 20)
        
        # Title
        title = QLabel("Face Matching & Age Estimation")
        title_font = QFont()
        title_font.setPointSize(20)
        title_font.setBold(True)
        title.setFont(title_font)
        title.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(title)
        
        # Images and controls layout
        images_controls_layout = QHBoxLayout()
        
        # Left panel: Image 1
        left_panel = QVBoxLayout()
        left_panel_widget = QFrame()
        left_panel_widget.setStyleSheet("background-color: #f5f5f5; border-radius: 10px; padding: 10px;")
        
        image1_label = QLabel("Image 1")
        image1_font = QFont()
        image1_font.setPointSize(12)
        image1_font.setBold(True)
        image1_label.setFont(image1_font)
        left_panel.addWidget(image1_label)
        
        self.image1_display = QLabel()
        self.image1_display.setMinimumSize(QSize(300, 300))
        self.image1_display.setAlignment(Qt.AlignCenter)
        self.image1_display.setStyleSheet("border: 2px dashed #999; background-color: white; border-radius: 5px;")
        left_panel.addWidget(self.image1_display)
        
        btn_select1 = QPushButton("📁 Select Image 1")
        btn_select1.clicked.connect(self.select_image1)
        btn_select1.setStyleSheet(self.get_button_stylesheet())
        left_panel.addWidget(btn_select1)
        
        self.label_info1 = QLabel("No image selected")
        self.label_info1.setAlignment(Qt.AlignCenter)
        left_panel.addWidget(self.label_info1)
        
        left_panel_widget.setLayout(left_panel)
        images_controls_layout.addWidget(left_panel_widget)
        
        # Center: Comparison button and results
        center_panel = QVBoxLayout()
        center_panel.addStretch()
        
        btn_compare = QPushButton("🔄 Compare Faces")
        btn_compare.clicked.connect(self.compare_faces)
        btn_compare.setStyleSheet(self.get_button_stylesheet(primary=True))
        btn_compare.setMinimumHeight(50)
        btn_compare_font = QFont()
        btn_compare_font.setPointSize(12)
        btn_compare_font.setBold(True)
        btn_compare.setFont(btn_compare_font)
        center_panel.addWidget(btn_compare)
        
        # Results display
        self.label_results = QLabel("")
        self.label_results.setAlignment(Qt.AlignCenter)
        self.label_results.setWordWrap(True)
        results_font = QFont()
        results_font.setPointSize(11)
        self.label_results.setFont(results_font)
        center_panel.addWidget(self.label_results)
        
        center_panel.addStretch()
        images_controls_layout.addLayout(center_panel)
        
        # Right panel: Image 2
        right_panel = QVBoxLayout()
        right_panel_widget = QFrame()
        right_panel_widget.setStyleSheet("background-color: #f5f5f5; border-radius: 10px; padding: 10px;")
        
        image2_label = QLabel("Image 2")
        image2_font = QFont()
        image2_font.setPointSize(12)
        image2_font.setBold(True)
        image2_label.setFont(image2_font)
        right_panel.addWidget(image2_label)
        
        self.image2_display = QLabel()
        self.image2_display.setMinimumSize(QSize(300, 300))
        self.image2_display.setAlignment(Qt.AlignCenter)
        self.image2_display.setStyleSheet("border: 2px dashed #999; background-color: white; border-radius: 5px;")
        right_panel.addWidget(self.image2_display)
        
        btn_select2 = QPushButton("📁 Select Image 2")
        btn_select2.clicked.connect(self.select_image2)
        btn_select2.setStyleSheet(self.get_button_stylesheet())
        right_panel.addWidget(btn_select2)
        
        self.label_info2 = QLabel("No image selected")
        self.label_info2.setAlignment(Qt.AlignCenter)
        right_panel.addWidget(self.label_info2)
        
        right_panel_widget.setLayout(right_panel)
        images_controls_layout.addWidget(right_panel_widget)
        
        main_layout.addLayout(images_controls_layout)

    def select_image1(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Image 1", "", "Image Files (*.jpg *.jpeg *.png *.bmp)"
        )
        if file_path:
            self.image1_path = file_path
            self.display_image(file_path, self.image1_display)
            self.label_info1.setText(f"Selected: {Path(file_path).name}")

    def select_image2(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Image 2", "", "Image Files (*.jpg *.jpeg *.png *.bmp)"
        )
        if file_path:
            self.image2_path = file_path
            self.display_image(file_path, self.image2_display)
            self.label_info2.setText(f"Selected: {Path(file_path).name}")

    def display_image(self, image_path: str, label: QLabel):
        """Display image in QLabel"""
        pixmap = QPixmap(image_path)
        scaled_pixmap = pixmap.scaledToWidth(300, Qt.SmoothTransformation)
        label.setPixmap(scaled_pixmap)

    def compare_faces(self):
        """Start face comparison in worker thread"""
        if not self.image1_path or not self.image2_path:
            QMessageBox.warning(self, "Error", "Please select both images first!")
            return
        
        if self.age_model is None:
            QMessageBox.warning(self, "Error", "Age model not loaded!")
            return
        
        # Disable button and show loading message
        self.label_results.setText("⏳ Processing... This may take a moment.")
        
        # Create worker thread
        self.worker = FaceProcessingWorker(
            self.image1_path,
            self.image2_path,
            self.age_model,
            self.device
        )
        self.worker.result_ready.connect(self.on_comparison_complete)
        self.worker.error.connect(self.on_comparison_error)
        self.worker.start()

    def on_comparison_complete(self, result: dict):
        """Handle comparison results"""
        age1 = result['age1']
        age2 = result['age2']
        similarity = result['similarity_score']
        same_person = result['same_person']
        conf1 = result['confidence1']
        conf2 = result['confidence2']
        
        # Determine result message and color
        if same_person:
            match_text = "✅ SAME PERSON DETECTED!"
            match_color = "#00aa00"
        else:
            match_text = "❌ DIFFERENT PERSONS"
            match_color = "#cc0000"
        
        # Format results text
        results_html = f"""
        <div style="background-color: {match_color}; color: white; padding: 15px; border-radius: 8px; margin-bottom: 10px;">
            <b style="font-size: 16px;">{match_text}</b>
        </div>
        <table style="border-collapse: collapse; width: 100%;">
            <tr style="background-color: #e8e8e8;">
                <td style="padding: 10px; border: 1px solid #999;"><b>Metric</b></td>
                <td style="padding: 10px; border: 1px solid #999;"><b>Image 1</b></td>
                <td style="padding: 10px; border: 1px solid #999;"><b>Image 2</b></td>
            </tr>
            <tr>
                <td style="padding: 10px; border: 1px solid #999;"><b>Predicted Age</b></td>
                <td style="padding: 10px; border: 1px solid #999;">{int(age1)} years</td>
                <td style="padding: 10px; border: 1px solid #999;">{int(age2)} years</td>
            </tr>
            <tr style="background-color: #f5f5f5;">
                <td style="padding: 10px; border: 1px solid #999;"><b>Face Confidence</b></td>
                <td style="padding: 10px; border: 1px solid #999;">{conf1:.1%}</td>
                <td style="padding: 10px; border: 1px solid #999;">{conf2:.1%}</td>
            </tr>
            <tr>
                <td style="padding: 10px; border: 1px solid #999;"><b>Similarity Score</b></td>
                <td colspan="2" style="padding: 10px; border: 1px solid #999; text-align: center;"><b>{similarity:.1%}</b></td>
            </tr>
        </table>
        """
        
        self.label_results.setText(results_html)

    def on_comparison_error(self, error_msg: str):
        """Handle comparison error"""
        QMessageBox.critical(self, "Error", f"Comparison failed:\n{error_msg}")
        self.label_results.setText("")

    def get_stylesheet(self) -> str:
        return """
        QMainWindow {
            background-color: #ffffff;
        }
        QLabel {
            color: #333333;
        }
        """

    def get_button_stylesheet(self, primary: bool = False) -> str:
        if primary:
            return """
            QPushButton {
                background-color: #0066cc;
                color: white;
                border: none;
                border-radius: 6px;
                padding: 10px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #0052a3;
            }
            QPushButton:pressed {
                background-color: #003d7a;
            }
            """
        else:
            return """
            QPushButton {
                background-color: #f0f0f0;
                color: #333333;
                border: 1px solid #999999;
                border-radius: 6px;
                padding: 10px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #e0e0e0;
            }
            QPushButton:pressed {
                background-color: #d0d0d0;
            }
            """


def main():
    app = QApplication(sys.argv)
    window = FaceMatchingApp()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
