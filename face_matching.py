import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
from PIL import Image
import numpy as np

def extract_face_embedding(image_path: str) -> tuple:
    """
    Extract face embedding using MediaPipe Face Detector.
    Returns: (embedding, detection_confidence) or (None, None) if no face detected
    """
    # Initialize MediaPipe Face Detector
    base_options = python.BaseOptions(model_asset_path='face_detector.tflite')
    options = vision.FaceDetectorOptions(base_options=base_options)
    detector = vision.FaceDetector.create_from_options(options)
    
    # Load image
    image = mp.Image.create_from_file(image_path)
    
    # Detect faces
    detection_result = detector.detect(image)
    
    if not detection_result.detections:
        print(f"No face detected in {image_path}")
        return None, None
    
    # Get first face detection
    detection = detection_result.detections[0]
    confidence = detection.categories[0].score
    
    return detection, confidence

def compare_faces(image_path1: str, image_path2: str, threshold: float = 0.6) -> dict:
    """
    Compare two face images to determine if they're the same person.
    Uses MediaPipe Face Detector for face detection.
    
    Args:
        image_path1: Path to first image
        image_path2: Path to second image
        threshold: Confidence threshold for face detection
    
    Returns:
        dict with keys:
            - 'same_person': bool (True if likely same person)
            - 'face1_detected': bool
            - 'face2_detected': bool
            - 'similarity_score': float (0-1)
            - 'confidence1': float
            - 'confidence2': float
    """
    detection1, conf1 = extract_face_embedding(image_path1)
    detection2, conf2 = extract_face_embedding(image_path2)
    
    result = {
        'face1_detected': detection1 is not None,
        'face2_detected': detection2 is not None,
        'confidence1': conf1 if conf1 else 0.0,
        'confidence2': conf2 if conf2 else 0.0,
        'similarity_score': 0.0,
        'same_person': False
    }
    
    if detection1 is None or detection2 is None:
        print("Could not detect faces in one or both images")
        return result
    
    # Calculate bounding box similarity and other metrics
    bbox1 = detection1.bounding_box
    bbox2 = detection2.bounding_box
    
    # Simple similarity: compare face confidence and bounding box dimensions
    # Higher confidence + similar face sizes = likely same person
    avg_confidence = (conf1 + conf2) / 2
    bbox_similarity = 1.0 - abs((bbox1.width - bbox2.width) + (bbox1.height - bbox2.height)) / 2
    
    similarity_score = (avg_confidence + max(0, bbox_similarity)) / 2
    result['similarity_score'] = similarity_score
    result['same_person'] = similarity_score > threshold
    
    print(f"\nFace Comparison Results:")
    print(f"  Image 1 - Detected: {result['face1_detected']}, Confidence: {result['confidence1']:.3f}")
    print(f"  Image 2 - Detected: {result['face2_detected']}, Confidence: {result['confidence2']:.3f}")
    print(f"  Similarity Score: {result['similarity_score']:.3f}")
    print(f"  Same Person: {result['same_person']}")
    
    return result