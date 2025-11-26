import os
import urllib.request
import cv2


def download_dnn_models():
    """Download required DNN model files for face detection"""
    print("📥 Downloading DNN face detection models...")

    # Model files URLs
    proto_url = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
    model_url = "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20180205_fp16/res10_300x300_ssd_iter_140000_fp16.caffemodel"

    # Local file paths
    proto_path = "deploy.prototxt"
    model_path = "res10_300x300_ssd_iter_140000.caffemodel"

    try:
        # Download prototxt file
        if not os.path.exists(proto_path):
            print("Downloading deploy.prototxt...")
            urllib.request.urlretrieve(proto_url, proto_path)
            print("✓ deploy.prototxt downloaded")
        else:
            print("✓ deploy.prototxt already exists")

        # Download model file
        if not os.path.exists(model_path):
            print("Downloading model file... (This may take a while)")
            urllib.request.urlretrieve(model_url, model_path)
            print("✓ Model file downloaded")
        else:
            print("✓ Model file already exists")

        # Verify the models work
        print("🔍 Verifying models...")
        net = cv2.dnn.readNetFromCaffe(proto_path, model_path)
        print("✅ DNN models loaded successfully!")

    except Exception as e:
        print(f"❌ Error downloading models: {e}")
        print("🔄 Trying alternative download method...")
        download_alternative_models()


def download_alternative_models():
    """Alternative method to get DNN models"""
    try:
        # Try using OpenCV's built-in DNN model
        print("Trying OpenCV's built-in face detector...")

        # Download OpenCV's face detector
        model_url = "https://github.com/opencv/opencv_zoo/raw/master/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
        model_path = "face_detection_yunet.onnx"

        if not os.path.exists(model_path):
            urllib.request.urlretrieve(model_url, model_path)
            print("✓ Alternative model downloaded")

    except Exception as e:
        print(f"❌ Alternative download failed: {e}")


def check_opencv_dnn():
    """Check what DNN models are available in OpenCV"""
    print("\n🔍 Checking OpenCV DNN capabilities...")
    print(f"OpenCV Version: {cv2.__version__}")

    # Check if OpenCV has face detection models
    try:
        # Try to load a pre-trained face detection model that might be included
        net = cv2.dnn.readNetFromTensorflow('opencv_face_detector_uint8.pb', 'opencv_face_detector.pbtxt')
        print("✅ OpenCV built-in face detector available")
        return True
    except:
        print("❌ OpenCV built-in face detector not available")
        return False


if __name__ == "__main__":
    download_dnn_models()
    check_opencv_dnn()