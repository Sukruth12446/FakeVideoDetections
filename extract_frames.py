# extract_frames.py
import cv2
import os

INPUT_VIDEO_PATH = "videos"            # must contain 'real' and 'fake' subfolders
OUTPUT_FRAMES_PATH = "dataset/frames"  # output folder
FPS_EXTRACTION = 5                     # frames per second to extract
IMG_SIZE = 224
DETECT_FACE = False                    # set True if you want face-cropped frames

if DETECT_FACE:
    face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def extract_frames_from_video(video_path, output_folder, fps_extract=FPS_EXTRACTION):
    os.makedirs(output_folder, exist_ok=True)
    vid = cv2.VideoCapture(video_path)
    if not vid.isOpened():
        print("Cannot open", video_path)
        return
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    video_fps = vid.get(cv2.CAP_PROP_FPS) or 25
    interval = max(int(video_fps / fps_extract), 1)
    count = 0
    saved = 0
    while True:
        ret, frame = vid.read()
        if not ret:
            break
        if count % interval == 0:
            img = frame
            if DETECT_FACE:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                faces = face_detector.detectMultiScale(gray, 1.3, 5)
                if len(faces) > 0:
                    x, y, w, h = faces[0]
                    img = img[y:y+h, x:x+w]
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            out_path = os.path.join(output_folder, f"{video_name}_frame_{saved:04d}.jpg")
            cv2.imwrite(out_path, img)
            saved += 1
        count += 1
    vid.release()
    print(f"Extracted {saved} frames from {video_path} -> {output_folder}")

def extract_all():
    for cls in ("real", "fake"):
        in_dir = os.path.join(INPUT_VIDEO_PATH, cls)
        out_dir = os.path.join(OUTPUT_FRAMES_PATH, cls)
        if not os.path.exists(in_dir):
            print(f"Folder not found: {in_dir} — skip")
            continue
        os.makedirs(out_dir, exist_ok=True)
        for fname in os.listdir(in_dir):
            if fname.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
                extract_frames_from_video(os.path.join(in_dir, fname), out_dir)

if __name__ == "__main__":
    extract_all()
