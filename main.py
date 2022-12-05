from deepface import DeepFace
import numpy as np
#demo1 and demo2 is pictures of same person
#demo3 is cat - raises value error during face detection
#demo4 is multiple faces
from timeit import default_timer as timer
from deepfaceStream import reworked_stream

metrics = ["cosine", "euclidean", "euclidean_l2"]
models = [
  "VGG-Face",
  "Facenet",
  "Facenet512",
  "OpenFace",
  "DeepFace",
  "DeepID",
  "ArcFace",
  "Dlib",
  "SFace",
]

def inintial_verif(path1, path2, metric):
    return DeepFace.verify(img1_path=path1, img2_path=path2, distance_metric=metric)

def get_emb(img_path):
    return DeepFace.represent(img_path=img_path,model_name=models[1])

#'emotion', 'age', 'gender', 'race'
def analyze_literal(img):
    #race throw an error OSError: Unable to open file (truncated file: eof = 409565888, sblock->base_addr = 0, stored_eof = 537149760)
    return DeepFace.analyze(img_path=img, actions=('age','emotion','gender', 'race'),models=None,detector_backend='opencv', prog_bar=True)

def stream_api(db,video):
    # use 0 for video to access the webcam
    return DeepFace.stream(db_path=db,detector_backend='opencv', model_name='VGG-Face',enable_face_analysis=True,source=video, time_threshold=5, frame_threshold=5)

def analyze(img):
    collector = []
    if type(img) is not str  and type(img) is not list:
        return 'PLEASE USE string or list as input'
    elif type(img) is str:
        return analyze_literal(img)
    elif type(img) is list:
        for i in img:
            collector.append(analyze_literal(i))
        return collector

# face preprocessing
# 1 get region with the face
# img, region = functions.preprocess_face(img = img_path, target_size = (48, 48), grayscale = True, enforce_detection = enforce_detection, detector_backend = detector_backend, return_region = True)
# predict model
# emotion_predictions = models['emotion'].predict(img)[0,:]


if __name__ == "__main__":
    reworked_stream(frame_threshold=5, detector_backend='ssd') # insert path to analyze video source = 'face_w_short.mov'
