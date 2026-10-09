# Expected outputs from the official MediaPipe Python package on MediaPipe's own
# test images, for the Rust golden tests. Usage: goldens.py <work dir> <golden dir>
import json, math, sys
import mediapipe as mp
from mediapipe.tasks import python as mpt
from mediapipe.tasks.python import vision as V

work, gold = sys.argv[1], sys.argv[2]
base = lambda f: mpt.BaseOptions(model_asset_path=f"{work}/{f}")
face = V.FaceLandmarker.create_from_options(V.FaceLandmarkerOptions(
    base_options=base("face_landmarker.task"), output_face_blendshapes=True,
    output_facial_transformation_matrixes=True))
pose = V.PoseLandmarker.create_from_options(V.PoseLandmarkerOptions(base_options=base("pose_landmarker_lite.task")))
hand = V.HandLandmarker.create_from_options(V.HandLandmarkerOptions(base_options=base("hand_landmarker.task"), num_hands=2))
out = {}
for name in ["portrait.jpg", "pose.jpg", "thumb_up.jpg"]:
    img = mp.Image.create_from_file(f"{gold}/{name}")
    f, p, h = face.detect(img), pose.detect(img), hand.detect(img)
    r = {"w": img.width, "h": img.height}
    if f.face_landmarks:
        R = f.facial_transformation_matrixes[0]  # 4x4, row-major math layout
        r["face"] = {
            "landmarks": [[l.x, l.y] for l in f.face_landmarks[0]],
            "blend": {c.category_name: c.score for c in f.face_blendshapes[0]},
            "yaw": math.degrees(math.asin(max(-1.0, min(1.0, -R[2][0])))),
            "pitch": math.degrees(math.atan2(R[2][1], R[2][2])),
        }
    if p.pose_landmarks:
        r["pose"] = [[l.x, l.y] for l in p.pose_landmarks[0]]
    r["hands"] = [[[l.x, l.y] for l in hl] for hl in h.hand_landmarks]
    out[name] = r
json.dump(out, open(f"{gold}/goldens.json", "w"), indent=1)
for k, v in out.items():
    print(k, "face" in v, "pose" in v, len(v["hands"]), "hands")
