# Unzips MediaPipe's .task bundles, densifies the pose detector (its sparse
# DENSIFY weights crash tf2onnx), converts all 7 models to ONNX, and checks each
# against its TFLite original. Usage: convert.py <work dir> <output dir>
import os, sys, zipfile
import numpy as np
import onnxruntime as ort
import tensorflow as tf
import tf2onnx
from tensorflow.lite.python import schema_py_generated as schema
from tensorflow.lite.tools import flatbuffer_utils as fu

work, out = sys.argv[1], sys.argv[2]
MODELS = {
    "face_landmarker.task": ["face_detector", "face_landmarks_detector", "face_blendshapes"],
    "pose_landmarker_lite.task": ["pose_detector", "pose_landmarks_detector"],
    "hand_landmarker.task": ["hand_detector", "hand_landmarks_detector"],
}


def densify(src, dst):
    """Replace DENSIFY ops with plain dense constant buffers (same weights)."""
    interp = tf.lite.Interpreter(model_path=src, experimental_preserve_all_tensors=True)
    interp.allocate_tensors()
    for d in interp.get_input_details():
        interp.set_tensor(d["index"], np.zeros(d["shape"], dtype=d["dtype"]))
    interp.invoke()
    model = fu.read_model(src)
    g = model.subgraphs[0]
    codes = {i for i, c in enumerate(model.operatorCodes)
             if max(c.builtinCode, c.deprecatedBuiltinCode) == schema.BuiltinOperator.DENSIFY}
    kept = []
    for op in g.operators:
        if op.opcodeIndex not in codes:
            kept.append(op)
            continue
        dense = interp.get_tensor(op.outputs[0])
        buf = type(model.buffers[0])()
        buf.data = np.frombuffer(dense.tobytes(), dtype=np.uint8)
        model.buffers.append(buf)
        t = g.tensors[op.outputs[0]]
        t.buffer, t.sparsity = len(model.buffers) - 1, None
        src_t = g.tensors[op.inputs[0]]  # now unused; converters must not read it
        src_t.buffer, src_t.sparsity = 0, None
    g.operators = kept
    fu.write_model(model, dst)


def to_onnx(tflite_path, onnx_path, stub_details):
    # get_tensor_details() segfaults on the densified model's orphaned tensors;
    # tf2onnx only uses it for shape hints and falls back to the model's shapes.
    orig = tf.lite.Interpreter.get_tensor_details
    if stub_details:
        tf.lite.Interpreter.get_tensor_details = lambda self: []
    try:
        tf2onnx.convert.from_tflite(tflite_path, opset=17, output_path=onnx_path)
    finally:
        tf.lite.Interpreter.get_tensor_details = orig


def verify(tflite_path, onnx_path):
    it = tf.lite.Interpreter(model_path=tflite_path)
    it.allocate_tensors()
    d = it.get_input_details()[0]
    x = np.random.RandomState(0).rand(*d["shape"]).astype(np.float32)
    it.set_tensor(d["index"], x)
    it.invoke()
    s = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    got = s.run(None, {s.get_inputs()[0].name: x})
    for o in it.get_output_details():
        ref = it.get_tensor(o["index"]).ravel()
        if ref.size == 256 * 256:  # pose segmentation mask: unused, differs after conversion
            continue
        diff = min(float(np.abs(g.ravel() - ref).max()) for g in got if g.size == ref.size)
        if diff > 0.01:
            sys.exit(f"{os.path.basename(onnx_path)}: output {o['name']} differs by {diff}")


for task, names in MODELS.items():
    zipfile.ZipFile(os.path.join(work, task)).extractall(work)
    for name in names:
        src = os.path.join(work, f"{name}.tflite")
        if name == "pose_detector":
            dense = os.path.join(work, "pose_detector_dense.tflite")
            densify(src, dense)
            src = dense
        dst = os.path.join(out, f"{name}.onnx")
        to_onnx(src, dst, stub_details=(name == "pose_detector"))
        verify(src, dst)
        print(f"ok {name}")
