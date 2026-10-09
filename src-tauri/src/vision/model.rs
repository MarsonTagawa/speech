//! One ONNX model (converted from MediaPipe's TFLite by
//! scripts/convert-vision-models.sh) run through ONNX Runtime on the CPU.
use ort::session::Session;
use ort::value::Tensor;
use std::path::Path;

fn err(e: impl std::fmt::Display) -> String {
    e.to_string()
}

pub struct Model {
    session: Session,
    input: String,
}

impl Model {
    pub fn load(dir: &Path, name: &str) -> Result<Model, String> {
        let path = dir.join(format!("{name}.onnx"));
        if !path.exists() {
            return Err(format!("Camera models missing ({name}.onnx) — run scripts/convert-vision-models.sh"));
        }
        // 2 threads each so whisper and audio aren't starved.
        let session = Session::builder().map_err(err)?.with_intra_threads(2).map_err(err)?.commit_from_file(&path).map_err(err)?;
        let input = session.inputs()[0].name().to_string();
        Ok(Model { session, input })
    }

    /// Runs one input tensor; returns each named output, flattened.
    pub fn run(&mut self, shape: &[i64], data: Vec<f32>, outputs: &[&str]) -> Result<Vec<Vec<f32>>, String> {
        let t = Tensor::from_array((shape.to_vec(), data)).map_err(err)?;
        let out = self.session.run(ort::inputs![self.input.as_str() => t]).map_err(err)?;
        outputs.iter().map(|n| out[*n].try_extract_tensor::<f32>().map(|(_, v)| v.to_vec()).map_err(err)).collect()
    }
}
