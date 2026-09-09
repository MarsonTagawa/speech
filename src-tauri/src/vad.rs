use ort::session::Session;
use std::sync::Mutex;

/// Silero VAD's recurrent state has shape [2, 1, 128].
const STATE_LEN: usize = 2 * 128;
/// The v5 graph expects each inference to be prefixed with the trailing 64
/// samples of the previous frame; the actual model input is therefore
/// CONTEXT_SAMPLES + 512 = 576 wide. Feeding a bare 512-sample frame runs
/// without error but pins the output near ~0.07 forever, so the speech gate
/// never opens — this context prefix is what makes it detect speech at all.
const CONTEXT_SAMPLES: usize = 64;

/// Wraps the Silero VAD ONNX model. One frame of exactly 512 samples
/// (32ms @ 16kHz) is fed per `process` call; the model internally sees that
/// frame prefixed with the previous frame's 64-sample context.
pub struct SileroVad {
    session: Session,
    state: Vec<f32>,
    context: Vec<f32>,
}

impl SileroVad {
    pub fn new(model_path: &str) -> ort::Result<Self> {
        let session = Session::builder()?.commit_from_file(model_path)?;
        Ok(Self {
            session,
            state: vec![0.0; STATE_LEN],
            context: vec![0.0; CONTEXT_SAMPLES],
        })
    }

    pub fn reset(&mut self) {
        self.state = vec![0.0; STATE_LEN];
        self.context = vec![0.0; CONTEXT_SAMPLES];
    }

    /// Runs one inference step on a 512-sample mono frame, returning the
    /// model's speech probability in [0, 1].
    pub fn process(&mut self, frame: &[f32]) -> ort::Result<f32> {
        // Prefix the rolling context so the model sees CONTEXT_SAMPLES + frame.
        let mut samples = Vec::with_capacity(self.context.len() + frame.len());
        samples.extend_from_slice(&self.context);
        samples.extend_from_slice(frame);

        let input = ort::value::Tensor::from_array(([1i64, samples.len() as i64], samples))?;
        let state = ort::value::Tensor::from_array(([2i64, 1i64, 128i64], self.state.clone()))?;
        let sr = ort::value::Tensor::from_array(((), vec![16000i64]))?;

        let outputs = self.session.run(ort::inputs![
            "input" => input,
            "sr" => sr,
            "state" => state,
        ])?;

        let (_, prob) = outputs["output"].try_extract_tensor::<f32>()?;
        let (_, new_state) = outputs["stateN"].try_extract_tensor::<f32>()?;
        self.state = new_state.to_vec();

        // Carry this frame's trailing samples as the next call's context.
        if frame.len() >= CONTEXT_SAMPLES {
            self.context = frame[frame.len() - CONTEXT_SAMPLES..].to_vec();
        }

        Ok(prob[0])
    }
}

pub struct VadModel(pub Mutex<SileroVad>);

#[cfg(test)]
mod tests {
    use super::*;

    fn model_path() -> String {
        format!("{}/resources/silero_vad.onnx", env!("CARGO_MANIFEST_DIR"))
    }

    #[test]
    fn processes_silence_and_tone_without_shape_errors() {
        let mut vad = SileroVad::new(&model_path()).expect("model should load");

        let silence = vec![0.0f32; 512];
        let p_silence = vad.process(&silence).expect("inference on silence");
        assert!((0.0..=1.0).contains(&p_silence));

        // A loud 440Hz tone isn't speech, but it exercises the model with
        // non-zero input/state on a second call to make sure the recurrent
        // state tensor round-trips correctly across calls.
        let tone: Vec<f32> = (0..512)
            .map(|i| (i as f32 * 440.0 * std::f32::consts::TAU / 16000.0).sin())
            .collect();
        let p_tone = vad.process(&tone).expect("inference on tone");
        assert!((0.0..=1.0).contains(&p_tone));

        vad.reset();
        let p_after_reset = vad.process(&silence).expect("inference after reset");
        assert!((0.0..=1.0).contains(&p_after_reset));
    }
}
