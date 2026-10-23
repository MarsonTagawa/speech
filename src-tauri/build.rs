fn main() {
    // C++ shim around ggml's backend registry; see whisper::init_backends.
    cc::Build::new().cpp(true).file("src/ggml_guard.cpp").compile("ggml_guard");
    println!("cargo:rerun-if-changed=src/ggml_guard.cpp");
    tauri_build::build()
}
