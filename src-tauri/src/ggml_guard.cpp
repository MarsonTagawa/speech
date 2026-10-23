// Takes ggml's first backend-registry lookup inside a try/catch; see
// whisper::init_backends for why.
#include <cstddef>

extern "C" size_t ggml_backend_dev_count(void);

extern "C" bool speech_init_ggml_backends(void) {
    try {
        ggml_backend_dev_count();
        return true;
    } catch (...) {
        return false;
    }
}
