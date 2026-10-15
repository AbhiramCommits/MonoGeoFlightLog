// Shared inference helpers for the MonoGeoFlightLog C++ depth path.
//
// Two backends implement the same interface:
//
//   * OpenCvDnnBackend  -- cv::dnn::readNetFromONNX, always available.
//   * LibTorchBackend   -- torch::jit::load, compiled only when LibTorch is
//                          found (CMake defines MONOGEO_HAVE_LIBTORCH).
//
// The OpenCV DNN path is the one this repository builds, runs and times; see
// results/cpp_timing.json for the backend that was actually used.

#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace monogeo {

/// Shape of a dense float tensor, row-major.
struct Shape {
  std::vector<int> dims;  ///< e.g. {1, 1, H, W}
  /// Number of elements implied by `dims`.
  std::size_t size() const;
  /// Human readable form, e.g. "1x1x128x416".
  std::string to_string() const;
};

/// Abstract depth inference backend.
class Backend {
 public:
  virtual ~Backend() = default;
  /// Run one forward pass. `input` must hold exactly the elements of `shape`.
  virtual std::vector<float> run(const std::vector<float>& input, const Shape& shape) = 0;
  /// Backend name recorded in the timing JSON.
  virtual std::string name() const = 0;
  /// Output shape of the last successful run.
  virtual Shape output_shape() const = 0;
};

/// Build the best available backend for `onnx_path`.
///
/// Uses LibTorch when the program was compiled with MONOGEO_HAVE_LIBTORCH and
/// the file is a TorchScript archive, otherwise cv::dnn. `threads` is a hint
/// forwarded to the backend.
std::unique_ptr<Backend> make_backend(const std::string& onnx_path, int threads);

// ---- I/O -----------------------------------------------------------------

/// Read a raw little-endian float32 file.
std::vector<float> read_floats(const std::string& path);
/// Write a raw little-endian float32 file.
void write_floats(const std::string& path, const std::vector<float>& data);
/// Overwrite a text file (parent directories are not created).
void write_text(const std::string& path, const std::string& text);

// ---- statistics ----------------------------------------------------------

/// Mean / p50 / p95 / min / max of a list of millisecond timings.
struct TimingStats {
  double mean = 0.0;
  double p50 = 0.0;
  double p95 = 0.0;
  double min = 0.0;
  double max = 0.0;
  std::size_t runs = 0;
};

/// Percentiles over a copy of `values` (so the input is left alone).
TimingStats summarise(std::vector<double> values);

/// Element-wise difference statistics between two buffers.
struct DiffStats {
  double max_abs = 0.0;
  double mean_abs = 0.0;
  std::size_t n = 0;
  std::size_t argmax = 0;
};

/// Compare two buffers; unequal lengths compare over the shorter prefix.
DiffStats compare(const std::vector<float>& a, const std::vector<float>& b);

// ---- misc ----------------------------------------------------------------

/// Elapsed seconds since `t0`.
double seconds_since(const std::chrono::steady_clock::time_point& t0);

/// True when every element is finite.
bool all_finite(const std::vector<float>& v);

}  // namespace monogeo
