// Implementation of the MonoGeoFlightLog C++ inference helpers.
//
// The OpenCV DNN backend builds a blob in the (N, C, H, W) layout the exported
// ONNX graph expects, forwards it, and flattens the output. Everything is CPU,
// single process, so the timings are directly comparable with the Python
// numbers in results/python_timing.json.

#include "depth_infer.hpp"

#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <numeric>
#include <sstream>
#include <stdexcept>

#ifdef MONOGEO_HAVE_LIBTORCH
#include <torch/script.h>
#endif

namespace monogeo {

// ---------------------------------------------------------------------------
// Shape
// ---------------------------------------------------------------------------
std::size_t Shape::size() const {
  std::size_t n = 1;
  for (int d : dims) n *= static_cast<std::size_t>(d);
  return n;
}

std::string Shape::to_string() const {
  std::ostringstream os;
  for (std::size_t i = 0; i < dims.size(); ++i) {
    if (i) os << "x";
    os << dims[i];
  }
  return os.str();
}

// ---------------------------------------------------------------------------
// OpenCV DNN backend
// ---------------------------------------------------------------------------
namespace {

class OpenCvDnnBackend : public Backend {
 public:
  OpenCvDnnBackend(const std::string& onnx_path, int threads) : threads_(threads) {
    net_ = cv::dnn::readNetFromONNX(onnx_path);
    if (net_.empty()) {
      throw std::runtime_error("cv::dnn could not read " + onnx_path);
    }
    net_.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    net_.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
    if (threads_ > 0) {
      cv::setNumThreads(threads_);
    }
  }

  std::vector<float> run(const std::vector<float>& input, const Shape& shape) override {
    if (shape.dims.size() != 4) {
      throw std::runtime_error("cv::dnn backend expects a 4-D (N, C, H, W) input");
    }
    if (input.size() != shape.size()) {
      throw std::runtime_error("input buffer size does not match the declared shape");
    }
    // Wrap the caller's buffer without copying, then forward.
    cv::Mat blob(4, shape.dims.data(), CV_32F, const_cast<float*>(input.data()));
    net_.setInput(blob);
    cv::Mat out = net_.forward();

    out_shape_.dims.assign(out.size.p, out.size.p + out.dims);
    // Materialise as a contiguous float buffer regardless of the blob layout.
    cv::Mat cont = out.isContinuous() ? out : out.clone();
    std::vector<float> result(cont.begin<float>(), cont.end<float>());
    return result;
  }

  std::string name() const override { return "cv::dnn (OpenCV " CV_VERSION ")"; }

  Shape output_shape() const override { return out_shape_; }

 private:
  cv::dnn::Net net_;
  Shape out_shape_;
  int threads_ = 0;
};

#ifdef MONOGEO_HAVE_LIBTORCH
class LibTorchBackend : public Backend {
 public:
  explicit LibTorchBackend(const std::string& path) {
    module_ = torch::jit::load(path);
    module_.eval();
  }

  std::vector<float> run(const std::vector<float>& input, const Shape& shape) override {
    if (shape.dims.size() != 4) {
      throw std::runtime_error("LibTorch backend expects a 4-D (N, C, H, W) input");
    }
    auto tensor = torch::from_blob(const_cast<float*>(input.data()),
                                   {shape.dims[0], shape.dims[1], shape.dims[2], shape.dims[3]},
                                   torch::kFloat32)
                      .clone();
    torch::NoGradGuard guard;
    auto out = module_.forward({tensor}).toTensor().contiguous();
    out_shape_.dims.assign(out.sizes().begin(), out.sizes().end());
    std::vector<float> result(out.data_ptr<float>(), out.data_ptr<float>() + out.numel());
    return result;
  }

  std::string name() const override { return "libtorch (TorchScript)"; }

  Shape output_shape() const override { return out_shape_; }

 private:
  torch::jit::Module module_;
  Shape out_shape_;
};
#endif  // MONOGEO_HAVE_LIBTORCH

}  // namespace

std::unique_ptr<Backend> make_backend(const std::string& onnx_path, int threads) {
#ifdef MONOGEO_HAVE_LIBTORCH
  // A TorchScript archive starts with the magic bytes of a zip file ("PK").
  std::ifstream probe(onnx_path, std::ios::binary);
  char magic[2] = {0, 0};
  probe.read(magic, 2);
  if (magic[0] == 'P' && magic[1] == 'K') {
    return std::make_unique<LibTorchBackend>(onnx_path);
  }
#endif
  return std::make_unique<OpenCvDnnBackend>(onnx_path, threads);
}

// ---------------------------------------------------------------------------
// I/O
// ---------------------------------------------------------------------------
std::vector<float> read_floats(const std::string& path) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f) throw std::runtime_error("cannot open " + path);
  const std::streamsize bytes = f.tellg();
  if (bytes <= 0 || bytes % static_cast<std::streamsize>(sizeof(float)) != 0) {
    throw std::runtime_error(path + " is not a whole number of float32 values");
  }
  f.seekg(0);
  std::vector<float> data(static_cast<std::size_t>(bytes) / sizeof(float));
  f.read(reinterpret_cast<char*>(data.data()), bytes);
  if (!f) throw std::runtime_error("short read from " + path);
  return data;
}

void write_floats(const std::string& path, const std::vector<float>& data) {
  std::ofstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("cannot write " + path);
  f.write(reinterpret_cast<const char*>(data.data()),
          static_cast<std::streamsize>(data.size() * sizeof(float)));
}

void write_text(const std::string& path, const std::string& text) {
  std::ofstream f(path);
  if (!f) throw std::runtime_error("cannot write " + path);
  f << text;
}

// ---------------------------------------------------------------------------
// statistics
// ---------------------------------------------------------------------------
TimingStats summarise(std::vector<double> values) {
  TimingStats s;
  if (values.empty()) return s;
  std::sort(values.begin(), values.end());
  const std::size_t n = values.size();
  const double sum = std::accumulate(values.begin(), values.end(), 0.0);
  auto pct = [&](double q) {
    const double pos = q * static_cast<double>(n - 1);
    const std::size_t lo = static_cast<std::size_t>(std::floor(pos));
    const std::size_t hi = std::min(lo + 1, n - 1);
    const double frac = pos - static_cast<double>(lo);
    return values[lo] * (1.0 - frac) + values[hi] * frac;
  };
  s.mean = sum / static_cast<double>(n);
  s.p50 = pct(0.50);
  s.p95 = pct(0.95);
  s.min = values.front();
  s.max = values.back();
  s.runs = n;
  return s;
}

DiffStats compare(const std::vector<float>& a, const std::vector<float>& b) {
  DiffStats d;
  const std::size_t n = std::min(a.size(), b.size());
  d.n = n;
  if (n == 0) return d;
  double sum = 0.0;
  double best = -1.0;
  for (std::size_t i = 0; i < n; ++i) {
    const double diff = std::fabs(static_cast<double>(a[i]) - static_cast<double>(b[i]));
    sum += diff;
    if (diff > best) {
      best = diff;
      d.argmax = i;
    }
  }
  d.max_abs = best;
  d.mean_abs = sum / static_cast<double>(n);
  return d;
}

double seconds_since(const std::chrono::steady_clock::time_point& t0) {
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

bool all_finite(const std::vector<float>& v) {
  for (float x : v) {
    if (!std::isfinite(x)) return false;
  }
  return true;
}

}  // namespace monogeo
