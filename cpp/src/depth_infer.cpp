// depth_infer -- C++17 inference for the MonoGeoFlightLog depth head.
//
//   depth_infer --onnx M --input artifacts/sample_input.bin \
//               --compare artifacts/python_output.bin \
//               --dump cpp_output.bin --dump-png depth.png \
//               --timing-json results/cpp_timing.json --runs 50 --warmup 10
//
// or feed it a real image, in which case OpenCV does the preprocessing
// (resize, BGR->RGB, scale to [0, 1]) exactly as the Python side does:
//
//   depth_infer --onnx M --image artifacts/sample_frame.png ...
//
// With `--compare` the program prints the max/mean absolute difference against
// the Python output and exits non-zero if the max exceeds the parity threshold
// (default 1e-3). That is the check CTest runs.

#include "depth_infer.hpp"

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct Options {
  std::string onnx;
  std::string image;
  std::string input_bin;
  std::string compare_path;
  std::string dump_bin;
  std::string dump_png;
  std::string timing_json;
  int runs = 50;
  int warmup = 10;
  int threads = 4;
  double parity_threshold = 1e-3;
  bool quiet = false;
  bool print_json = false;
};

void usage() {
  std::cout <<
      "usage: depth_infer --onnx MODEL.onnx (--input IN.bin | --image FRAME.png) [options]\n"
      "\n"
      "  --onnx PATH            ONNX model exported by `monogeo export` (required)\n"
      "  --input PATH           raw float32 NCHW input (bit-exact parity path)\n"
      "  --image PATH           instead of --input: run OpenCV preprocessing on an image\n"
      "  --compare PATH         raw float32 PyTorch reference output; exits 1 above threshold\n"
      "  --dump PATH            write the raw float32 depth output\n"
      "  --dump-png PATH        write a colour-mapped PNG of the depth output\n"
      "  --timing-json PATH     write mean/p50/p95 ms and backend metadata here\n"
      "  --runs N               timed iterations (default 50)\n"
      "  --warmup N             warm-up iterations (default 10)\n"
      "  --threads N            OpenCV thread count hint (default 4)\n"
      "  --parity-threshold X   maximum allowed max-abs difference (default 1e-3)\n"
      "  --print-json           echo the timing JSON to stdout\n"
      "  --quiet                suppress progress messages\n";
}

/// Parse "--key value" style arguments.
Options parse(int argc, char** argv) {
  Options o;
  auto need = [&](int i) {
    if (i + 1 >= argc) throw std::runtime_error(std::string("missing value after ") + argv[i]);
    return std::string(argv[i + 1]);
  };
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--onnx") o.onnx = need(i++);
    else if (a == "--input") o.input_bin = need(i++);
    else if (a == "--image") o.image = need(i++);
    else if (a == "--compare") o.compare_path = need(i++);
    else if (a == "--dump") o.dump_bin = need(i++);
    else if (a == "--dump-png") o.dump_png = need(i++);
    else if (a == "--timing-json") o.timing_json = need(i++);
    else if (a == "--runs") o.runs = std::stoi(need(i++));
    else if (a == "--warmup") o.warmup = std::stoi(need(i++));
    else if (a == "--threads") o.threads = std::stoi(need(i++));
    else if (a == "--parity-threshold") o.parity_threshold = std::stod(need(i++));
    else if (a == "--print-json") o.print_json = true;
    else if (a == "--quiet") o.quiet = true;
    else if (a == "-h" || a == "--help") { usage(); std::exit(0); }
    else throw std::runtime_error("unknown argument: " + a);
  }
  if (o.onnx.empty()) throw std::runtime_error("--onnx is required");
  if (o.input_bin.empty() && o.image.empty()) {
    throw std::runtime_error("one of --input or --image is required");
  }
  if (o.runs < 1) throw std::runtime_error("--runs must be >= 1");
  return o;
}

/// Preprocess an image exactly like the Python pipeline: BGR->RGB, resize to
/// (W, H), scale to [0, 1], then lay it out as NCHW.
std::vector<float> preprocess_image(const std::string& path, int height, int width) {
  cv::Mat bgr = cv::imread(path, cv::IMREAD_COLOR);
  if (bgr.empty()) throw std::runtime_error("cv::imread failed for " + path);
  cv::Mat resized;
  cv::resize(bgr, resized, cv::Size(width, height), 0, 0, cv::INTER_AREA);
  cv::Mat rgb;
  cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);
  rgb.convertTo(rgb, CV_32FC3, 1.0 / 255.0);
  std::vector<cv::Mat> channels(3);
  cv::split(rgb, channels);
  std::vector<float> out(static_cast<std::size_t>(3) * height * width);
  const std::size_t plane = static_cast<std::size_t>(height) * width;
  for (int c = 0; c < 3; ++c) {
    std::copy(channels[c].begin<float>(), channels[c].end<float>(), out.begin() + c * plane);
  }
  return out;
}

/// Quantise a depth map into a colour-mapped BGR image for inspection.
cv::Mat colormap_depth(const std::vector<float>& depth, int height, int width) {
  cv::Mat d(height, width, CV_32F, const_cast<float*>(depth.data()));
  cv::Mat scaled;
  cv::normalize(d, scaled, 0, 255, cv::NORM_MINMAX, CV_32F);
  cv::Mat u8;
  scaled.convertTo(u8, CV_8U);
  cv::Mat color;
  cv::applyColorMap(u8, color, cv::COLORMAP_TURBO);
  return color;
}

std::string json_escape(const std::string& s) {
  std::ostringstream os;
  for (char c : s) {
    if (c == '"' || c == '\\') os << '\\' << c;
    else os << c;
  }
  return os.str();
}

std::string build_json(const Options& o, const monogeo::Backend& backend,
                       const monogeo::Shape& shape, const monogeo::TimingStats& s,
                       double load_ms, const monogeo::DiffStats* diff) {
  std::ostringstream os;
  os << std::fixed;
  os << "{\n";
  os << "  \"step\": \"cpp-depth-infer\",\n";
  os << "  \"backend\": \"" << json_escape(backend.name()) << "\",\n";
  os << "  \"threads\": " << o.threads << ",\n";
  os << "  \"runs\": " << s.runs << ",\n";
  os << "  \"warmup\": " << o.warmup << ",\n";
  os << "  \"input_shape\": [";
  for (std::size_t i = 0; i < shape.dims.size(); ++i) {
    if (i) os << ", ";
    os << shape.dims[i];
  }
  os << "],\n";
  os << std::setprecision(6);
  os << "  \"input_shape_string\": \"" << shape.to_string() << "\",\n";
  os << "  \"mean_ms\": " << s.mean << ",\n";
  os << "  \"p50_ms\": " << s.p50 << ",\n";
  os << "  \"p95_ms\": " << s.p95 << ",\n";
  os << "  \"min_ms\": " << s.min << ",\n";
  os << "  \"max_ms\": " << s.max << ",\n";
  os << "  \"model_load_ms\": " << load_ms << ",\n";
  os << "  \"output_shape\": \"" << backend.output_shape().to_string() << "\",\n";
  os << "  \"method\": \"one frame per timed call, cv::dnn forward, CPU, wall clock\",\n";
  if (diff != nullptr) {
    os << std::setprecision(10);
    os << "  \"parity_max_abs_diff\": " << diff->max_abs << ",\n";
    os << "  \"parity_mean_abs_diff\": " << diff->mean_abs << ",\n";
    os << "  \"parity_threshold\": " << o.parity_threshold << ",\n";
  }
  os << "  \"onnx_path\": \"" << json_escape(o.onnx) << "\"\n";
  os << "}\n";
  return os.str();
}

}  // namespace

int main(int argc, char** argv) {
  Options opt;
  try {
    opt = parse(argc, argv);
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << "\n";
    usage();
    return 2;
  }

  try {
    const auto t_load0 = std::chrono::steady_clock::now();
    auto backend = monogeo::make_backend(opt.onnx, opt.threads);
    const double load_ms = monogeo::seconds_since(t_load0) * 1000.0;
    if (!opt.quiet) {
      std::cout << "[cpp] backend: " << backend->name() << "\n";
      std::cout << "[cpp] model load: " << std::fixed << std::setprecision(2) << load_ms << " ms\n";
    }

    // The network shape comes from the input buffer's length; for the image
    // path we keep a 1x3xHxW layout derived from the image itself.
    monogeo::Shape shape;
    std::vector<float> input;
    if (!opt.input_bin.empty()) {
      input = monogeo::read_floats(opt.input_bin);
      monogeo::Shape probe;
      probe.dims = {1, 3, 128, 416};
      // Derive H and W from the file size: N*C*H*W with N=1, C=3.
      const std::size_t per_channel = input.size() / 3;
      int h = 128;
      int w = 416;
      if (per_channel % 416 == 0) {
        h = static_cast<int>(per_channel / 416);
      } else if (per_channel % 128 == 0) {
        w = static_cast<int>(per_channel / 128);
      }
      shape.dims = {1, 3, h, w};
      if (!opt.quiet) {
        std::cout << "[cpp] input " << opt.input_bin << " -> " << shape.to_string()
                  << " (" << input.size() << " floats)\n";
      }
    } else {
      cv::Mat bgr = cv::imread(opt.image, cv::IMREAD_COLOR);
      if (bgr.empty()) throw std::runtime_error("cv::imread failed for " + opt.image);
      shape.dims = {1, 3, bgr.rows, bgr.cols};
      input = preprocess_image(opt.image, bgr.rows, bgr.cols);
      if (!opt.quiet) {
        std::cout << "[cpp] image " << opt.image << " -> " << shape.to_string()
                  << " (OpenCV resize + BGR->RGB + /255)\n";
      }
    }

    std::vector<float> out = backend->run(input, shape);
    if (!monogeo::all_finite(out)) {
      std::cerr << "error: non-finite values in the output\n";
      return 3;
    }
    if (!opt.quiet) {
      std::cout << "[cpp] output " << backend->output_shape().to_string() << " ("
                << out.size() << " floats)\n";
    }

    // warm-up then timed iterations
    for (int i = 0; i < opt.warmup; ++i) backend->run(input, shape);
    std::vector<double> times;
    times.reserve(static_cast<std::size_t>(opt.runs));
    for (int i = 0; i < opt.runs; ++i) {
      const auto t0 = std::chrono::steady_clock::now();
      out = backend->run(input, shape);
      times.push_back(monogeo::seconds_since(t0) * 1000.0);
    }
    const monogeo::TimingStats stats = monogeo::summarise(times);
    if (!opt.quiet) {
      std::cout << std::fixed << std::setprecision(3)
                << "[cpp] " << stats.runs << " runs: mean " << stats.mean << " ms, p50 "
                << stats.p50 << " ms, p95 " << stats.p95 << " ms\n";
    }

    // parity against the Python reference
    monogeo::DiffStats diff{};
    bool have_diff = false;
    bool parity_ok = true;
    if (!opt.compare_path.empty()) {
      const std::vector<float> reference = monogeo::read_floats(opt.compare_path);
      diff = monogeo::compare(out, reference);
      have_diff = true;
      parity_ok = diff.max_abs <= opt.parity_threshold;
      std::cout << std::fixed << std::setprecision(9)
                << "[cpp] parity vs " << opt.compare_path << ": max abs diff " << diff.max_abs
                << ", mean abs diff " << diff.mean_abs << " over " << diff.n << " values";
      if (diff.n != std::min(out.size(), reference.size())) std::cout << " (truncated compare)";
      std::cout << "\n";
      if (!parity_ok) {
        std::cout << "PARITY FAILED: max abs diff " << diff.max_abs << " > threshold "
                  << opt.parity_threshold << "\n";
      } else {
        std::cout << "[cpp] parity OK (threshold " << opt.parity_threshold << ")\n";
      }
    }

    if (!opt.dump_bin.empty()) {
      monogeo::write_floats(opt.dump_bin, out);
      if (!opt.quiet) std::cout << "[cpp] wrote " << opt.dump_bin << "\n";
    }
    if (!opt.dump_png.empty() && backend->output_shape().dims.size() == 4) {
      const int h = backend->output_shape().dims[2];
      const int w = backend->output_shape().dims[3];
      if (static_cast<std::size_t>(h) * w <= out.size()) {
        cv::Mat color = colormap_depth(out, h, w);
        cv::imwrite(opt.dump_png, color);
        if (!opt.quiet) std::cout << "[cpp] wrote " << opt.dump_png << "\n";
      }
    }
    if (!opt.timing_json.empty()) {
      const std::string json = build_json(opt, *backend, shape, stats, load_ms,
                                          have_diff ? &diff : nullptr);
      monogeo::write_text(opt.timing_json, json);
      if (!opt.quiet) std::cout << "[cpp] wrote " << opt.timing_json << "\n";
      if (opt.print_json) std::cout << json;
    }

    return parity_ok ? 0 : 1;
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << "\n";
    return 4;
  }
}
