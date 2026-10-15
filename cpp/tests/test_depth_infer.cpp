// CTest parity test for the C++ depth inference path.
//
//   test_depth_infer <model.onnx> <sample_input.bin> <python_output.bin>
//                    <cpp_output.bin> <cpp_depth.png>
//
// Asserts, in this order:
//   1. the model loads and runs on a bit-identical input;
//   2. the output shape is 1x1xHxW (one depth map, same H/W as the input);
//   3. every output value is finite and positive (the depth head is bounded by
//      min_depth/max_depth, so a zero or negative value means the graph is wrong);
//   4. the max absolute difference against the PyTorch reference is below 1e-3,
//      which is what makes the C++ path a drop-in replacement.
//
// A failure prints "PARITY FAILED" so CMake's FAIL_REGULAR_EXPRESSION also trips.

#include "depth_infer.hpp"

#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>

namespace {

int failures = 0;

void check(bool ok, const std::string& what) {
  std::cout << (ok ? "[ ok ] " : "[FAIL] ") << what << "\n";
  if (!ok) ++failures;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 5) {
    std::cerr << "usage: test_depth_infer MODEL.onnx INPUT.bin PYTHON_OUT.bin CPP_OUT.bin "
                 "[CPP_DEPTH.png]\n";
    return 2;
  }
  const std::string model = argv[1];
  const std::string input_path = argv[2];
  const std::string python_out = argv[3];
  const std::string cpp_out = argv[4];
  const std::string png_out = argc > 5 ? argv[5] : "";

  try {
    const std::vector<float> input = monogeo::read_floats(input_path);
    const std::vector<float> reference = monogeo::read_floats(python_out);
    check(!input.empty(), "input buffer is non-empty");
    check(!reference.empty(), "python reference buffer is non-empty");

    auto backend = monogeo::make_backend(model, 4);
    std::cout << "[info] backend: " << backend->name() << "\n";

    monogeo::Shape shape;
    const std::size_t per_channel = input.size() / 3;
    int h = 128;
    int w = 416;
    if (per_channel % 416 == 0) {
      h = static_cast<int>(per_channel / 416);
    } else if (per_channel % 128 == 0) {
      w = static_cast<int>(per_channel / 128);
    }
    shape.dims = {1, 3, h, w};

    const std::vector<float> out = backend->run(input, shape);
    const monogeo::Shape oshape = backend->output_shape();
    std::cout << "[info] input " << shape.to_string() << " -> output " << oshape.to_string()
              << "\n";

    check(oshape.dims.size() == 4, "output is rank 4");
    check(oshape.dims[0] == 1 && oshape.dims[1] == 1, "output has a single depth channel");
    check(oshape.dims[2] == h && oshape.dims[3] == w, "output spatial size matches the input");
    check(oshape.size() == out.size(), "output buffer length matches its declared shape");
    check(out.size() == reference.size(), "output length matches the PyTorch reference");
    check(monogeo::all_finite(out), "all output values are finite");

    // The exported head returns metres, always positive and bounded.
    double lo = std::numeric_limits<double>::infinity();
    double hi = 0.0;
    for (float v : out) {
      lo = std::min(lo, static_cast<double>(v));
      hi = std::max(hi, static_cast<double>(v));
    }
    std::cout << std::fixed << std::setprecision(4) << "[info] depth range " << lo << " .. " << hi
              << " m\n";
    check(lo > 0.0, "all depths are positive");
    check(hi < 1e4, "all depths are within a plausible metric range");

    const monogeo::DiffStats diff = monogeo::compare(out, reference);
    std::cout << std::fixed << std::setprecision(9) << "[info] max abs diff " << diff.max_abs
              << ", mean abs diff " << diff.mean_abs << ", n " << diff.n << "\n";
    const double threshold = 1e-3;
    if (diff.max_abs <= threshold) {
      check(true, "parity with PyTorch within " + std::to_string(threshold));
    } else {
      std::cout << "PARITY FAILED: max abs diff " << diff.max_abs << " exceeds " << threshold
                << " (at flat index " << diff.argmax << ")\n";
      ++failures;
    }

    if (!cpp_out.empty()) monogeo::write_floats(cpp_out, out);

    if (failures == 0) {
      std::cout << "all checks passed\n";
    } else {
      std::cout << failures << " check(s) failed\n";
    }
    return failures == 0 ? 0 : 1;
  } catch (const std::exception& e) {
    std::cerr << "PARITY FAILED: " << e.what() << "\n";
    return 3;
  }
}
