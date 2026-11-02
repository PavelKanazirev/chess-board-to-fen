// board2fen C++ demo: straightened grayscale board (binary PGM) -> FEN, with the ONNX Runtime C++ API.
//
// It needs nothing but ONNX Runtime (no OpenCV, no Python), like a program on an embedded device
// whose camera pipeline already delivers a straightened grayscale frame.
//
//   board2fen_cpp <model.onnx> <board.pgm> [timing_runs]
//
// Make board.pgm with:  python scripts/predict.py photo.jpg --save-board board.pgm

#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

// The same order as CLASSES in board2fen/labels.py; '.' is an empty square.
constexpr std::array<char, 13> kClasses = {'.', 'P', 'N', 'B', 'R', 'Q', 'K', 'p', 'n', 'b', 'r', 'q', 'k'};

struct GrayImage {
  int width = 0;
  int height = 0;
  std::vector<uint8_t> pixels;  // row by row, one byte per pixel
};

// Reads a binary PGM file: "P5", width, height, 255 as text, then width * height bytes.
GrayImage ReadPgm(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error("cannot open " + path);
  std::string magic;
  int max_value = 0;
  GrayImage image;
  in >> magic >> image.width >> image.height >> max_value;
  if (magic != "P5" || max_value != 255) throw std::runtime_error(path + " is not an 8-bit binary PGM file");
  in.get();  // the single whitespace character that ends the header

  image.pixels.resize(static_cast<size_t>(image.width) * image.height);
  in.read(reinterpret_cast<char*>(image.pixels.data()), static_cast<std::streamsize>(image.pixels.size()));
  if (!in) throw std::runtime_error(path + " is shorter than its header says");
  return image;
}

// Cuts the board into 64 squares of size x size pixels in image order (a8, b8, ..., h1), as one
// float tensor of shape (64, size, size, 1) with raw 0..255 values: the model rescales them itself.
// This is split_squares from board2fen/detect.py, written as loops.
std::vector<float> SplitSquares(const GrayImage& board, int size) {
  if (board.width != 8 * size || board.height != 8 * size) {
    throw std::runtime_error("expected a board of " + std::to_string(8 * size) + " x " + std::to_string(8 * size) +
                             " pixels, got " + std::to_string(board.width) + " x " + std::to_string(board.height));
  }
  std::vector<float> squares(static_cast<size_t>(64) * size * size);
  for (int square = 0; square < 64; ++square) {
    const int row = square / 8;  // 0 = rank 8
    const int col = square % 8;  // 0 = file a
    for (int y = 0; y < size; ++y) {
      for (int x = 0; x < size; ++x) {
        const size_t pixel = static_cast<size_t>(row * size + y) * board.width + (col * size + x);
        squares[(static_cast<size_t>(square) * size + y) * size + x] = board.pixels[pixel];
      }
    }
  }
  return squares;
}

// The piece placement part of a FEN, like labels_to_fen in board2fen/labels.py.
std::string PlacementFromLabels(const std::vector<int>& labels) {
  std::string fen;
  for (int row = 0; row < 8; ++row) {
    int empty = 0;
    for (int col = 0; col < 8; ++col) {
      const char symbol = kClasses[labels[row * 8 + col]];
      if (symbol == '.') {
        ++empty;
        continue;
      }
      if (empty > 0) fen += std::to_string(empty);
      empty = 0;
      fen += symbol;
    }
    if (empty > 0) fen += std::to_string(empty);
    if (row < 7) fen += '/';
  }
  return fen;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 3) {
    std::cerr << "usage: " << argv[0] << " <model.onnx> <board.pgm> [timing_runs]\n";
    return 1;
  }
  const std::string model_path = argv[1];
  const std::string board_path = argv[2];
  const int runs = argc > 3 ? std::max(1, std::stoi(argv[3])) : 100;

  try {
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "board2fen");
    Ort::SessionOptions options;
    options.SetIntraOpNumThreads(1);  // like a small edge CPU, and comparable with benchmark.py
#ifdef _WIN32
    const std::wstring wide_path(model_path.begin(), model_path.end());  // Windows wants wide paths
    Ort::Session session(env, wide_path.c_str(), options);
#else
    Ort::Session session(env, model_path.c_str(), options);
#endif

    // Ask the model for its input/output names and its square size, as OnnxClassifier does.
    Ort::AllocatorWithDefaultOptions allocator;
    const auto input_name = session.GetInputNameAllocated(0, allocator);
    const auto output_name = session.GetOutputNameAllocated(0, allocator);
    const auto input_shape = session.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
    const int size = static_cast<int>(input_shape[1]);  // (batch, size, size, 1)

    std::vector<float> squares = SplitSquares(ReadPgm(board_path), size);
    const std::array<int64_t, 4> shape = {64, size, size, 1};
    const auto memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    // The tensor only points at our vector: no copy is made.
    Ort::Value input =
        Ort::Value::CreateTensor<float>(memory, squares.data(), squares.size(), shape.data(), shape.size());
    const char* input_names[] = {input_name.get()};
    const char* output_names[] = {output_name.get()};

    auto run = [&]() { return session.Run(Ort::RunOptions{nullptr}, input_names, &input, 1, output_names, 1); };
    std::vector<Ort::Value> outputs = run();  // the first run also warms up

    std::vector<double> times_ms;
    for (int i = 0; i < runs; ++i) {
      const auto start = std::chrono::steady_clock::now();
      run();
      times_ms.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count());
    }
    std::sort(times_ms.begin(), times_ms.end());

    // probabilities has shape (64, 13): pick the most likely class of every square.
    const float* probabilities = outputs[0].GetTensorData<float>();
    const auto num_classes = static_cast<int>(outputs[0].GetTensorTypeAndShapeInfo().GetShape()[1]);
    std::vector<int> labels(64);
    float lowest_confidence = 1.0f;
    for (int square = 0; square < 64; ++square) {
      const float* row = probabilities + static_cast<size_t>(square) * num_classes;
      const float* best = std::max_element(row, row + num_classes);
      labels[square] = static_cast<int>(best - row);
      lowest_confidence = std::min(lowest_confidence, *best);
    }

    std::cout << PlacementFromLabels(labels) << " w - - 0 1\n";
    std::cout << "median " << times_ms[times_ms.size() / 2] << " ms per board (64 squares, 1 thread, " << runs
              << " runs), lowest confidence " << lowest_confidence << "\n";
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
  }
  return 0;
}
