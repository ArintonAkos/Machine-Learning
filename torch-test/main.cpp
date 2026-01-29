#include <c10/core/TensorOptions.h>
#include <iostream>
#include <torch/torch.h>

void train(torch::Tensor x, torch::Tensor y, torch::Tensor w, torch::Tensor b);

int main() {
  // Ellenőrizzük, hogy elérhető-e a GPU (MPS)
  if (torch::mps::is_available()) {
    std::cout
        << "MPS (Metal Performance Shaders) is available! Running on M4 GPU."
        << std::endl;
  } else {
    std::cout << "MPS not available. Running on CPU." << std::endl;
  }

  // Létrehozunk egy 3x3-as tenzort (Matrix)
  torch::Tensor tensor = torch::rand({3, 3});
  torch::Device device = torch::kMPS;
  // Ha van GPU, átmozgatjuk oda
  if (torch::mps::is_available()) {
    tensor = tensor.to(device);
    device = torch::kMPS;
  }

  std::cout << "Random Tensor:\n" << tensor << std::endl;

  // Linearis Regresszios modell
  // X -> A bemeneti adat. 10 paraméteres vektor
  // What is int64 vs int64_t ? A: int64_t is a C++ type, int64 is a PyTorch
  // type
  const int64_t n_samples = 10e7;
  const float true_w = 2.0;
  const float true_b = 1.0;

  std::cout << "Generating " << n_samples << " samples..." << std::endl;

  torch::Tensor x = torch::randn({n_samples, 1}, device);
  // Y = W * X + b + noise
  torch::Tensor y =
      true_w * x + true_b + 0.1 * torch::randn({n_samples, 1}, device);

  torch::Tensor w = torch::randn({1, 1}, device);
  w.set_requires_grad(true);

  torch::Tensor b = torch::randn({1, 1}, device);
  b.set_requires_grad(true);

  std::cout << "Training..." << std::endl;
  auto start_time = std::chrono::high_resolution_clock::now();
  train(x, y, w, b);

  if (device.type() == torch::kMPS) {
    torch::mps::synchronize();
  }

  auto end_time = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed = end_time - start_time;
  std::cout << "Training completed in " << elapsed.count() << " seconds."
            << std::endl;

  std::cout << "True w: " << true_w << ", True b: " << true_b << std::endl;
  std::cout << "Learned w: " << w.item() << ", Learned b: " << b.item()
            << std::endl;
  std::cout << "Loss: " << torch::mse_loss(w * x + b, y).item() << std::endl;

  return 0;
}

void train(torch::Tensor x, torch::Tensor y, torch::Tensor w, torch::Tensor b) {

  const int64_t batch_size = 65536;
  const int64_t n_samples = x.size(0);

  const int patience = 5;
  const float min_delta = 0.0001;

  float best_loss = std::numeric_limits<float>::max();
  int epochs_without_improvement = 0;

  torch::optim::SGD optimizer({w, b}, 0.01);
  const int64_t n_epochs = 500;

  for (int64_t epoch = 0; epoch < n_epochs; epoch++) {

    float epoch_loss = 0.0f;
    int batches = 0;

    for (int64_t i = 0; i < n_samples; i += batch_size) {
      int64_t end = std::min(i + batch_size, n_samples);
      torch::Tensor x_batch = x.slice(0, i, end);
      torch::Tensor y_batch = y.slice(0, i, end);

      optimizer.zero_grad();

      torch::Tensor y_pred = w * x_batch + b;
      torch::Tensor loss = torch::mse_loss(y_pred, y_batch);

      loss.backward();
      optimizer.step();

      epoch_loss += loss.item().to<float>();
      batches++;
    }

    float avg_loss = epoch_loss / batches;
    std::cout << "Epoch " << epoch << ", Avg Loss: " << avg_loss << std::endl;

    // Early stopping
    if (avg_loss < best_loss - min_delta) {
      // If found better loss, update best loss and reset counter
      best_loss = avg_loss;
      epochs_without_improvement = 0;
    } else {
      // If not found better loss, increment counter
      epochs_without_improvement++;
      if (epochs_without_improvement >= patience) {
        std::cout << "Early stopping at epoch " << epoch << std::endl;
        break;
      }
    }
  }
}
