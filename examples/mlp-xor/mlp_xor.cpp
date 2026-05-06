#include <plastix/plastix.hpp>

#include <cmath>
#include <iomanip>
#include <iostream>

constexpr float LearningRate = 0.5f;
constexpr size_t NumEpochs = 5000;

constexpr size_t NumInputs = 3; // x1, x2, bias
constexpr size_t HiddenSize = 4;
constexpr size_t OutputLayerStartId = NumInputs + HiddenSize;

// Per-unit pre-activation gradient dL/dz, persisted across the backward
// pass so each level can read its successors' values. The framework's
// BackwardAcc is reset to zero after every per-level Apply, so we cannot
// reuse it as the inter-level read buffer.
struct GradPreActTag {};

struct SigmoidForwardPass {
  using Accumulator = float;

  static float Map(auto &U, size_t, size_t SrcId, auto &C, size_t ConnId,
                   auto &) {
    return plastix::GetWeight(C, ConnId) * plastix::GetActivation(U, SrcId);
  }
  static float Combine(float A, float B) { return A + B; }
  static void Apply(auto &U, size_t Id, auto &, float Accumulated) {
    plastix::GetActivation(U, Id) = 1.0f / (1.0f + std::exp(-Accumulated));
  }
};

// At each level, Map reads the destination's already-finalized dL/dz and
// multiplies by the connection weight; Combine sums these into the source
// unit's dL/da via BackwardAcc. Apply then converts dL/da -> dL/dz using
// the sigmoid derivative a*(1-a) and stores the result in GradPreAct,
// where the next level down can read it.
struct SigmoidBackwardPass {
  using Accumulator = float;

  static float Map(auto &U, size_t, size_t ToId, auto &C, size_t ConnId,
                   auto &) {
    return plastix::GetWeight(C, ConnId) *
           plastix::GetField<GradPreActTag>(U, ToId);
  }
  static float Combine(float A, float B) { return A + B; }
  static void Apply(auto &U, size_t Id, auto &, float Accumulated) {
    float A = plastix::GetActivation(U, Id);
    plastix::GetField<GradPreActTag>(U, Id) = Accumulated * A * (1.0f - A);
  }
};

// w_ij -= lr * (dL/dz_dst) * a_src.
struct GradientDescentConn {
  static void UpdateIncomingConnection(auto &U, size_t DstId, size_t SrcId,
                                       auto &C, size_t ConnId, auto &) {
    float Grad;
    if (DstId >= OutputLayerStartId) { // Output layer
      // For output layer, dL/dz is in BackwardAcc (set by MSELoss)
      Grad = plastix::GetBackwardAcc(U, DstId);
    } else { // Hidden layer
      Grad = plastix::GetField<GradPreActTag>(U, DstId); // dL/dz for hidden layer (from SigmoidBackwardPass)
    }
    float Input = plastix::GetActivation(U, SrcId);
    plastix::GetWeight(C, ConnId) -= LearningRate * Grad * Input;
  }
  static void UpdateOutgoingConnection(auto &, size_t, size_t, auto &, size_t,
                                       auto &) {}
};

struct MlpTraits : plastix::DefaultNetworkTraits<> {
  using ForwardPass = SigmoidForwardPass;
  using BackwardPass = SigmoidBackwardPass;
  using Loss = plastix::MSELoss;
  using UpdateConn = GradientDescentConn;
  using ExtraUnitFields =
      plastix::UnitFieldList<plastix::alloc::SOAField<GradPreActTag, float>>;
};

using MlpNetwork = plastix::Network<MlpTraits>;

int main() {
  std::cout << "Plastix MLP / XOR Example\n";
  std::cout << "=========================\n";

  // 3 inputs (x1, x2, bias=1.0) -> 4 sigmoid hidden -> 1 sigmoid output.
  // The constant bias input lets the hidden and output units learn a
  // per-unit bias through their incoming weights without extra plumbing.
  MlpNetwork Net(
      3, plastix::FullyConnected{4, plastix::RandomUniformWeight{1234}},
      plastix::FullyConnected{1, plastix::RandomUniformWeight{1234}});

  const float Inputs[4][3] = {
      {0.0f, 0.0f, 1.0f},
      {0.0f, 1.0f, 1.0f},
      {1.0f, 0.0f, 1.0f},
      {1.0f, 1.0f, 1.0f},
  };
  const float Targets[4] = {0.0f, 1.0f, 1.0f, 0.0f};

  std::cout << std::fixed << std::setprecision(4);

  for (size_t Epoch = 0; Epoch < NumEpochs; ++Epoch) {
    float EpochLoss = 0.0f;
    for (size_t S = 0; S < 4; ++S) {
      float Tgt[1] = {Targets[S]};
      Net.DoStep(Inputs[S], Tgt);
      float Pred = Net.GetOutput()[0];
      float Diff = Pred - Targets[S];
      EpochLoss += 0.5f * Diff * Diff;
    }
    if (Epoch % 500 == 0)
      std::cout << "Epoch " << std::setw(5) << Epoch << "  loss: " << EpochLoss
                << "\n";
  }

  std::cout << "\nFinal predictions:\n";
  bool Ok = true;
  for (size_t S = 0; S < 4; ++S) {
    Net.DoForwardPass(Inputs[S]);
    float Pred = Net.GetOutput()[0];
    int Class = Pred > 0.5f ? 1 : 0;
    int Want = static_cast<int>(Targets[S]);
    std::cout << "  [" << Inputs[S][0] << ", " << Inputs[S][1] << "] -> "
              << Pred << " (class " << Class << ", target " << Want << ")\n";
    if (Class != Want)
      Ok = false;
  }

  std::cout << "\n" << (Ok ? "PASS" : "FAIL") << "\n";
  return Ok ? 0 : 1;
}
