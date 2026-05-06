#include <plastix/plastix.hpp>

#include <iomanip>
#include <iostream>
#include <vector>
#include <cmath>
#include <limits> // For std::numeric_limits

constexpr float LearningRate = 0.01f; // Reduced learning rate for deep networks
constexpr size_t NumEpochs = 5000;   // More epochs for deeper networks
constexpr size_t NumInputs = 3;      // x, y, bias
constexpr size_t HiddenSize = 16;
constexpr size_t NumOutputs = 3;

// Helper to determine unit type based on ID
// Input IDs: 0 to NumInputs-1
// Hidden IDs: NumInputs to NumInputs + HiddenSize - 1
// Output IDs: NumInputs + HiddenSize to NumInputs + HiddenSize + NumOutputs - 1
constexpr size_t HiddenLayerStartId = NumInputs;
constexpr size_t OutputLayerStartId = NumInputs + HiddenSize;

// --- Extra Unit Fields ---
// PreActivationTag: Stores the raw accumulated value (z) before activation function.
//                   Needed for ReLU derivative in backward pass.
// GradPreActTag: Stores dL/dz (gradient with respect to pre-activation).
//                Used by the backward pass to propagate gradients and by
//                GradientDescent to update weights.
struct PreActivationTag {};
struct GradPreActTag {};

// --- Forward Pass with ReLU for hidden, Linear for output ---
struct ReLUForwardPass {
    using Accumulator = float;
    static float Map(auto &U, size_t, size_t SrcId, auto &C, size_t ConnId, auto &) {
        return plastix::GetWeight(C, ConnId) * plastix::GetActivation(U, SrcId);
    }
    static float Combine(float A, float B) { return A + B; }
    static void Apply(auto &U, size_t Id, auto &, float Accumulated) {
        // Store pre-activation for backward pass
        plastix::GetField<PreActivationTag>(U, Id) = Accumulated;

        if (Id >= OutputLayerStartId) {
            // Output layer: linear activation (softmax handled by loss)
            plastix::GetActivation(U, Id) = Accumulated;
        } else if (Id >= HiddenLayerStartId) {
            // Hidden layer: ReLU activation
            plastix::GetActivation(U, Id) = std::max(0.0f, Accumulated);
        }
        // Input layer activations are set by DoForwardPass
    }
};

// --- Backward Pass with ReLU derivative for hidden ---
// Note: This policy's Apply method is NOT called for the output layer (MaxLevel)
// in Topological propagation mode. The output layer's dL/dz is directly
// handled by the Loss function (SoftmaxCrossEntropyLoss) into BackwardAcc.
// The GradientDescent policy will then correctly pick up the gradient.
struct ReLUBackwardPass {
    using Accumulator = float;
    static float Map(auto &U, size_t, size_t ToId, auto &C, size_t ConnId, auto &) {
        // Propagate dL/dz from the downstream unit, multiplied by weight
        return plastix::GetWeight(C, ConnId) * plastix::GetField<GradPreActTag>(U, ToId);
    }
    static float Combine(float A, float B) { return A + B; }
    static void Apply(auto &U, size_t Id, auto &, float Accumulated) {
        // This Apply is only called for hidden layers (MaxLevel-1 down to 1)
        // Hidden layer: calculate dL/dz = dL/da * ReLU_derivative(pre_activation)
        float PreAct = plastix::GetField<PreActivationTag>(U, Id);
        float ReLU_deriv = (PreAct > 0.0f) ? 1.0f : 0.0f;
        plastix::GetField<GradPreActTag>(U, Id) = Accumulated * ReLU_deriv; // Accumulated is dL/da
    }
};

// --- Gradient Descent: W = W - LR * (dL/dz * input) ---
struct GradientDescent {
    // Need to know where the output layer starts to pick the correct gradient source
    static constexpr size_t OutputLayerStart = OutputLayerStartId;

    static void UpdateIncomingConnection(auto &U, size_t DstId, size_t SrcId,
                                         auto &C, size_t ConnId, auto &) {
        float Grad;
        if (DstId >= OutputLayerStart) {
            // For output layer, dL/dz is in BackwardAcc (set by SoftmaxCrossEntropyLoss)
            Grad = plastix::GetBackwardAcc(U, DstId);
        } else {
            // For hidden layers, dL/dz is in GradPreActTag (set by ReLUBackwardPass)
            Grad = plastix::GetField<GradPreActTag>(U, DstId);
        }
        float Input = plastix::GetActivation(U, SrcId);
        plastix::GetWeight(C, ConnId) -= LearningRate * Grad * Input;
    }
    static void UpdateOutgoingConnection(auto &, size_t, size_t, auto &, size_t, auto &) {}
};

// --- Network Traits ---
struct DeepMLPTraits : plastix::DefaultNetworkTraits<> {
    using ForwardPass = ReLUForwardPass;
    using BackwardPass = ReLUBackwardPass;
    using Loss = plastix::SoftmaxCrossEntropyLoss;
    using UpdateConn = GradientDescent;
    using ExtraUnitFields = plastix::UnitFieldList<
        plastix::alloc::SOAField<PreActivationTag, float>,
        plastix::alloc::SOAField<GradPreActTag, float>>;
    // Default Propagation::Topological is suitable for standard backprop
};

using DeepMLPClassifierNet = plastix::Network<DeepMLPTraits>;

int main() {
    std::cout << "Plastix Deep MLP Softmax Classifier Example\n";
    std::cout << "===========================================\n";

    // Network structure: 3 inputs -> HiddenSize ReLU hidden -> 3 outputs (linear for softmax)
    DeepMLPClassifierNet Net(NumInputs,
                             plastix::FullyConnected{HiddenSize, plastix::RandomUniformWeight{42, -0.1f, 0.1f}},
                             plastix::FullyConnected{NumOutputs, plastix::RandomUniformWeight{42, -0.1f, 0.1f}});

    // Synthetic Data: 3 classes based on simple geometric rules
    // Class 0: x > 0.5, Class 1: y > 0.5 (and x <= 0.5), Class 2: Otherwise
    struct Sample { std::vector<float> in; std::vector<float> target; };
    std::vector<Sample> TrainingData = {
        {{0.8f, 0.2f, 1.0f}, {1.0f, 0.0f, 0.0f}}, // Class 0
        {{0.9f, 0.9f, 1.0f}, {1.0f, 0.0f, 0.0f}}, // Class 0
        {{0.2f, 0.8f, 1.0f}, {0.0f, 1.0f, 0.0f}}, // Class 1
        {{0.1f, 0.7f, 1.0f}, {0.0f, 1.0f, 0.0f}}, // Class 1
        {{0.1f, 0.1f, 1.0f}, {0.0f, 0.0f, 1.0f}}, // Class 2
        {{0.4f, 0.3f, 1.0f}, {0.0f, 0.0f, 1.0f}}  // Class 2
    };

    std::cout << "Training for " << NumEpochs << " epochs...\n";

    for (size_t Epoch = 0; Epoch < NumEpochs; ++Epoch) {
        for (const auto& S : TrainingData) {
            Net.DoStep(S.in, S.target);
        }
    }

    std::cout << "\nTesting Predictions:\n";
    std::cout << std::fixed << std::setprecision(4);
    
    auto TestPoints = TrainingData; // Use training data for testing
    size_t Correct = 0;

    for (const auto& S : TestPoints) {
        Net.DoForwardPass(S.in);
        auto RawOut = Net.GetOutput();
        
        // Manual softmax for display
        float Max = -std::numeric_limits<float>::infinity();
        for(float v : RawOut) if(v > Max) Max = v;
        
        std::vector<float> Soft(RawOut.size());
        float Sum = 0;
        for(size_t i=0; i < RawOut.size(); ++i) {
            Soft[i] = std::exp(RawOut[i] - Max);
            Sum += Soft[i];
        }

        int PredClass = 0;
        float BestProb = 0;
        for(size_t i=0; i < Soft.size(); ++i) {
            Soft[i] /= Sum;
            if(Soft[i] > BestProb) { BestProb = Soft[i]; PredClass = i; }
        }

        int TargetClass = S.target[0] > 0.5 ? 0 : (S.target[1] > 0.5 ? 1 : 2);
        
        std::cout << "Input: [" << S.in[0] << ", " << S.in[1] << "] -> Probabilities: ["
                  << Soft[0] << ", " << Soft[1] << ", " << Soft[2] << "] "
                  << "Pred: " << PredClass << " Target: " << TargetClass;
        
        if (PredClass == TargetClass) {
            std::cout << " [OK]\n";
            Correct++;
        } else {
            std::cout << " [FAIL]\n";
        }
    }

    std::cout << "\nAccuracy: " << (100.0f * Correct / TestPoints.size()) << "%\n";
    
    if (Correct == TestPoints.size()) {
        std::cout << "PASS\n";
        return 0;
    }
    std::cout << "FAIL\n";
    return 1;
}