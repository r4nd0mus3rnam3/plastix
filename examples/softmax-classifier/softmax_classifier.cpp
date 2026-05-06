#include <plastix/plastix.hpp>

#include <iomanip>
#include <iostream>
#include <vector>
#include <cmath>
#include <limits>

constexpr float LearningRate = 0.1f;
constexpr size_t NumEpochs = 1000;

// Linear Forward Pass: Output = sum(weight * input).
// We leave the activation linear because SoftmaxCrossEntropyLoss handles the 
// softmax transformation internally.
struct LinearForwardPass {
    using Accumulator = float;
    static float Map(auto &U, size_t, size_t SrcId, auto &C, size_t ConnId, auto &) {
        return plastix::GetWeight(C, ConnId) * plastix::GetActivation(U, SrcId);
    }
    static float Combine(float A, float B) { return A + B; }
    static void Apply(auto &U, size_t Id, auto &, float Acc) {
        plastix::GetActivation(U, Id) = Acc;
    }
};

// Simple Gradient Descent: W = W - LR * (dL/dz * input)
// SoftmaxCrossEntropyLoss stages (softmax - target) into BackwardAcc.
struct GradientDescent {
    static void UpdateIncomingConnection(auto &U, size_t DstId, size_t SrcId,
                                         auto &C, size_t ConnId, auto &) {
        float Grad = plastix::GetBackwardAcc(U, DstId);
        float Input = plastix::GetActivation(U, SrcId);
        plastix::GetWeight(C, ConnId) -= LearningRate * Grad * Input;
    }
    static void UpdateOutgoingConnection(auto &, size_t, size_t, auto &, size_t, auto &) {}
};

struct SoftmaxClassifierTraits : plastix::DefaultNetworkTraits<> {
    using ForwardPass = LinearForwardPass;
    using BackwardPass = plastix::NoBackwardPass; // Gradient is already in BackwardAcc from Loss
    using Loss = plastix::SoftmaxCrossEntropyLoss;
    using UpdateConn = GradientDescent;
};

using ClassifierNet = plastix::Network<SoftmaxClassifierTraits>;

int main() {
    std::cout << "Plastix Softmax Classifier Example\n";
    std::cout << "==================================\n";

    // 3 inputs (x, y, bias) -> 3 outputs (Class 0, 1, 2)
    ClassifierNet Net(3, plastix::FullyConnected{3, plastix::RandomUniformWeight{42, -0.1f, 0.1f}});

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
    
    auto TestPoints = TrainingData;
    size_t Correct = 0;

    for (const auto& S : TestPoints) {
        Net.DoForwardPass(S.in);
        auto RawOut = Net.GetOutput();
        
        // Manual softmax for display
        float Max = -std::numeric_limits<float>::infinity();
        float Sum = 0;
        for(float v : RawOut) if(v > Max) Max = v;

        std::vector<float> Soft(RawOut.size());
        for(size_t i=0; i < RawOut.size(); ++i) { Soft[i] = std::exp(RawOut[i] - Max); Sum += Soft[i]; }

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