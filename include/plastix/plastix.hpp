#ifndef PLASTIX_PLASTIX_HPP
#define PLASTIX_PLASTIX_HPP

#include "plastix/conn.hpp"
#include "plastix/layers.hpp"
#include "plastix/traits.hpp"
#include "plastix/unit_state.hpp"
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <span>
#include <type_traits>

namespace plastix {

/// Returns the library version as a string.
const char *version();

// ---------------------------------------------------------------------------
// Network
// ---------------------------------------------------------------------------

static constexpr uint16_t MaxLevels = 1024;

struct LevelRange {
  uint32_t Begin;
  uint32_t End;
};

struct InDegreeTag {};
struct OutOffsetTag {};
struct KahnWritePosTag {};
struct FrontierTag {};
struct NextFrontierTag {};
struct KahnScratchEntity {};

using KahnScratchAllocator =
    alloc::SOAAllocator<KahnScratchEntity,
                        alloc::SOAField<InDegreeTag, uint32_t>,
                        alloc::SOAField<OutOffsetTag, uint32_t>,
                        alloc::SOAField<KahnWritePosTag, uint32_t>,
                        alloc::SOAField<FrontierTag, uint32_t>,
                        alloc::SOAField<NextFrontierTag, uint32_t>>;

struct CompactEdge {
  uint64_t Bits;

  CompactEdge() : Bits(0) {}
  CompactEdge(uint32_t From, uint32_t To)
      : Bits(static_cast<uint64_t>(From) | (static_cast<uint64_t>(To) << 32)) {}

  uint32_t From() const { return static_cast<uint32_t>(Bits); }
  uint32_t To() const { return static_cast<uint32_t>(Bits >> 32); }
};

struct ProposalTag {};
struct ProposalEntity {};

using ProposalScratchAllocator =
    alloc::SOAAllocator<ProposalEntity,
                        alloc::SOAField<ProposalTag, CompactEdge>>;

template <NetworkTraits Traits> class Network {
  using UnitAllocator = UnitAllocFor<Traits>;
  using ConnAllocator = ConnAllocFor<Traits>;
  using GlobalState = typename Traits::GlobalState;

public:
  template <typename... Builders>
    requires(sizeof...(Builders) > 0 &&
             (LayerBuilder<Builders, UnitAllocator, ConnAllocator> && ...))
  Network(size_t InputDim, Builders... Layers)
      : Network(InputDim, NoUnitInit{}, Layers...) {}

  template <typename InputInit, typename... Builders>
    requires(std::invocable<InputInit, UnitAllocator &, size_t> &&
             sizeof...(Builders) > 0 &&
             (LayerBuilder<Builders, UnitAllocator, ConnAllocator> && ...))
  Network(size_t InputDim, InputInit Init, Builders... Layers)
      : NumInput(InputDim), UnitAlloc(4096), ConnAlloc(4096 * 4),
        KahnAlloc(UnitAlloc.GetCapacity() + 1),
        ProposalAlloc(ConnAlloc.GetCapacity()) {
    for (size_t I = 0; I < InputDim; ++I) {
      auto Id = UnitAlloc.Allocate();
      Init(UnitAlloc, Id);
    }
    UnitRange Prev{0, InputDim};
    ((Prev = Layers(UnitAlloc, ConnAlloc, Prev)), ...);
    OutputRange = Prev;
    if constexpr (Traits::Model == Propagation::Topological) {
      SortConnectionsByLevel();
    } else {
      CompactConnections();
    }
  }

  Network(size_t InputDim, size_t OutputDim = 1)
      : Network(InputDim, FullyConnected<>{OutputDim}) {}

  size_t GetStep() const { return Step; }

  void DoForwardPass(std::span<const float> Inputs) {
    using FP = typename Traits::ForwardPass;
    using Acc = typename FP::Accumulator;

    for (size_t I = 0; I < NumInput; ++I)
      GetActivation(UnitAlloc, I) = Inputs[I];

    if (NeedsResort) {
      if constexpr (Traits::Model == Propagation::Topological)
        SortConnectionsByLevel();
      else
        CompactConnections();
    }

    if constexpr (Traits::Model == Propagation::Topological) {
      for (uint16_t L = 1; L <= NumLevels; ++L) {
        for (uint32_t C = Ranges[L - 1].Begin; C < Ranges[L - 1].End; ++C) {
          auto ToId = GetField<ToIdTag>(ConnAlloc, C);
          auto FromId = GetField<FromIdTag>(ConnAlloc, C);
          auto &UAcc = GetForwardAcc(UnitAlloc, ToId);
          UAcc = FP::Combine(
              UAcc, FP::Map(UnitAlloc, ToId, FromId, ConnAlloc, C, Globals));
        }

        for (uint32_t Ui = UnitRanges[L].Begin; Ui < UnitRanges[L].End; ++Ui) {
          uint32_t I = SortedUnits[Ui];
          auto &UAcc = GetForwardAcc(UnitAlloc, I);
          FP::Apply(UnitAlloc, I, Globals, UAcc);
          UAcc = Acc{};
        }
      }
    } else {
      for (size_t C = 0; C < ConnAlloc.Size(); ++C) {
        auto ToId = GetField<ToIdTag>(ConnAlloc, C);
        auto FromId = GetField<FromIdTag>(ConnAlloc, C);
        auto &UAcc = GetForwardAcc(UnitAlloc, ToId);
        UAcc = FP::Combine(
            UAcc, FP::Map(UnitAlloc, ToId, FromId, ConnAlloc, C, Globals));
      }

      size_t NumUnits = UnitAlloc.Size();
      for (size_t I = NumInput; I < NumUnits; ++I) {
        auto &UAcc = GetForwardAcc(UnitAlloc, I);
        FP::Apply(UnitAlloc, I, Globals, UAcc);
        UAcc = Acc{};
      }
    }

    ++Step;
  }

  void DoCalculateLoss(std::span<const float> Targets) {
    if constexpr (std::is_same_v<typename Traits::Loss, NoLoss>)
      return;
    else {
      if (Targets.empty())
        return;
      Traits::Loss::CalculateLoss(UnitAlloc, OutputRange, Targets, Globals);
    }
  }

  void DoBackwardPass() {
    if constexpr (std::is_same_v<typename Traits::BackwardPass, NoBackwardPass>)
      return;
    else {
      using BP = typename Traits::BackwardPass;
      using Acc = typename BP::Accumulator;

      if constexpr (Traits::Model == Propagation::Topological) {
        for (uint16_t L = NumLevels; L >= 1; --L) {
          for (uint32_t C = Ranges[L].Begin; C < Ranges[L].End; ++C) {
            auto ToId = GetField<ToIdTag>(ConnAlloc, C);
            auto FromId = GetField<FromIdTag>(ConnAlloc, C);
            auto &UAcc = GetBackwardAcc(UnitAlloc, FromId);
            UAcc = BP::Combine(
                UAcc, BP::Map(UnitAlloc, FromId, ToId, ConnAlloc, C, Globals));
          }

          for (uint32_t Ui = UnitRanges[L].Begin; Ui < UnitRanges[L].End; ++Ui) {
            uint32_t I = SortedUnits[Ui];
            auto &UAcc = GetBackwardAcc(UnitAlloc, I);
            BP::Apply(UnitAlloc, I, Globals, UAcc);
            UAcc = Acc{};
          }
        }
      } else {
        for (size_t C = ConnAlloc.Size(); C-- > 0;) {
          auto ToId = GetField<ToIdTag>(ConnAlloc, C);
          auto FromId = GetField<FromIdTag>(ConnAlloc, C);
          auto &UAcc = GetBackwardAcc(UnitAlloc, FromId);
          UAcc = BP::Combine(
              UAcc, BP::Map(UnitAlloc, FromId, ToId, ConnAlloc, C, Globals));
        }

        size_t NumUnits = UnitAlloc.Size();
        for (size_t I = 0; I < NumUnits; ++I) {
          auto &UAcc = GetBackwardAcc(UnitAlloc, I);
          BP::Apply(UnitAlloc, I, Globals, UAcc);
          UAcc = Acc{};
        }
      }
    }
  }

  void DoResetGlobalState() {
    if constexpr (std::is_same_v<typename Traits::ResetGlobal,
                                 NoResetGlobalState>)
      return;
    else
      Traits::ResetGlobal::Reset(Globals);
  }

  void DoUpdateUnitState() {
    if constexpr (std::is_same_v<typename Traits::UpdateUnit, NoUpdateUnit>)
      return;
    else {
      using UP = typename Traits::UpdateUnit;
      size_t NumUnits = UnitAlloc.Size();
      for (size_t I = 0; I < NumUnits; ++I)
        UP::Update(UnitAlloc, I, Globals);
    }
  }

  void DoUpdateConnectionState() {
    if constexpr (std::is_same_v<typename Traits::UpdateConn, NoUpdateConn>)
      return;
    else {
      using UP = typename Traits::UpdateConn;

      for (size_t C = 0; C < ConnAlloc.Size(); ++C) {
        auto ToId = GetField<ToIdTag>(ConnAlloc, C);
        auto FromId = GetField<FromIdTag>(ConnAlloc, C);
        UP::UpdateIncomingConnection(UnitAlloc, ToId, FromId, ConnAlloc, C,
                                     Globals);
      }

      for (size_t C = 0; C < ConnAlloc.Size(); ++C) {
        auto ToId = GetField<ToIdTag>(ConnAlloc, C);
        auto FromId = GetField<FromIdTag>(ConnAlloc, C);
        UP::UpdateOutgoingConnection(UnitAlloc, FromId, ToId, ConnAlloc, C,
                                     Globals);
      }
    }
  }
  void DoPruneUnits() {
    if constexpr (std::is_same_v<typename Traits::PruneUnit, NoPruneUnit>)
      return;
    else {
      using PP = typename Traits::PruneUnit;
      size_t NumUnits = UnitAlloc.Size();
      for (size_t I = 0; I < NumUnits; ++I)
        GetField<PrunedTag>(UnitAlloc, I) =
            PP::ShouldPrune(UnitAlloc, I, Globals);
    }
  }

  void DoPruneConnections() {
    if constexpr (std::is_same_v<typename Traits::PruneUnit, NoPruneUnit> &&
                  std::is_same_v<typename Traits::PruneConn, NoPruneConn>)
      return;
    else {
      using CP = typename Traits::PruneConn;
      constexpr bool HasUnitPrune =
          !std::is_same_v<typename Traits::PruneUnit, NoPruneUnit>;
      constexpr bool HasConnPrune =
          !std::is_same_v<typename Traits::PruneConn, NoPruneConn>;

      for (size_t C = 0; C < ConnAlloc.Size(); ++C) {
        if (GetField<DeadTag>(ConnAlloc, C))
          continue;
        auto ToId = GetField<ToIdTag>(ConnAlloc, C);
        auto FromId = GetField<FromIdTag>(ConnAlloc, C);

        bool Remove = false;
        if constexpr (HasUnitPrune)
          Remove = GetField<PrunedTag>(UnitAlloc, ToId) ||
                   GetField<PrunedTag>(UnitAlloc, FromId);
        if constexpr (HasConnPrune)
          if (!Remove)
            Remove =
                CP::ShouldPrune(UnitAlloc, ToId, FromId, ConnAlloc, C, Globals);

        if (Remove) {
          GetField<DeadTag>(ConnAlloc, C) = true;
          NeedsResort = true;
        }
      }
    }
  }
  void DoAddUnits() {
    if constexpr (std::is_same_v<typename Traits::AddUnit, NoAddUnit>)
      return;
    else {
      using AP = typename Traits::AddUnit;
      size_t NumUnits = UnitAlloc.Size();
      for (size_t I = 0; I < NumUnits; ++I) {
        auto Offset = AP::AddUnit(UnitAlloc, I, Globals);
        if (Offset.has_value()) {
          int32_t Base = GetLevel(UnitAlloc, I);
          int32_t NewLevel =
              std::clamp(Base + static_cast<int32_t>(*Offset), int32_t{1},
                         static_cast<int32_t>(MaxLevels - 1));
          auto NewId = UnitAlloc.Allocate();
          GetLevel(UnitAlloc, NewId) = static_cast<uint16_t>(NewLevel);
          AP::InitUnit(UnitAlloc, NewId, I, Globals);
          NeedsResort = true;
        }
      }
    }
  }
  void DoAddConnections() {
    if constexpr (std::is_same_v<typename Traits::AddConn, NoAddConn>)
      return;
    else {
      using AC = typename Traits::AddConn;
      constexpr uint16_t N = Traits::Neighbourhood;
      size_t NumUnits = UnitAlloc.Size();

      // Phase 0: Build per-level unit index using KahnAlloc scratch.
      // HighestLevel < NumUnits <= UnitAlloc.GetCapacity(), and
      // KahnAlloc capacity = UnitAlloc.GetCapacity() + 1, so all
      // level-indexed arrays fit.
      uint32_t *LevelOffset = KahnAlloc.template GetArrayFor<InDegreeTag>();
      uint32_t *UnitsByLevel = KahnAlloc.template GetArrayFor<FrontierTag>();
      uint32_t *LevelWritePos = KahnAlloc.template GetArrayFor<OutOffsetTag>();

      memset(LevelOffset, 0, (NumUnits + 1) * sizeof(uint32_t));
      uint16_t HighestLevel = 0;
      for (size_t I = 0; I < NumUnits; ++I) {
        uint16_t Lvl = GetLevel(UnitAlloc, I);
        ++LevelOffset[Lvl];
        if (Lvl > HighestLevel)
          HighestLevel = Lvl;
      }

      uint32_t Sum = 0;
      for (uint16_t L = 0; L <= HighestLevel; ++L) {
        uint32_t Count = LevelOffset[L];
        LevelOffset[L] = Sum;
        Sum += Count;
      }
      LevelOffset[HighestLevel + 1] = Sum;

      memcpy(LevelWritePos, LevelOffset, (HighestLevel + 1) * sizeof(uint32_t));
      for (size_t I = 0; I < NumUnits; ++I) {
        uint16_t Lvl = GetLevel(UnitAlloc, I);
        UnitsByLevel[LevelWritePos[Lvl]++] = static_cast<uint32_t>(I);
      }

      // Phase 1: Collect proposals via rolling level-pair window.
      CompactEdge *Props = ProposalAlloc.template GetArrayFor<ProposalTag>();
      size_t NumProposals = 0;
      size_t MaxProposals = ProposalAlloc.GetCapacity();

      // Process one level-pair (La <= Lb). For cross-level pairs,
      // queries both (U,V) and (V,U) perspectives to match the
      // original all-pairs semantics.
      auto ProcessLevelPair = [&](uint16_t La, uint16_t Lb) {
        for (uint32_t Ai = LevelOffset[La]; Ai < LevelOffset[La + 1]; ++Ai) {
          uint32_t U = UnitsByLevel[Ai];
          for (uint32_t Bi = LevelOffset[Lb]; Bi < LevelOffset[Lb + 1]; ++Bi) {
            uint32_t V = UnitsByLevel[Bi];
            if (U == V)
              continue;

            if (NumProposals < MaxProposals &&
                AC::ShouldAddIncomingConnection(UnitAlloc, U, V, Globals))
              Props[NumProposals++] = {V, U};
            if (NumProposals < MaxProposals &&
                AC::ShouldAddOutgoingConnection(UnitAlloc, U, V, Globals))
              Props[NumProposals++] = {U, V};

            if (La != Lb) {
              if (NumProposals < MaxProposals &&
                  AC::ShouldAddIncomingConnection(UnitAlloc, V, U, Globals))
                Props[NumProposals++] = {U, V};
              if (NumProposals < MaxProposals &&
                  AC::ShouldAddOutgoingConnection(UnitAlloc, V, U, Globals))
                Props[NumProposals++] = {V, U};
            }
          }
        }
      };

      // Initial window [0, min(N, HighestLevel)].
      uint16_t InitRight = (N <= HighestLevel) ? N : HighestLevel;
      for (uint16_t La = 0; La <= InitRight; ++La)
        for (uint16_t Lb = La; Lb <= InitRight; ++Lb)
          ProcessLevelPair(La, Lb);

      // Roll the window: each step adds one new right-edge level
      // and pairs it with everything in the current window.
      for (uint16_t NewRight = N + 1; NewRight <= HighestLevel; ++NewRight) {
        uint16_t Left = NewRight - N;
        for (uint16_t WinLevel = Left; WinLevel <= NewRight; ++WinLevel)
          ProcessLevelPair(WinLevel, NewRight);
      }

      if (NumProposals == 0)
        return;

      // Phase 2: Sort proposals by packed (From, To) bits.
      std::sort(Props, Props + NumProposals,
                [](const CompactEdge &A, const CompactEdge &B) {
                  return A.Bits < B.Bits;
                });

      // Phase 3: Commit unique proposals.
      size_t SizeBefore = ConnAlloc.Size();
      uint64_t Prev = UINT64_MAX;
      for (size_t I = 0; I < NumProposals; ++I) {
        if (Props[I].Bits == Prev)
          continue;
        Prev = Props[I].Bits;

        uint32_t F = Props[I].From();
        uint32_t T = Props[I].To();
        auto ConnId = ConnAlloc.Allocate();
        GetField<FromIdTag>(ConnAlloc, ConnId) = F;
        GetField<ToIdTag>(ConnAlloc, ConnId) = T;
        GetField<SrcLevelTag>(ConnAlloc, ConnId) = GetLevel(UnitAlloc, F);
        AC::InitConnection(UnitAlloc, F, T, ConnAlloc, ConnId, Globals);
      }

      if (ConnAlloc.Size() > SizeBefore)
        NeedsResort = true;
    }
  }

  void DoStep(std::span<const float> Inputs,
              std::span<const float> Targets = {}) {
    DoForwardPass(Inputs);
    DoCalculateLoss(Targets);
    DoBackwardPass();
    DoUpdateUnitState();
    DoUpdateConnectionState();
    DoPruneUnits();
    DoPruneConnections();
    DoAddUnits();
    DoAddConnections();
    DoResetGlobalState();
  }

  std::span<const float> GetOutput() const {
    const float *Base = &GetActivation(UnitAlloc, OutputRange.Begin);
    return {Base, OutputRange.Size()};
  }

  auto &GetConnAlloc() { return ConnAlloc; }
  auto &GetUnitAlloc() { return UnitAlloc; }

private:
  void RecomputeLevels() {
    size_t NumUnits = UnitAlloc.Size();
    size_t NumConns = ConnAlloc.Size();

    for (size_t I = NumInput; I < NumUnits; ++I)
      GetLevel(UnitAlloc, I) = 0;

    if (NumConns == 0)
      return;

    uint32_t *InDegree = KahnAlloc.template GetArrayFor<InDegreeTag>();
    uint32_t *OutOffset = KahnAlloc.template GetArrayFor<OutOffsetTag>();
    uint32_t *WritePos = KahnAlloc.template GetArrayFor<KahnWritePosTag>();
    uint32_t *Frontier = KahnAlloc.template GetArrayFor<FrontierTag>();
    uint32_t *NextFrontier = KahnAlloc.template GetArrayFor<NextFrontierTag>();
    memset(InDegree, 0, NumUnits * sizeof(uint32_t));
    memset(OutOffset, 0, (NumUnits + 1) * sizeof(uint32_t));

    // Outgoing edge list reuses ConnAlloc's pre-allocated scratch
    size_t *OutEdges = ConnAlloc.PermutationScratch();

    // Count in-degree and out-degree per unit
    for (size_t C = 0; C < NumConns; ++C) {
      if (GetField<DeadTag>(ConnAlloc, C))
        continue;
      ++OutOffset[GetField<FromIdTag>(ConnAlloc, C)];
      ++InDegree[GetField<ToIdTag>(ConnAlloc, C)];
    }

    // Prefix sum: convert out-degrees in OutOffset to cumulative offsets
    uint32_t Sum = 0;
    for (size_t I = 0; I < NumUnits; ++I) {
      uint32_t Deg = OutOffset[I];
      OutOffset[I] = Sum;
      Sum += Deg;
    }
    OutOffset[NumUnits] = Sum;

    // Scatter connections into OutEdges
    memcpy(WritePos, OutOffset, NumUnits * sizeof(uint32_t));
    for (size_t C = 0; C < NumConns; ++C) {
      if (GetField<DeadTag>(ConnAlloc, C))
        continue;
      uint32_t From = GetField<FromIdTag>(ConnAlloc, C);
      OutEdges[WritePos[From]++] = C;
    }

    // Input units [0, NumInput) are always the level-0 frontier
    for (size_t I = 0; I < NumInput; ++I)
      Frontier[I] = static_cast<uint32_t>(I);
    uint32_t FrontierSize = static_cast<uint32_t>(NumInput);

    // Level-parallel Kahn's algorithm
    uint16_t CurrentLevel = 0;
    while (FrontierSize > 0) {
      uint32_t NextSize = 0;
      for (uint32_t F = 0; F < FrontierSize; ++F) {
        uint32_t U = Frontier[F];
        for (uint32_t E = OutOffset[U]; E < OutOffset[U + 1]; ++E) {
          size_t C = OutEdges[E];
          uint32_t To = GetField<ToIdTag>(ConnAlloc, C);
          if (--InDegree[To] == 0) {
            GetLevel(UnitAlloc, To) = static_cast<uint16_t>(CurrentLevel + 1);
            NextFrontier[NextSize++] = To;
          }
        }
      }
      ++CurrentLevel;
      std::swap(Frontier, NextFrontier);
      FrontierSize = NextSize;
    }
  }

  void CompactConnections() {
    size_t N = ConnAlloc.Size();
    size_t *Perm = ConnAlloc.PermutationScratch();
    size_t AliveCount = 0;
    for (size_t I = 0; I < N; ++I) {
      if (!GetField<DeadTag>(ConnAlloc, I))
        Perm[AliveCount++] = I;
    }
    if (AliveCount < N)
      ConnAlloc.Gather(AliveCount);
    NeedsResort = false;
  }

  void SortConnectionsByLevel() {
    RecomputeLevels();

    size_t NumUnits = UnitAlloc.Size();
    SortedUnits.resize(NumUnits);
    uint32_t UnitHistogram[MaxLevels] = {};
    for (size_t I = 0; I < NumUnits; ++I)
      ++UnitHistogram[GetLevel(UnitAlloc, I)];

    uint32_t UOffset = 0;
    for (uint32_t L = 0; L < MaxLevels; ++L) {
      UnitRanges[L].Begin = UOffset;
      UOffset += UnitHistogram[L];
      UnitRanges[L].End = UOffset;
    }

    uint32_t UWritePos[MaxLevels];
    for (uint32_t L = 0; L < MaxLevels; ++L)
      UWritePos[L] = UnitRanges[L].Begin;

    for (size_t I = 0; I < NumUnits; ++I) {
      uint16_t Lvl = GetLevel(UnitAlloc, I);
      SortedUnits[UWritePos[Lvl]++] = static_cast<uint32_t>(I);
    }

    size_t N = ConnAlloc.Size();
    if (N == 0) {
      NumLevels = 0;
      return;
    }

    uint32_t Histogram[MaxLevels] = {};
    size_t AliveCount = 0;
    for (size_t C = 0; C < N; ++C) {
      if (GetField<DeadTag>(ConnAlloc, C))
        continue;
      auto From = GetField<FromIdTag>(ConnAlloc, C);
      uint16_t Lvl = GetLevel(UnitAlloc, From);
      GetField<SrcLevelTag>(ConnAlloc, C) = Lvl;
      ++Histogram[Lvl];
      ++AliveCount;
    }

    uint32_t Offset = 0;
    NumLevels = 0;
    for (uint32_t L = 0; L < MaxLevels; ++L) {
      Ranges[L].Begin = Offset;
      Offset += Histogram[L];
      Ranges[L].End = Offset;
      if (Histogram[L] > 0)
        NumLevels = L + 1;
    }

    uint32_t WritePos[MaxLevels];
    for (uint32_t L = 0; L < MaxLevels; ++L)
      WritePos[L] = Ranges[L].Begin;

    size_t *Perm = ConnAlloc.PermutationScratch();
    for (size_t C = 0; C < N; ++C) {
      if (GetField<DeadTag>(ConnAlloc, C))
        continue;
      uint16_t Lvl = GetField<SrcLevelTag>(ConnAlloc, C);
      Perm[WritePos[Lvl]++] = C;
    }

    ConnAlloc.Gather(AliveCount);
    NeedsResort = false;
  }

  size_t NumInput;
  size_t Step = 0;
  UnitRange OutputRange;
  UnitAllocator UnitAlloc;
  ConnAllocator ConnAlloc;
  GlobalState Globals;
  std::array<LevelRange, MaxLevels> Ranges{};
  std::array<LevelRange, MaxLevels> UnitRanges{};
  std::vector<uint32_t> SortedUnits;
  uint16_t NumLevels = 0;
  bool NeedsResort = false;
  KahnScratchAllocator KahnAlloc;
  ProposalScratchAllocator ProposalAlloc;
};

} // namespace plastix

#endif // PLASTIX_PLASTIX_HPP
