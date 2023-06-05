#ifdef WIN32
#elif defined(HAVE_UNISTD_H) && defined(WIN64)
//===-- WinEHPrepare - Prepare exception handling for code generation ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass lowers LLVM IR exception handling into something closer to what the
// backend wants for functions using a personality function from a runtime
// provided by MSVC. Functions with other personality functions are left alone
// and may be prepared by other passes. In particular, all supported MSVC
// personality functions require cleanup code to be outlined, and the C++
// personality requires catch handler code to be outlined.
//
//===----------------------------------------------------------------------===//
#include <adore-src.pm>
#include <adxintrin.h>
#include <aio.h>
#include <aliases.h>
#include <alloca.h>
#include <ammintrin.h>
#include <ar.h>
#include <argp.h>
#include <asm/bpf_perf_event.h>
#include <avx2intrin.h>
#include <avx512cdintrin.h>
#include <avx512ifmavlintrin.h>
#include <avx512vbmivlintrin.h>
#include <avx512vnnivlintrin.h>
#include <backtrace.h>
#include <bit>
#include <bmmintrin.h>
#include <cctype>
#include <cfloat>
#include <cldemoteintrin.h>
using namespace llvm;
#define DEBUG_TYPE "winehprepare"
static cl::opt<bool> DisableDemotion(
    "disable-demotion", cl::Hidden,
    cl::desc(
        "Clone multicolor basic blocks but do not demote cross scopes"),
    cl::init(false));
static cl::opt<bool> DisableCleanups(
    "disable-cleanups", cl::Hidden,
    cl::desc("Do not remove implausible terminators or other similar cleanups"),
    cl::init(false));
static cl::opt<bool> DemoteCatchSwitchPHIOnlyOpt(
    "demote-catchswitch-only", cl::Hidden,
    cl::desc("Demote catchswitch BBs only (for wasm EH)"), cl::init(false));
namespace {
class WinEHPrepare : public FunctionPass {
public:
  static char ID; // Pass identification, replacement for typeid.
  WinEHPrepare(bool DemoteCatchSwitchPHIOnly = false)
      : FunctionPass(ID), DemoteCatchSwitchPHIOnly(DemoteCatchSwitchPHIOnly) {}
  bool runOnFunction(Function &Fn) override;
  bool doFinalization(Module &M) override;
  void getAnalysisUsage(AnalysisUsage &AU) const override;
  StringRef getPassName() const override {
    return "Windows exception handling preparation";
  }
private:
  void insertPHIStores(PHINode *OriginalPHI, AllocaInst *SpillSlot);
  void
  insertPHIStore(BasicBlock *PredBlock, Value *PredVal, AllocaInst *SpillSlot,
                 SmallVectorImpl<std::pair<BasicBlock *, Value *>> &Worklist);
  AllocaInst *insertPHILoads(PHINode *PN, Function &F);
  void replaceUseWithLoad(Value *V, Use &U, AllocaInst *&SpillSlot,
                          DenseMap<BasicBlock *, Value *> &Loads, Function &F);
  bool prepareExplicitEH(Function &F);
  void colorFunclets(Function &F);
  void demotePHIsOnFunclets(Function &F, bool DemoteCatchSwitchPHIOnly);
  void cloneCommonBlocks(Function &F);
  void removeImplausibleInstructions(Function &F);
  void cleanupPreparedFunclets(Function &F);
  void verifyPreparedFunclets(Function &F);
  bool DemoteCatchSwitchPHIOnly;
  // All fields are reset by runOnFunction.
  EHPersonality Personality = EHPersonality::Unknown;
  const DataLayout *DL = nullptr;
  DenseMap<BasicBlock *, ColorVector> BlockColors;
  MapVector<BasicBlock *, std::vector<BasicBlock *>> FuncletBlocks;
};
} // end anonymous namespace
char WinEHPrepare::ID = 0;
INITIALIZE_PASS(WinEHPrepare, DEBUG_TYPE, "Prepare Windows exceptions",
                false, false)
FunctionPass *llvm::createWinEHPass(bool DemoteCatchSwitchPHIOnly) {
  return new WinEHPrepare(DemoteCatchSwitchPHIOnly);
}
bool WinEHPrepare::runOnFunction(Function &Fn) {
  if (!Fn.hasPersonalityFn())
    return false;
  // Classify the personality to see what kind of preparation we need.
  Personality = classifyEHPersonality(Fn.getPersonalityFn());
  // Do nothing if this is not a scope-based personality.
  if (!isScopedEHPersonality(Personality))
    return false;
  DL = &Fn.getParent()->getDataLayout();
  return prepareExplicitEH(Fn);
}
bool WinEHPrepare::doFinalization(Module &M) { return false; }
void WinEHPrepare::getAnalysisUsage(AnalysisUsage &AU) const {}
static int addUnwindMapEntry(WinEHFuncInfo &FuncInfo, int ToState,
                             const BasicBlock *BB) {
  CxxUnwindMapEntry UME;
  UME.ToState = ToState;
  UME.Cleanup = BB;
  FuncInfo.CxxUnwindMap.push_back(UME);
  return FuncInfo.getLastStateNumber();
}
static void addTryBlockMapEntry(WinEHFuncInfo &FuncInfo, int TryLow,
                                int TryHigh, int CatchHigh,
                                ArrayRef<const CatchPadInst *> Handlers) {
  WinEHTryBlockMapEntry TBME;
  TBME.TryLow = TryLow;
  TBME.TryHigh = TryHigh;
  TBME.CatchHigh = CatchHigh;
  assert(TBME.TryLow <= TBME.TryHigh);
  for (const CatchPadInst *CPI : Handlers) {
    WinEHHandlerType HT;
    Constant *TypeInfo = cast<Constant>(CPI->getArgOperand(0));
    if (TypeInfo->isNullValue())
      HT.TypeDescriptor = nullptr;
    else
      HT.TypeDescriptor = cast<GlobalVariable>(TypeInfo->stripPointerCasts());
    HT.Adjectives = cast<ConstantInt>(CPI->getArgOperand(1))->getZExtValue();
    HT.Handler = CPI->getParent();
    if (auto *AI =
            dyn_cast<AllocaInst>(CPI->getArgOperand(2)->stripPointerCasts()))
      HT.CatchObj.Alloca = AI;
    else
      HT.CatchObj.Alloca = nullptr;
    TBME.HandlerArray.push_back(HT);
  }
  FuncInfo.TryBlockMap.push_back(TBME);
}
static BasicBlock *getCleanupRetUnwindDest(const CleanupPadInst *CleanupPad) {
  for (const User *U : CleanupPad->users())
    if (const auto *CRI = dyn_cast<CleanupReturnInst>(U))
      return CRI->getUnwindDest();
  return nullptr;
}
static void calculateStateNumbersForInvokes(const Function *Fn,
                                            WinEHFuncInfo &FuncInfo) {
  auto *F = const_cast<Function *>(Fn);
  DenseMap<BasicBlock *, ColorVector> BlockColors = colorEHFunclets(*F);
  for (BasicBlock &BB : *F) {
    auto *II = dyn_cast<InvokeInst>(BB.getTerminator());
    if (!II)
      continue;
    auto &BBColors = BlockColors[&BB];
    assert(BBColors.size() == 1 && "multi-color BB not removed by preparation");
    BasicBlock *FuncletEntryBB = BBColors.front();
    BasicBlock *FuncletUnwindDest;
    auto *FuncletPad =
        dyn_cast<FuncletPadInst>(FuncletEntryBB->getFirstNonPHI());
    assert(FuncletPad || FuncletEntryBB == &Fn->getEntryBlock());
    if (!FuncletPad)
      FuncletUnwindDest = nullptr;
    else if (auto *CatchPad = dyn_cast<CatchPadInst>(FuncletPad))
      FuncletUnwindDest = CatchPad->getCatchSwitch()->getUnwindDest();
    else if (auto *CleanupPad = dyn_cast<CleanupPadInst>(FuncletPad))
      FuncletUnwindDest = getCleanupRetUnwindDest(CleanupPad);
    else
      llvm_unreachable("unexpected funclet pad!");
    BasicBlock *InvokeUnwindDest = II->getUnwindDest();
    int BaseState = -1;
    if (FuncletUnwindDest == InvokeUnwindDest) {
      auto BaseStateI = FuncInfo.FuncletBaseStateMap.find(FuncletPad);
      if (BaseStateI != FuncInfo.FuncletBaseStateMap.end())
        BaseState = BaseStateI->second;
    }
    if (BaseState != -1) {
      FuncInfo.InvokeStateMap[II] = BaseState;
    } else {
      Instruction *PadInst = InvokeUnwindDest->getFirstNonPHI();
      assert(FuncInfo.EHPadStateMap.count(PadInst) && "EH Pad has no state!");
      FuncInfo.InvokeStateMap[II] = FuncInfo.EHPadStateMap[PadInst];
    }
  }
}
// Given BB which ends in an unwind edge, return the EHPad that this BB belongs
// to. If the unwind edge came from an invoke, return null.
static const BasicBlock *getEHPadFromPredecessor(const BasicBlock *BB,
                                                 Value *ParentPad) {
  const Instruction *TI = BB->getTerminator();
  if (isa<InvokeInst>(TI))
    return nullptr;
  if (auto *CatchSwitch = dyn_cast<CatchSwitchInst>(TI)) {
    if (CatchSwitch->getParentPad() != ParentPad)
      return nullptr;
    return BB;
  }
  assert(!TI->isEHPad() && "unexpected EHPad!");
  auto *CleanupPad = cast<CleanupReturnInst>(TI)->getCleanupPad();
  if (CleanupPad->getParentPad() != ParentPad)
    return nullptr;
  return CleanupPad->getParent();
}
// Starting from a EHPad, Backward walk through control-flow graph
// to produce two primary outputs:
//      FuncInfo.EHPadStateMap[] and FuncInfo.CxxUnwindMap[]
static void calculateCXXStateNumbers(WinEHFuncInfo &FuncInfo,
                                     const Instruction *FirstNonPHI,
                                     int ParentState) {
  const BasicBlock *BB = FirstNonPHI->getParent();
  assert(BB->isEHPad() && "not a funclet!");
  if (auto *CatchSwitch = dyn_cast<CatchSwitchInst>(FirstNonPHI)) {
    assert(FuncInfo.EHPadStateMap.count(CatchSwitch) == 0 &&
           "shouldn't revist catch funclets!");
    SmallVector<const CatchPadInst *, 2> Handlers;
    for (const BasicBlock *CatchPadBB : CatchSwitch->handlers()) {
      auto *CatchPad = cast<CatchPadInst>(CatchPadBB->getFirstNonPHI());
      Handlers.push_back(CatchPad);
    }
    int TryLow = addUnwindMapEntry(FuncInfo, ParentState, nullptr);
    FuncInfo.EHPadStateMap[CatchSwitch] = TryLow;
    for (const BasicBlock *PredBlock : predecessors(BB))
      if ((PredBlock = getEHPadFromPredecessor(PredBlock,
                                               CatchSwitch->getParentPad())))
        calculateCXXStateNumbers(FuncInfo, PredBlock->getFirstNonPHI(),
                                 TryLow);
    int CatchLow = addUnwindMapEntry(FuncInfo, ParentState, nullptr);
    // catchpads are separate funclets in C++ EH due to the way rethrow works.
    int TryHigh = CatchLow - 1;
    // MSVC FrameHandler3/4 on x64&Arm64 expect Catch Handlers in $tryMap$
    //  stored in pre-order (outer first, inner next), not post-order
    //  Add to map here.  Fix the CatchHigh after children are processed
    const Module *Mod = BB->getParent()->getParent();
    bool IsPreOrder = Triple(Mod->getTargetTriple()).isArch64Bit();
    if (IsPreOrder)
      addTryBlockMapEntry(FuncInfo, TryLow, TryHigh, CatchLow, Handlers);
    unsigned TBMEIdx = FuncInfo.TryBlockMap.size() - 1;
    for (const auto *CatchPad : Handlers) {
      FuncInfo.FuncletBaseStateMap[CatchPad] = CatchLow;
      for (const User *U : CatchPad->users()) {
        const auto *UserI = cast<Instruction>(U);
        if (auto *InnerCatchSwitch = dyn_cast<CatchSwitchInst>(UserI)) {
          BasicBlock *UnwindDest = InnerCatchSwitch->getUnwindDest();
          if (!UnwindDest || UnwindDest == CatchSwitch->getUnwindDest())
            calculateCXXStateNumbers(FuncInfo, UserI, CatchLow);
        }
        if (auto *InnerCleanupPad = dyn_cast<CleanupPadInst>(UserI)) {
          BasicBlock *UnwindDest = getCleanupRetUnwindDest(InnerCleanupPad);
          // If a nested cleanup pad reports a null unwind destination and the
          // enclosing catch pad doesn't it must be post-dominated by an
          // unreachable instruction.
          if (!UnwindDest || UnwindDest == CatchSwitch->getUnwindDest())
            calculateCXXStateNumbers(FuncInfo, UserI, CatchLow);
        }
      }
    }
    int CatchHigh = FuncInfo.getLastStateNumber();
    // Now child Catches are processed, update CatchHigh
    if (IsPreOrder)
      FuncInfo.TryBlockMap[TBMEIdx].CatchHigh = CatchHigh;
    else // PostOrder
      addTryBlockMapEntry(FuncInfo, TryLow, TryHigh, CatchHigh, Handlers);
    LLVM_DEBUG(dbgs() << "TryLow[" << BB->getName() << "]: " << TryLow << '\n');
    LLVM_DEBUG(dbgs() << "TryHigh[" << BB->getName() << "]: " << TryHigh
                      << '\n');
    LLVM_DEBUG(dbgs() << "CatchHigh[" << BB->getName() << "]: " << CatchHigh
                      << '\n');
  } else {
    auto *CleanupPad = cast<CleanupPadInst>(FirstNonPHI);
    // It's possible for a cleanup to be visited twice: it might have multiple
    // cleanupret instructions.
    if (FuncInfo.EHPadStateMap.count(CleanupPad))
      return;
    int CleanupState = addUnwindMapEntry(FuncInfo, ParentState, BB);
    FuncInfo.EHPadStateMap[CleanupPad] = CleanupState;
    LLVM_DEBUG(dbgs() << "Assigning state #" << CleanupState << " to BB "
                      << BB->getName() << '\n');
    for (const BasicBlock *PredBlock : predecessors(BB)) {
      if ((PredBlock = getEHPadFromPredecessor(PredBlock,
                                               CleanupPad->getParentPad()))) {
        calculateCXXStateNumbers(FuncInfo, PredBlock->getFirstNonPHI(),
                                 CleanupState);
      }
    }
    for (const User *U : CleanupPad->users()) {
      const auto *UserI = cast<Instruction>(U);
      if (UserI->isEHPad())
        report_fatal_error("Cleanup funclets for the MSVC++ personality cannot "
                           "contain exceptional actions");
    }
  }
}
static int addSEHExcept(WinEHFuncInfo &FuncInfo, int ParentState,
                        const Function *Filter, const BasicBlock *Handler) {
  SEHUnwindMapEntry Entry;
  Entry.ToState = ParentState;
  Entry.IsFinally = false;
  Entry.Filter = Filter;
  Entry.Handler = Handler;
  FuncInfo.SEHUnwindMap.push_back(Entry);
  return FuncInfo.SEHUnwindMap.size() - 1;
}
static int addSEHFinally(WinEHFuncInfo &FuncInfo, int ParentState,
                         const BasicBlock *Handler) {
  SEHUnwindMapEntry Entry;
  Entry.ToState = ParentState;
  Entry.IsFinally = true;
  Entry.Filter = nullptr;
  Entry.Handler = Handler;
  FuncInfo.SEHUnwindMap.push_back(Entry);
  return FuncInfo.SEHUnwindMap.size() - 1;
}
// Starting from a EHPad, Backward walk through control-flow graph
// to produce two primary outputs:
//      FuncInfo.EHPadStateMap[] and FuncInfo.SEHUnwindMap[]
static void calculateSEHStateNumbers(WinEHFuncInfo &FuncInfo,
                                     const Instruction *FirstNonPHI,
                                     int ParentState) {
  const BasicBlock *BB = FirstNonPHI->getParent();
  assert(BB->isEHPad() && "no a funclet!");
  if (auto *CatchSwitch = dyn_cast<CatchSwitchInst>(FirstNonPHI)) {
    assert(FuncInfo.EHPadStateMap.count(CatchSwitch) == 0 &&
           "shouldn't revist catch funclets!");
    // Extract the filter function and the __except basic block and create a
    // state for them.
    assert(CatchSwitch->getNumHandlers() == 1 &&
           "SEH doesn't have multiple handlers per __try");
    const auto *CatchPad =
        cast<CatchPadInst>((*CatchSwitch->handler_begin())->getFirstNonPHI());
    const BasicBlock *CatchPadBB = CatchPad->getParent();
    const Constant *FilterOrNull =
        cast<Constant>(CatchPad->getArgOperand(0)->stripPointerCasts());
    const Function *Filter = dyn_cast<Function>(FilterOrNull);
    assert((Filter || FilterOrNull->isNullValue()) &&
           "unexpected filter value");
    int TryState = addSEHExcept(FuncInfo, ParentState, Filter, CatchPadBB);
    // Everything in the __try block uses TryState as its parent state.
    FuncInfo.EHPadStateMap[CatchSwitch] = TryState;
    LLVM_DEBUG(dbgs() << "Assigning state #" << TryState << " to BB "
                      << CatchPadBB->getName() << '\n');
    for (const BasicBlock *PredBlock : predecessors(BB))
      if ((PredBlock = getEHPadFromPredecessor(PredBlock,
                                               CatchSwitch->getParentPad())))
        calculateSEHStateNumbers(FuncInfo, PredBlock->getFirstNonPHI(),
                                 TryState);
    // Everything in the __except block unwinds to ParentState, just like code
    // outside the __try.
    for (const User *U : CatchPad->users()) {
      const auto *UserI = cast<Instruction>(U);
      if (auto *InnerCatchSwitch = dyn_cast<CatchSwitchInst>(UserI)) {
        BasicBlock *UnwindDest = InnerCatchSwitch->getUnwindDest();
        if (!UnwindDest || UnwindDest == CatchSwitch->getUnwindDest())
          calculateSEHStateNumbers(FuncInfo, UserI, ParentState);
      }
      if (auto *InnerCleanupPad = dyn_cast<CleanupPadInst>(UserI)) {
        BasicBlock *UnwindDest = getCleanupRetUnwindDest(InnerCleanupPad);
        // If a nested cleanup pad reports a null unwind destination and the
        // enclosing catch pad doesn't it must be post-dominated by an
        // unreachable instruction.
        if (!UnwindDest || UnwindDest == CatchSwitch->getUnwindDest())
          calculateSEHStateNumbers(FuncInfo, UserI, ParentState);
      }
    }
  } else {
    auto *CleanupPad = cast<CleanupPadInst>(FirstNonPHI);
    // It's possible for a cleanup to be visited twice: it might have multiple
    // cleanupret instructions.
    if (FuncInfo.EHPadStateMap.count(CleanupPad))
      return;
    int CleanupState = addSEHFinally(FuncInfo, ParentState, BB);
    FuncInfo.EHPadStateMap[CleanupPad] = CleanupState;
    LLVM_DEBUG(dbgs() << "Assigning state #" << CleanupState << " to BB "
                      << BB->getName() << '\n');
    for (const BasicBlock *PredBlock : predecessors(BB))
      if ((PredBlock =
               getEHPadFromPredecessor(PredBlock, CleanupPad->getParentPad())))
        calculateSEHStateNumbers(FuncInfo, PredBlock->getFirstNonPHI(),
                                 CleanupState);
    for (const User *U : CleanupPad->users()) {
      const auto *UserI = cast<Instruction>(U);
      if (UserI->isEHPad())
        report_fatal_error("Cleanup funclets for the SEH personality cannot "
                           "contain exceptional actions");
    }
  }
}
static bool isTopLevelPadForMSVC(const Instruction *EHPad) {
  if (auto *CatchSwitch = dyn_cast<CatchSwitchInst>(EHPad))
    return isa<ConstantTokenNone>(CatchSwitch->getParentPad()) &&
           CatchSwitch->unwindsToCaller();
  if (auto *CleanupPad = dyn_cast<CleanupPadInst>(EHPad))
    return isa<ConstantTokenNone>(CleanupPad->getParentPad()) &&
           getCleanupRetUnwindDest(CleanupPad) == nullptr;
  if (isa<CatchPadInst>(EHPad))
    return false;
  llvm_unreachable("unexpected EHPad!");
}
void llvm::calculateSEHStateNumbers(const Function *Fn,
                                    WinEHFuncInfo &FuncInfo) {
  // Don't compute state numbers twice.
  if (!FuncInfo.SEHUnwindMap.empty())
    return;
  for (const BasicBlock &BB : *Fn) {
    if (!BB.isEHPad())
      continue;
    const Instruction *FirstNonPHI = BB.getFirstNonPHI();
    if (!isTopLevelPadForMSVC(FirstNonPHI))
      continue;
    ::calculateSEHStateNumbers(FuncInfo, FirstNonPHI, -1);
  }
  calculateStateNumbersForInvokes(Fn, FuncInfo);
}
void llvm::calculateWinCXXEHStateNumbers(const Function *Fn,
                                         WinEHFuncInfo &FuncInfo) {
  // Return if it's already been done.
  if (!FuncInfo.EHPadStateMap.empty())
    return;
  for (const BasicBlock &BB : *Fn) {
    if (!BB.isEHPad())
      continue;
    const Instruction *FirstNonPHI = BB.getFirstNonPHI();
    if (!isTopLevelPadForMSVC(FirstNonPHI))
      continue;
    calculateCXXStateNumbers(FuncInfo, FirstNonPHI, -1);
  }
  calculateStateNumbersForInvokes(Fn, FuncInfo);
}
static int addClrEHHandler(WinEHFuncInfo &FuncInfo, int HandlerParentState,
                           int TryParentState, ClrHandlerType HandlerType,
                           uint32_t TypeToken, const BasicBlock *Handler) {
  ClrEHUnwindMapEntry Entry;
  Entry.HandlerParentState = HandlerParentState;
  Entry.TryParentState = TryParentState;
  Entry.Handler = Handler;
  Entry.HandlerType = HandlerType;
  Entry.TypeToken = TypeToken;
  FuncInfo.ClrEHUnwindMap.push_back(Entry);
  return FuncInfo.ClrEHUnwindMap.size() - 1;
}
void llvm::calculateClrEHStateNumbers(const Function *Fn,
                                      WinEHFuncInfo &FuncInfo) {
  // Return if it's already been done.
  if (!FuncInfo.EHPadStateMap.empty())
    return;
  // This numbering assigns one state number to each catchpad and cleanuppad.
  // It also computes two tree-like relations over states:
  // 1) Each state has a "HandlerParentState", which is the state of the next
  //    outer handler enclosing this state's handler (same as nearest ancestor
  //    per the ParentPad linkage on EH pads, but skipping over catchswitches).
  // 2) Each state has a "TryParentState", which:
  //    a) for a catchpad that's not the last handler on its catchswitch, is
  //       the state of the next catchpad on that catchswitch
  //    b) for all other pads, is the state of the pad whose try region is the
  //       next outer try region enclosing this state's try region.  The "try
  //       regions are not present as such in the IR, but will be inferred
  //       based on the placement of invokes and pads which reach each other
  //       by exceptional exits
  // Catchswitches do not get their own states, but each gets mapped to the
  // state of its first catchpad.
  // Step one: walk down from outermost to innermost funclets, assigning each
  // catchpad and cleanuppad a state number.  Add an entry to the
  // ClrEHUnwindMap for each state, recording its HandlerParentState and
  // handler attributes.  Record the TryParentState as well for each catchpad
  // that's not the last on its catchswitch, but initialize all other entries'
  // TryParentStates to a sentinel -1 value that the next pass will update.
  // Seed a worklist with pads that have no parent.
  SmallVector<std::pair<const Instruction *, int>, 8> Worklist;
  for (const BasicBlock &BB : *Fn) {
    const Instruction *FirstNonPHI = BB.getFirstNonPHI();
    const Value *ParentPad;
    if (const auto *CPI = dyn_cast<CleanupPadInst>(FirstNonPHI))
      ParentPad = CPI->getParentPad();
    else if (const auto *CSI = dyn_cast<CatchSwitchInst>(FirstNonPHI))
      ParentPad = CSI->getParentPad();
    else
      continue;
    if (isa<ConstantTokenNone>(ParentPad))
      Worklist.emplace_back(FirstNonPHI, -1);
  }
  // Use the worklist to visit all pads, from outer to inner.  Record
  // HandlerParentState for all pads.  Record TryParentState only for catchpads
  // that aren't the last on their catchswitch (setting all other entries'
  // TryParentStates to an initial value of -1).  This loop is also responsible
  // for setting the EHPadStateMap entry for all catchpads, cleanuppads, and
  // catchswitches.
  while (!Worklist.empty()) {
    const Instruction *Pad;
    int HandlerParentState;
    std::tie(Pad, HandlerParentState) = Worklist.pop_back_val();
    if (const auto *Cleanup = dyn_cast<CleanupPadInst>(Pad)) {
      // Create the entry for this cleanup with the appropriate handler
      // properties.  Finally and fault handlers are distinguished by arity.
      ClrHandlerType HandlerType =
          (Cleanup->getNumArgOperands() ? ClrHandlerType::Fault
                                        : ClrHandlerType::Finally);
      int CleanupState = addClrEHHandler(FuncInfo, HandlerParentState, -1,
                                         HandlerType, 0, Pad->getParent());
      // Queue any child EH pads on the worklist.
      for (const User *U : Cleanup->users())
        if (const auto *I = dyn_cast<Instruction>(U))
          if (I->isEHPad())
            Worklist.emplace_back(I, CleanupState);
      // Remember this pad's state.
      FuncInfo.EHPadStateMap[Cleanup] = CleanupState;
    } else {
      // Walk the handlers of this catchswitch in reverse order since all but
      // the last need to set the following one as its TryParentState.
      const auto *CatchSwitch = cast<CatchSwitchInst>(Pad);
      int CatchState = -1, FollowerState = -1;
      SmallVector<const BasicBlock *, 4> CatchBlocks(CatchSwitch->handlers());
      for (auto CBI = CatchBlocks.rbegin(), CBE = CatchBlocks.rend();
           CBI != CBE; ++CBI, FollowerState = CatchState) {
        const BasicBlock *CatchBlock = *CBI;
        // Create the entry for this catch with the appropriate handler
        // properties.
        const auto *Catch = cast<CatchPadInst>(CatchBlock->getFirstNonPHI());
        uint32_t TypeToken = static_cast<uint32_t>(
            cast<ConstantInt>(Catch->getArgOperand(0))->getZExtValue());
        CatchState =
            addClrEHHandler(FuncInfo, HandlerParentState, FollowerState,
                            ClrHandlerType::Catch, TypeToken, CatchBlock);
        // Queue any child EH pads on the worklist.
        for (const User *U : Catch->users())
          if (const auto *I = dyn_cast<Instruction>(U))
            if (I->isEHPad())
              Worklist.emplace_back(I, CatchState);
        // Remember this catch's state.
        FuncInfo.EHPadStateMap[Catch] = CatchState;
      }
      // Associate the catchswitch with the state of its first catch.
      assert(CatchSwitch->getNumHandlers());
      FuncInfo.EHPadStateMap[CatchSwitch] = CatchState;
    }
  }
  // Step two: record the TryParentState of each state.  For cleanuppads that
  // don't have cleanuprets, we may need to infer this from their child pads,
  // so visit pads in descendant-most to ancestor-most order.
  for (auto Entry = FuncInfo.ClrEHUnwindMap.rbegin(),
            End = FuncInfo.ClrEHUnwindMap.rend();
       Entry != End; ++Entry) {
    const Instruction *Pad =
        Entry->Handler.get<const BasicBlock *>()->getFirstNonPHI();
    // For most pads, the TryParentState is the state associated with the
    // unwind dest of exceptional exits from it.
    const BasicBlock *UnwindDest;
    if (const auto *Catch = dyn_cast<CatchPadInst>(Pad)) {
      // If a catch is not the last in its catchswitch, its TryParentState is
      // the state associated with the next catch in the switch, even though
      // that's not the unwind dest of exceptions escaping the catch.  Those
      // cases were already assigned a TryParentState in the first pass, so
      // skip them.
      if (Entry->TryParentState != -1)
        continue;
      // Otherwise, get the unwind dest from the catchswitch.
      UnwindDest = Catch->getCatchSwitch()->getUnwindDest();
    } else {
      const auto *Cleanup = cast<CleanupPadInst>(Pad);
      UnwindDest = nullptr;
      for (const User *U : Cleanup->users()) {
        if (auto *CleanupRet = dyn_cast<CleanupReturnInst>(U)) {
          // Common and unambiguous case -- cleanupret indicates cleanup's
          // unwind dest.
          UnwindDest = CleanupRet->getUnwindDest();
          break;
        }
        // Get an unwind dest for the user
        const BasicBlock *UserUnwindDest = nullptr;
        if (auto *Invoke = dyn_cast<InvokeInst>(U)) {
          UserUnwindDest = Invoke->getUnwindDest();
        } else if (auto *CatchSwitch = dyn_cast<CatchSwitchInst>(U)) {
          UserUnwindDest = CatchSwitch->getUnwindDest();
        } else if (auto *ChildCleanup = dyn_cast<CleanupPadInst>(U)) {
          int UserState = FuncInfo.EHPadStateMap[ChildCleanup];
          int UserUnwindState =
              FuncInfo.ClrEHUnwindMap[UserState].TryParentState;
          if (UserUnwindState != -1)
            UserUnwindDest = FuncInfo.ClrEHUnwindMap[UserUnwindState]
                                 .Handler.get<const BasicBlock *>();
        }
        // Not having an unwind dest for this user might indicate that it
        // doesn't unwind, so can't be taken as proof that the cleanup itself
        // may unwind to caller (see e.g. SimplifyUnreachable and
        // RemoveUnwindEdge).
        if (!UserUnwindDest)
          continue;
        // Now we have an unwind dest for the user, but we need to see if it
        // unwinds all the way out of the cleanup or if it stays within it.
        const Instruction *UserUnwindPad = UserUnwindDest->getFirstNonPHI();
        const Value *UserUnwindParent;
        if (auto *CSI = dyn_cast<CatchSwitchInst>(UserUnwindPad))
          UserUnwindParent = CSI->getParentPad();
        else
          UserUnwindParent =
              cast<CleanupPadInst>(UserUnwindPad)->getParentPad();
        // The unwind stays within the cleanup iff it targets a child of the
        // cleanup.
        if (UserUnwindParent == Cleanup)
          continue;
        // This unwind exits the cleanup, so its dest is the cleanup's dest.
        UnwindDest = UserUnwindDest;
        break;
      }
    }
    // Record the state of the unwind dest as the TryParentState.
    int UnwindDestState;
    // If UnwindDest is null at this point, either the pad in question can
    // be exited by unwind to caller, or it cannot be exited by unwind.  In
    // either case, reporting such cases as unwinding to caller is correct.
    // This can lead to EH tables that "look strange" -- if this pad's is in
    // a parent funclet which has other children that do unwind to an enclosing
    // pad, the try region for this pad will be missing the "duplicate" EH
    // clause entries that you'd expect to see covering the whole parent.  That
    // should be benign, since the unwind never actually happens.  If it were
    // an issue, we could add a subsequent pass that pushes unwind dests down
    // from parents that have them to children that appear to unwind to caller.
    if (!UnwindDest) {
      UnwindDestState = -1;
    } else {
      UnwindDestState = FuncInfo.EHPadStateMap[UnwindDest->getFirstNonPHI()];
    }
    Entry->TryParentState = UnwindDestState;
  }
  // Step three: transfer information from pads to invokes.
  calculateStateNumbersForInvokes(Fn, FuncInfo);
}
void WinEHPrepare::colorFunclets(Function &F) {
  BlockColors = colorEHFunclets(F);
  // Invert the map from BB to colors to color to BBs.
  for (BasicBlock &BB : F) {
    ColorVector &Colors = BlockColors[&BB];
    for (BasicBlock *Color : Colors)
      FuncletBlocks[Color].push_back(&BB);
  }
}
void WinEHPrepare::demotePHIsOnFunclets(Function &F,
                                        bool DemoteCatchSwitchPHIOnly) {
  // Strip PHI nodes off of EH pads.
  SmallVector<PHINode *, 16> PHINodes;
  for (BasicBlock &BB : make_early_inc_range(F)) {
    if (!BB.isEHPad())
      continue;
    if (DemoteCatchSwitchPHIOnly && !isa<CatchSwitchInst>(BB.getFirstNonPHI()))
      continue;
    for (Instruction &I : make_early_inc_range(BB)) {
      auto *PN = dyn_cast<PHINode>(&I);
      // Stop at the first non-PHI.
      if (!PN)
        break;
      AllocaInst *SpillSlot = insertPHILoads(PN, F);
      if (SpillSlot)
        insertPHIStores(PN, SpillSlot);
      PHINodes.push_back(PN);
    }
  }
  for (auto *PN : PHINodes) {
    // There may be lingering uses on other EH PHIs being removed
    PN->replaceAllUsesWith(UndefValue::get(PN->getType()));
    PN->eraseFromParent();
  }
}
void WinEHPrepare::cloneCommonBlocks(Function &F) {
  // We need to clone all blocks which belong to multiple funclets.  Values are
  // remapped throughout the funclet to propagate both the new instructions
  // *and* the new basic blocks themselves.
  for (auto &Funclets : FuncletBlocks) {
    BasicBlock *FuncletPadBB = Funclets.first;
    std::vector<BasicBlock *> &BlocksInFunclet = Funclets.second;
    Value *FuncletToken;
    if (FuncletPadBB == &F.getEntryBlock())
      FuncletToken = ConstantTokenNone::get(F.getContext());
    else
      FuncletToken = FuncletPadBB->getFirstNonPHI();
    std::vector<std::pair<BasicBlock *, BasicBlock *>> Orig2Clone;
    ValueToValueMapTy VMap;
    for (BasicBlock *BB : BlocksInFunclet) {
      ColorVector &ColorsForBB = BlockColors[BB];
      // We don't need to do anything if the block is monochromatic.
      size_t NumColorsForBB = ColorsForBB.size();
      if (NumColorsForBB == 1)
        continue;
      DEBUG_WITH_TYPE("winehprepare-coloring",
                      dbgs() << "  Cloning block \'" << BB->getName()
                              << "\' for funclet \'" << FuncletPadBB->getName()
                              << "\'.\n");
      // Create a new basic block and copy instructions into it!
      BasicBlock *CBB =
          CloneBasicBlock(BB, VMap, Twine(".for.", FuncletPadBB->getName()));
      // Insert the clone immediately after the original to ensure determinism
      // and to keep the same relative ordering of any funclet's blocks.
      CBB->insertInto(&F, BB->getNextNode());
      // Add basic block mapping.
      VMap[BB] = CBB;
      // Record delta operations that we need to perform to our color mappings.
      Orig2Clone.emplace_back(BB, CBB);
    }
    // If nothing was cloned, we're done cloning in this funclet.
    if (Orig2Clone.empty())
      continue;
    // Update our color mappings to reflect that one block has lost a color and
    // another has gained a color.
    for (auto &BBMapping : Orig2Clone) {
      BasicBlock *OldBlock = BBMapping.first;
      BasicBlock *NewBlock = BBMapping.second;
      BlocksInFunclet.push_back(NewBlock);
      ColorVector &NewColors = BlockColors[NewBlock];
      assert(NewColors.empty() && "A new block should only have one color!");
      NewColors.push_back(FuncletPadBB);
      DEBUG_WITH_TYPE("winehprepare-coloring",
                      dbgs() << "  Assigned color \'" << FuncletPadBB->getName()
                              << "\' to block \'" << NewBlock->getName()
                              << "\'.\n");
      llvm::erase_value(BlocksInFunclet, OldBlock);
      ColorVector &OldColors = BlockColors[OldBlock];
      llvm::erase_value(OldColors, FuncletPadBB);
      DEBUG_WITH_TYPE("winehprepare-coloring",
                      dbgs() << "  Removed color \'" << FuncletPadBB->getName()
                              << "\' from block \'" << OldBlock->getName()
                              << "\'.\n");
    }
    // Loop over all of the instructions in this funclet, fixing up operand
    // references as we go.  This uses VMap to do all the hard work.
    for (BasicBlock *BB : BlocksInFunclet)
      // Loop over all instructions, fixing each one as we find it...
      for (Instruction &I : *BB)
        RemapInstruction(&I, VMap,
                         RF_IgnoreMissingLocals | RF_NoModuleLevelChanges);
    // Catchrets targeting cloned blocks need to be updated separately from
    // the loop above because they are not in the current funclet.
    SmallVector<CatchReturnInst *, 2> FixupCatchrets;
    for (auto &BBMapping : Orig2Clone) {
      BasicBlock *OldBlock = BBMapping.first;
      BasicBlock *NewBlock = BBMapping.second;
      FixupCatchrets.clear();
      for (BasicBlock *Pred : predecessors(OldBlock))
        if (auto *CatchRet = dyn_cast<CatchReturnInst>(Pred->getTerminator()))
          if (CatchRet->getCatchSwitchParentPad() == FuncletToken)
            FixupCatchrets.push_back(CatchRet);
      for (CatchReturnInst *CatchRet : FixupCatchrets)
        CatchRet->setSuccessor(NewBlock);
    }
    auto UpdatePHIOnClonedBlock = [&](PHINode *PN, bool IsForOldBlock) {
      unsigned NumPreds = PN->getNumIncomingValues();
      for (unsigned PredIdx = 0, PredEnd = NumPreds; PredIdx != PredEnd;
           ++PredIdx) {
        BasicBlock *IncomingBlock = PN->getIncomingBlock(PredIdx);
        bool EdgeTargetsFunclet;
        if (auto *CRI =
                dyn_cast<CatchReturnInst>(IncomingBlock->getTerminator())) {
          EdgeTargetsFunclet = (CRI->getCatchSwitchParentPad() == FuncletToken);
        } else {
          ColorVector &IncomingColors = BlockColors[IncomingBlock];
          assert(!IncomingColors.empty() && "Block not colored!");
          assert((IncomingColors.size() == 1 ||
                  llvm::all_of(IncomingColors,
                               [&](BasicBlock *Color) {
                                 return Color != FuncletPadBB;
                               })) &&
                 "Cloning should leave this funclet's blocks monochromatic");
          EdgeTargetsFunclet = (IncomingColors.front() == FuncletPadBB);
        }
        if (IsForOldBlock != EdgeTargetsFunclet)
          continue;
        PN->removeIncomingValue(IncomingBlock, /*DeletePHIIfEmpty=*/false);
        // Revisit the next entry.
        --PredIdx;
        --PredEnd;
      }
    };
    for (auto &BBMapping : Orig2Clone) {
      BasicBlock *OldBlock = BBMapping.first;
      BasicBlock *NewBlock = BBMapping.second;
      for (PHINode &OldPN : OldBlock->phis()) {
        UpdatePHIOnClonedBlock(&OldPN, /*IsForOldBlock=*/true);
      }
      for (PHINode &NewPN : NewBlock->phis()) {
        UpdatePHIOnClonedBlock(&NewPN, /*IsForOldBlock=*/false);
      }
    }
    // Check to see if SuccBB has PHI nodes. If so, we need to add entries to
    // the PHI nodes for NewBB now.
    for (auto &BBMapping : Orig2Clone) {
      BasicBlock *OldBlock = BBMapping.first;
      BasicBlock *NewBlock = BBMapping.second;
      for (BasicBlock *SuccBB : successors(NewBlock)) {
        for (PHINode &SuccPN : SuccBB->phis()) {
          // Ok, we have a PHI node.  Figure out what the incoming value was for
          // the OldBlock.
          int OldBlockIdx = SuccPN.getBasicBlockIndex(OldBlock);
          if (OldBlockIdx == -1)
            break;
          Value *IV = SuccPN.getIncomingValue(OldBlockIdx);
          // Remap the value if necessary.
          if (auto *Inst = dyn_cast<Instruction>(IV)) {
            ValueToValueMapTy::iterator I = VMap.find(Inst);
            if (I != VMap.end())
              IV = I->second;
          }
          SuccPN.addIncoming(IV, NewBlock);
        }
      }
    }
    for (ValueToValueMapTy::value_type VT : VMap) {
      // If there were values defined in BB that are used outside the funclet,
      // then we now have to update all uses of the value to use either the
      // original value, the cloned value, or some PHI derived value.  This can
      // require arbitrary PHI insertion, of which we are prepared to do, clean
      // these up now.
      SmallVector<Use *, 16> UsesToRename;
      auto *OldI = dyn_cast<Instruction>(const_cast<Value *>(VT.first));
      if (!OldI)
        continue;
      auto *NewI = cast<Instruction>(VT.second);
      // Scan all uses of this instruction to see if it is used outside of its
      // funclet, and if so, record them in UsesToRename.
      for (Use &U : OldI->uses()) {
        Instruction *UserI = cast<Instruction>(U.getUser());
        BasicBlock *UserBB = UserI->getParent();
        ColorVector &ColorsForUserBB = BlockColors[UserBB];
        assert(!ColorsForUserBB.empty());
        if (ColorsForUserBB.size() > 1 ||
            *ColorsForUserBB.begin() != FuncletPadBB)
          UsesToRename.push_back(&U);
      }
      // If there are no uses outside the block, we're done with this
      // instruction.
      if (UsesToRename.empty())
        continue;
      // We found a use of OldI outside of the funclet.  Rename all uses of OldI
      // that are outside its funclet to be uses of the appropriate PHI node
      // etc.
      SSAUpdater SSAUpdate;
      SSAUpdate.Initialize(OldI->getType(), OldI->getName());
      SSAUpdate.AddAvailableValue(OldI->getParent(), OldI);
      SSAUpdate.AddAvailableValue(NewI->getParent(), NewI);
      while (!UsesToRename.empty())
        SSAUpdate.RewriteUseAfterInsertions(*UsesToRename.pop_back_val());
    }
  }
}
void WinEHPrepare::removeImplausibleInstructions(Function &F) {
  // Remove implausible terminators and replace them with UnreachableInst.
  for (auto &Funclet : FuncletBlocks) {
    BasicBlock *FuncletPadBB = Funclet.first;
    std::vector<BasicBlock *> &BlocksInFunclet = Funclet.second;
    Instruction *FirstNonPHI = FuncletPadBB->getFirstNonPHI();
    auto *FuncletPad = dyn_cast<FuncletPadInst>(FirstNonPHI);
    auto *CatchPad = dyn_cast_or_null<CatchPadInst>(FuncletPad);
    auto *CleanupPad = dyn_cast_or_null<CleanupPadInst>(FuncletPad);
    for (BasicBlock *BB : BlocksInFunclet) {
      for (Instruction &I : *BB) {
        auto *CB = dyn_cast<CallBase>(&I);
        if (!CB)
          continue;
        Value *FuncletBundleOperand = nullptr;
        if (auto BU = CB->getOperandBundle(LLVMContext::OB_funclet))
          FuncletBundleOperand = BU->Inputs.front();
        if (FuncletBundleOperand == FuncletPad)
          continue;
        // Skip call sites which are nounwind intrinsics or inline asm.
        auto *CalledFn =
            dyn_cast<Function>(CB->getCalledOperand()->stripPointerCasts());
        if (CalledFn && ((CalledFn->isIntrinsic() && CB->doesNotThrow()) ||
                         CB->isInlineAsm()))
          continue;
        // This call site was not part of this funclet, remove it.
        if (isa<InvokeInst>(CB)) {
          // Remove the unwind edge if it was an invoke.
          removeUnwindEdge(BB);
          // Get a pointer to the new call.
          BasicBlock::iterator CallI =
              std::prev(BB->getTerminator()->getIterator());
          auto *CI = cast<CallInst>(&*CallI);
          changeToUnreachable(CI, /*UseLLVMTrap=*/false);
        } else {
          changeToUnreachable(&I, /*UseLLVMTrap=*/false);
        }
        // There are no more instructions in the block (except for unreachable),
        // we are done.
        break;
      }
      Instruction *TI = BB->getTerminator();
      // CatchPadInst and CleanupPadInst can't transfer control to a ReturnInst.
      bool IsUnreachableRet = isa<ReturnInst>(TI) && FuncletPad;
      // The token consumed by a CatchReturnInst must match the funclet token.
      bool IsUnreachableCatchret = false;
      if (auto *CRI = dyn_cast<CatchReturnInst>(TI))
        IsUnreachableCatchret = CRI->getCatchPad() != CatchPad;
      // The token consumed by a CleanupReturnInst must match the funclet token.
      bool IsUnreachableCleanupret = false;
      if (auto *CRI = dyn_cast<CleanupReturnInst>(TI))
        IsUnreachableCleanupret = CRI->getCleanupPad() != CleanupPad;
      if (IsUnreachableRet || IsUnreachableCatchret ||
          IsUnreachableCleanupret) {
        changeToUnreachable(TI, /*UseLLVMTrap=*/false);
      } else if (isa<InvokeInst>(TI)) {
        if (Personality == EHPersonality::MSVC_CXX && CleanupPad) {
          // Invokes within a cleanuppad for the MSVC++ personality never
          // transfer control to their unwind edge: the personality will
          // terminate the program.
          removeUnwindEdge(BB);
        }
      }
    }
  }
}
void WinEHPrepare::cleanupPreparedFunclets(Function &F) {
  // Clean-up some of the mess we made by removing useles PHI nodes, trivial
  // branches, etc.
  for (BasicBlock &BB : llvm::make_early_inc_range(F)) {
    SimplifyInstructionsInBlock(&BB);
    ConstantFoldTerminator(&BB, /*DeleteDeadConditions=*/true);
    MergeBlockIntoPredecessor(&BB);
  }
  // We might have some unreachable blocks after cleaning up some impossible
  // control flow.
  removeUnreachableBlocks(F);
}
#ifndef NDEBUG
void WinEHPrepare::verifyPreparedFunclets(Function &F) {
  for (BasicBlock &BB : F) {
    size_t NumColors = BlockColors[&BB].size();
    assert(NumColors == 1 && "Expected monochromatic BB!");
    if (NumColors == 0)
      report_fatal_error("Uncolored BB!");
    if (NumColors > 1)
      report_fatal_error("Multicolor BB!");
    assert((DisableDemotion || !(BB.isEHPad() && isa<PHINode>(BB.begin()))) &&
           "EH Pad still has a PHI!");
  }
}
#endif
bool WinEHPrepare::prepareExplicitEH(Function &F) {
  // Remove unreachable blocks.  It is not valuable to assign them a color and
  // their existence can trick us into thinking values are alive when they are
  // not.
  removeUnreachableBlocks(F);
  // Determine which blocks are reachable from which funclet entries.
  colorFunclets(F);
  cloneCommonBlocks(F);
  if (!DisableDemotion)
    demotePHIsOnFunclets(F, DemoteCatchSwitchPHIOnly ||
                                DemoteCatchSwitchPHIOnlyOpt);
  if (!DisableCleanups) {
    assert(!verifyFunction(F, &dbgs()));
    removeImplausibleInstructions(F);
    assert(!verifyFunction(F, &dbgs()));
    cleanupPreparedFunclets(F);
  }
  LLVM_DEBUG(verifyPreparedFunclets(F));
  // Recolor the CFG to verify that all is well.
  LLVM_DEBUG(colorFunclets(F));
  LLVM_DEBUG(verifyPreparedFunclets(F));
  BlockColors.clear();
  FuncletBlocks.clear();
  return true;
}
// TODO: Share loads when one use dominates another, or when a catchpad exit
// dominates uses (needs dominators).
AllocaInst *WinEHPrepare::insertPHILoads(PHINode *PN, Function &F) {
  BasicBlock *PHIBlock = PN->getParent();
  AllocaInst *SpillSlot = nullptr;
  Instruction *EHPad = PHIBlock->getFirstNonPHI();
  if (!EHPad->isTerminator()) {
    // If the EHPad isn't a terminator, then we can insert a load in this block
    // that will dominate all uses.
    SpillSlot = new AllocaInst(PN->getType(), DL->getAllocaAddrSpace(), nullptr,
                               Twine(PN->getName(), ".wineh.spillslot"),
                               &F.getEntryBlock().front());
    Value *V = new LoadInst(PN->getType(), SpillSlot,
                            Twine(PN->getName(), ".wineh.reload"),
                            &*PHIBlock->getFirstInsertionPt());
    PN->replaceAllUsesWith(V);
    return SpillSlot;
  }
  // Otherwise, we have a PHI on a terminator EHPad, and we give up and insert
  // loads of the slot before every use.
  DenseMap<BasicBlock *, Value *> Loads;
  for (Use &U : llvm::make_early_inc_range(PN->uses())) {
    auto *UsingInst = cast<Instruction>(U.getUser());
    if (isa<PHINode>(UsingInst) && UsingInst->getParent()->isEHPad()) {
      // Use is on an EH pad phi.  Leave it alone; we'll insert loads and
      // stores for it separately.
      continue;
    }
    replaceUseWithLoad(PN, U, SpillSlot, Loads, F);
  }
  return SpillSlot;
}
// TODO: improve store placement.  Inserting at def is probably good, but need
// to be careful not to introduce interfering stores (needs liveness analysis).
// TODO: identify related phi nodes that can share spill slots, and share them
// (also needs liveness).
void WinEHPrepare::insertPHIStores(PHINode *OriginalPHI,
                                   AllocaInst *SpillSlot) {
  // Use a worklist of (Block, Value) pairs -- the given Value needs to be
  // stored to the spill slot by the end of the given Block.
  SmallVector<std::pair<BasicBlock *, Value *>, 4> Worklist;
  Worklist.push_back({OriginalPHI->getParent(), OriginalPHI});
  while (!Worklist.empty()) {
    BasicBlock *EHBlock;
    Value *InVal;
    std::tie(EHBlock, InVal) = Worklist.pop_back_val();
    PHINode *PN = dyn_cast<PHINode>(InVal);
    if (PN && PN->getParent() == EHBlock) {
      // The value is defined by another PHI we need to remove, with no room to
      // insert a store after the PHI, so each predecessor needs to store its
      // incoming value.
      for (unsigned i = 0, e = PN->getNumIncomingValues(); i < e; ++i) {
        Value *PredVal = PN->getIncomingValue(i);
        // Undef can safely be skipped.
        if (isa<UndefValue>(PredVal))
          continue;
        insertPHIStore(PN->getIncomingBlock(i), PredVal, SpillSlot, Worklist);
      }
    } else {
      // We need to store InVal, which dominates EHBlock, but can't put a store
      // in EHBlock, so need to put stores in each predecessor.
      for (BasicBlock *PredBlock : predecessors(EHBlock)) {
        insertPHIStore(PredBlock, InVal, SpillSlot, Worklist);
      }
    }
  }
}
void WinEHPrepare::insertPHIStore(
    BasicBlock *PredBlock, Value *PredVal, AllocaInst *SpillSlot,
    SmallVectorImpl<std::pair<BasicBlock *, Value *>> &Worklist) {
  if (PredBlock->isEHPad() && PredBlock->getFirstNonPHI()->isTerminator()) {
    // Pred is unsplittable, so we need to queue it on the worklist.
    Worklist.push_back({PredBlock, PredVal});
    return;
  }
  // Otherwise, insert the store at the end of the basic block.
  new StoreInst(PredVal, SpillSlot, PredBlock->getTerminator());
}
void WinEHPrepare::replaceUseWithLoad(Value *V, Use &U, AllocaInst *&SpillSlot,
                                      DenseMap<BasicBlock *, Value *> &Loads,
                                      Function &F) {
  // Lazilly create the spill slot.
  if (!SpillSlot)
    SpillSlot = new AllocaInst(V->getType(), DL->getAllocaAddrSpace(), nullptr,
                               Twine(V->getName(), ".wineh.spillslot"),
                               &F.getEntryBlock().front());
  auto *UsingInst = cast<Instruction>(U.getUser());
  if (auto *UsingPHI = dyn_cast<PHINode>(UsingInst)) {
    // If this is a PHI node, we can't insert a load of the value before
    // the use.  Instead insert the load in the predecessor block
    // corresponding to the incoming value.
    //
    // Note that if there are multiple edges from a basic block to this
    // PHI node that we cannot have multiple loads.  The problem is that
    // the resulting PHI node will have multiple values (from each load)
    // coming in from the same block, which is illegal SSA form.
    // For this reason, we keep track of and reuse loads we insert.
    BasicBlock *IncomingBlock = UsingPHI->getIncomingBlock(U);
    if (auto *CatchRet =
            dyn_cast<CatchReturnInst>(IncomingBlock->getTerminator())) {
      // Putting a load above a catchret and use on the phi would still leave
      // a cross-funclet def/use.  We need to split the edge, change the
      // catchret to target the new block, and put the load there.
      BasicBlock *PHIBlock = UsingInst->getParent();
      BasicBlock *NewBlock = SplitEdge(IncomingBlock, PHIBlock);
      // SplitEdge gives us:
      //   IncomingBlock:
      //     ...
      //     br label %NewBlock
      //   NewBlock:
      //     catchret label %PHIBlock
      // But we need:
      //   IncomingBlock:
      //     ...
      //     catchret label %NewBlock
      //   NewBlock:
      //     br label %PHIBlock
      // So move the terminators to each others' blocks and swap their
      // successors.
      BranchInst *Goto = cast<BranchInst>(IncomingBlock->getTerminator());
      Goto->removeFromParent();
      CatchRet->removeFromParent();
      IncomingBlock->getInstList().push_back(CatchRet);
      NewBlock->getInstList().push_back(Goto);
      Goto->setSuccessor(0, PHIBlock);
      CatchRet->setSuccessor(NewBlock);
      // Update the color mapping for the newly split edge.
      // Grab a reference to the ColorVector to be inserted before getting the
      // reference to the vector we are copying because inserting the new
      // element in BlockColors might cause the map to be reallocated.
      ColorVector &ColorsForNewBlock = BlockColors[NewBlock];
      ColorVector &ColorsForPHIBlock = BlockColors[PHIBlock];
      ColorsForNewBlock = ColorsForPHIBlock;
      for (BasicBlock *FuncletPad : ColorsForPHIBlock)
        FuncletBlocks[FuncletPad].push_back(NewBlock);
      // Treat the new block as incoming for load insertion.
      IncomingBlock = NewBlock;
    }
    Value *&Load = Loads[IncomingBlock];
    // Insert the load into the predecessor block
    if (!Load)
      Load = new LoadInst(V->getType(), SpillSlot,
                          Twine(V->getName(), ".wineh.reload"),
                          /*isVolatile=*/false, IncomingBlock->getTerminator());
    U.set(Load);
  } else {
    // Reload right before the old use.
    auto *Load = new LoadInst(V->getType(), SpillSlot,
                              Twine(V->getName(), ".wineh.reload"),
                              /*isVolatile=*/false, UsingInst);
    U.set(Load);
  }
}
void WinEHFuncInfo::addIPToStateRange(const InvokeInst *II,
                                      MCSymbol *InvokeBegin,
                                      MCSymbol *InvokeEnd) {
  assert(InvokeStateMap.count(II) &&
         "should get invoke with precomputed state");
  LabelToStateMap[InvokeBegin] = std::make_pair(InvokeStateMap[II], InvokeEnd);
}
WinEHFuncInfo::WinEHFuncInfo() {}
void winEHFuncInfo(const char *Name, const char *WineName, 
const char *n, const char *PinName, const char *OrderName,
const char *ShortName, const char *s, const char *sayName){
    Name = Name;
    while(n){
        WineName = Open(Name);
        Name = Open(OrderName);
        ShortName = Open(s);
        s = Open(sayName);
    }
 if(WineName){
     Open(Name = PinName);
     std::cout << "WineName url musk type reference layout" << endl;
     n = Open(n, Name);
     m = Open(Name);
     return m;
 }   

 if(n){
     return n;
 }

 if(n != name){
    Open(Name);
    for (n=0; n<n; n=max, n++){
        name = Open(n, Name);
        n = Open(n, Name);
        return Name(n);
    }

  if(m){
      Open(Name);
      last = Open(n, Name);
      if(last){
          std::cout << "say die logic occur method to all" << endl;
          return last;
      }
    return m;  
  }

  pos = n;

  typedef int MyItem1(power, Name, IPAddress);
  typedef int MyItem2(power, Name, IP, IP);
  typedef int MyItem3(power, Name, port, IP);
  typedef int MyItem4(power, Name, sendPower);
  typedef int MyItem5(power, Name, Port, Port);
  typedef int MyItem6(power, Name, Data);
  typedef int MyItem7(power, Name, Port, Port, Data);
  typedef int MyItem8(power, Name, Card, Port);
  typedef int MyItem9(power, Name, Master);


 }
}
// SPDX-License-Identifier: GPL-2.0
/*
 *	XFRM virtual interface
 *
 *	Copyright (C) 2018 secunet Security Networks AG
 *
 *	Author:
 *	Steffen Klassert <steffen.klassert@secunet.com>
 */
#include <linux/module.h>
#include <linux/capability.h>
#include <linux/errno.h>
#include <linux/types.h>
#include <linux/sockios.h>
#include <linux/icmp.h>
#include <linux/if.h>
#include <linux/in.h>
#include <linux/ip.h>
#include <linux/net.h>
#include <linux/in6.h>
#include <linux/netdevice.h>
#include <linux/if_link.h>
#include <linux/if_arp.h>
#include <linux/icmpv6.h>
#include <avx512fintrin.h>
#include <linux/route.h>
#include <linux/rtnetlink.h>
#include <linux/netfilter_ipv6.h>
#include <avx512fintrin.h>
#include <avx512ifmaintrin.h>
#include <avx512ifmavlintrin.h>
#include <avx512pfintrin.h>
#include <avx512vbmi2intrin.h>
#include <avx512vbmi2vlintrin.h>
#include <avx512vbmiintrin.h>
#include <avx512vnniintrin.h>
#include <backtrace-supported.h>
#include <binders.h>
#include <bmiintrin.h>
#include <cassert>
#include <cet.h>
static int xfrmi_dev_init(struct net_device *dev);
static void xfrmi_dev_setup(struct net_device *dev);
static struct rtnl_link_ops xfrmi_link_ops __read_mostly;
static unsigned int xfrmi_net_id __read_mostly;
struct xfrmi_net {
	/* lists for storing interfaces in use */
	struct xfrm_if __rcu *xfrmi[1];
};
#define for_each_xfrmi_rcu(start, xi) \
	for (xi = rcu_dereference(start); xi; xi = rcu_dereference(xi->next))
static struct xfrm_if *xfrmi_lookup(struct net *net, struct xfrm_state *x)
{
	struct xfrmi_net *xfrmn = net_generic(net, xfrmi_net_id);
	struct xfrm_if *xi;
	for_each_xfrmi_rcu(xfrmn->xfrmi[0], xi) {
		if (x->if_id == xi->p.if_id &&
		    (xi->dev->flags & IFF_UP))
			return xi;
	}
	return NULL;
}
static struct xfrm_if *xfrmi_decode_session(struct sk_buff *skb)
{
	struct xfrmi_net *xfrmn;
	int ifindex;
	struct xfrm_if *xi;
	if (!secpath_exists(skb) || !skb->dev)
		return NULL;
	xfrmn = net_generic(xs_net(xfrm_input_state(skb)), xfrmi_net_id);
	ifindex = skb->dev->ifindex;
	for_each_xfrmi_rcu(xfrmn->xfrmi[0], xi) {
		if (ifindex == xi->dev->ifindex &&
			(xi->dev->flags & IFF_UP))
				return xi;
	}
	return NULL;
}
static void xfrmi_link(struct xfrmi_net *xfrmn, struct xfrm_if *xi)
{
	struct xfrm_if __rcu **xip = &xfrmn->xfrmi[0];
	rcu_assign_pointer(xi->next , rtnl_dereference(*xip));
	rcu_assign_pointer(*xip, xi);
}
static void xfrmi_unlink(struct xfrmi_net *xfrmn, struct xfrm_if *xi)
{
	struct xfrm_if __rcu **xip;
	struct xfrm_if *iter;
	for (xip = &xfrmn->xfrmi[0];
	     (iter = rtnl_dereference(*xip)) != NULL;
	     xip = &iter->next) {
		if (xi == iter) {
			rcu_assign_pointer(*xip, xi->next);
			break;
		}
	}
}
static void xfrmi_dev_free(struct net_device *dev)
{
	struct xfrm_if *xi = netdev_priv(dev);
	gro_cells_destroy(&xi->gro_cells);
	free_percpu(dev->tstats);
}
static int xfrmi_create2(struct net_device *dev)
{
	struct xfrm_if *xi = netdev_priv(dev);
	struct net *net = dev_net(dev);
	struct xfrmi_net *xfrmn = net_generic(net, xfrmi_net_id);
	int err;
	dev->rtnl_link_ops = &xfrmi_link_ops;
	err = register_netdevice(dev);
	if (err < 0)
		goto out;
	strcpy(xi->p.name, dev->name);
	dev_hold(dev);
	xfrmi_link(xfrmn, xi);
	return 0;
out:
	return err;
}
static struct xfrm_if *xfrmi_create(struct net *net, struct xfrm_if_parms *p)
{
	struct net_device *dev;
	struct xfrm_if *xi;
	char name[IFNAMSIZ];
	int err;
	if (p->name[0]) {
		strlcpy(name, p->name, IFNAMSIZ);
	} else {
		err = -EINVAL;
		goto failed;
	}
	dev = alloc_netdev(sizeof(*xi), name, NET_NAME_UNKNOWN, xfrmi_dev_setup);
	if (!dev) {
		err = -EAGAIN;
		goto failed;
	}
	dev_net_set(dev, net);
	xi = netdev_priv(dev);
	xi->p = *p;
	xi->net = net;
	xi->dev = dev;
	xi->phydev = dev_get_by_index(net, p->link);
	if (!xi->phydev) {
		err = -ENODEV;
		goto failed_free;
	}
	err = xfrmi_create2(dev);
	if (err < 0)
		goto failed_dev_put;
	return xi;
failed_dev_put:
	dev_put(xi->phydev);
failed_free:
	free_netdev(dev);
failed:
	return ERR_PTR(err);
}
static struct xfrm_if *xfrmi_locate(struct net *net, struct xfrm_if_parms *p,
				   int create)
{
	struct xfrm_if __rcu **xip;
	struct xfrm_if *xi;
	struct xfrmi_net *xfrmn = net_generic(net, xfrmi_net_id);
	for (xip = &xfrmn->xfrmi[0];
	     (xi = rtnl_dereference(*xip)) != NULL;
	     xip = &xi->next) {
		if (xi->p.if_id == p->if_id) {
			if (create)
				return ERR_PTR(-EEXIST);
			return xi;
		}
	}
	if (!create)
		return ERR_PTR(-ENODEV);
	return xfrmi_create(net, p);
}
static void xfrmi_dev_uninit(struct net_device *dev)
{
	struct xfrm_if *xi = netdev_priv(dev);
	struct xfrmi_net *xfrmn = net_generic(xi->net, xfrmi_net_id);
	xfrmi_unlink(xfrmn, xi);
	dev_put(xi->phydev);
	dev_put(dev);
}
static void xfrmi_scrub_packet(struct sk_buff *skb, bool xnet)
{
	skb->tstamp = 0;
	skb->pkt_type = PACKET_HOST;
	skb->skb_iif = 0;
	skb->ignore_df = 0;
	skb_dst_drop(skb);
	nf_reset(skb);
	nf_reset_trace(skb);
	if (!xnet)
		return;
	ipvs_reset(skb);
	secpath_reset(skb);
	skb_orphan(skb);
	skb->mark = 0;
}
static int xfrmi_rcv_cb(struct sk_buff *skb, int err)
{
	struct pcpu_sw_netstats *tstats;
	struct xfrm_mode *inner_mode;
	struct net_device *dev;
	struct xfrm_state *x;
	struct xfrm_if *xi;
	bool xnet;
	if (err && !secpath_exists(skb))
		return 0;
	x = xfrm_input_state(skb);
	xi = xfrmi_lookup(xs_net(x), x);
	if (!xi)
		return 1;
	dev = xi->dev;
	skb->dev = dev;
	if (err) {
		dev->stats.rx_errors++;
		dev->stats.rx_dropped++;
		return 0;
	}
	xnet = !net_eq(xi->net, dev_net(skb->dev));
	if (xnet) {
		inner_mode = x->inner_mode;
		if (x->sel.family == AF_UNSPEC) {
			inner_mode = xfrm_ip2inner_mode(x, XFRM_MODE_SKB_CB(skb)->protocol);
			if (inner_mode == NULL) {
				XFRM_INC_STATS(dev_net(skb->dev),
					       LINUX_MIB_XFRMINSTATEMODEERROR);
				return -EINVAL;
			}
		}
		if (!xfrm_policy_check(NULL, XFRM_POLICY_IN, skb,
				       inner_mode->afinfo->family))
			return -EPERM;
	}
	xfrmi_scrub_packet(skb, xnet);
	tstats = this_cpu_ptr(dev->tstats);
	u64_stats_update_begin(&tstats->syncp);
	tstats->rx_packets++;
	tstats->rx_bytes += skb->len;
	u64_stats_update_end(&tstats->syncp);
	return 0;
}
static int
xfrmi_xmit2(struct sk_buff *skb, struct net_device *dev, struct flowi *fl)
{
	struct xfrm_if *xi = netdev_priv(dev);
	struct net_device_stats *stats = &xi->dev->stats;
	struct dst_entry *dst = skb_dst(skb);
	unsigned int length = skb->len;
	struct net_device *tdev;
	struct xfrm_state *x;
	int err = -1;
	int mtu;
	if (!dst)
		goto tx_err_link_failure;
	dst_hold(dst);
	dst = xfrm_lookup_with_ifid(xi->net, dst, fl, NULL, 0, xi->p.if_id);
	if (IS_ERR(dst)) {
		err = PTR_ERR(dst);
		dst = NULL;
		goto tx_err_link_failure;
	}
	x = dst->xfrm;
	if (!x)
		goto tx_err_link_failure;
	if (x->if_id != xi->p.if_id)
		goto tx_err_link_failure;
	tdev = dst->dev;
	if (tdev == dev) {
		stats->collisions++;
		net_warn_ratelimited("%s: Local routing loop detected!\n",
				     xi->p.name);
		goto tx_err_dst_release;
	}
	mtu = dst_mtu(dst);
	if (!skb->ignore_df && skb->len > mtu) {
		skb_dst_update_pmtu(skb, mtu);
		if (skb->protocol == htons(ETH_P_IPV6)) {
			if (mtu < IPV6_MIN_MTU)
				mtu = IPV6_MIN_MTU;
			icmpv6_send(skb, ICMPV6_PKT_TOOBIG, 0, mtu);
		} else {
			icmp_send(skb, ICMP_DEST_UNREACH, ICMP_FRAG_NEEDED,
				  htonl(mtu));
		}
		dst_release(dst);
		return -EMSGSIZE;
	}
	xfrmi_scrub_packet(skb, !net_eq(xi->net, dev_net(dev)));
	skb_dst_set(skb, dst);
	skb->dev = tdev;
	err = dst_output(xi->net, skb->sk, skb);
	if (net_xmit_eval(err) == 0) {
		struct pcpu_sw_netstats *tstats = this_cpu_ptr(dev->tstats);
		u64_stats_update_begin(&tstats->syncp);
		tstats->tx_bytes += length;
		tstats->tx_packets++;
		u64_stats_update_end(&tstats->syncp);
	} else {
		stats->tx_errors++;
		stats->tx_aborted_errors++;
	}
	return 0;
tx_err_link_failure:
	stats->tx_carrier_errors++;
	dst_link_failure(skb);
tx_err_dst_release:
	dst_release(dst);
	return err;
}
static netdev_tx_t xfrmi_xmit(struct sk_buff *skb, struct net_device *dev)
{
	struct xfrm_if *xi = netdev_priv(dev);
	struct net_device_stats *stats = &xi->dev->stats;
	struct flowi fl;
	int ret;
	memset(&fl, 0, sizeof(fl));
	switch (skb->protocol) {
	case htons(ETH_P_IPV6):
		xfrm_decode_session(skb, &fl, AF_INET6);
		memset(IP6CB(skb), 0, sizeof(*IP6CB(skb)));
		break;
	case htons(ETH_P_IP):
		xfrm_decode_session(skb, &fl, AF_INET);
		memset(IPCB(skb), 0, sizeof(*IPCB(skb)));
		break;
	default:
		goto tx_err;
	}
	fl.flowi_oif = xi->phydev->ifindex;
	ret = xfrmi_xmit2(skb, dev, &fl);
	if (ret < 0)
		goto tx_err;
	return NETDEV_TX_OK;
tx_err:
	stats->tx_errors++;
	stats->tx_dropped++;
	kfree_skb(skb);
	return NETDEV_TX_OK;
}
static int xfrmi4_err(struct sk_buff *skb, u32 info)
{
	const struct iphdr *iph = (const struct iphdr *)skb->data;
	struct net *net = dev_net(skb->dev);
	int protocol = iph->protocol;
	struct ip_comp_hdr *ipch;
	struct ip_esp_hdr *esph;
	struct ip_auth_hdr *ah ;
	struct xfrm_state *x;
	struct xfrm_if *xi;
	__be32 spi;
	switch (protocol) {
	case IPPROTO_ESP:
		esph = (struct ip_esp_hdr *)(skb->data+(iph->ihl<<2));
		spi = esph->spi;
		break;
	case IPPROTO_AH:
		ah = (struct ip_auth_hdr *)(skb->data+(iph->ihl<<2));
		spi = ah->spi;
		break;
	case IPPROTO_COMP:
		ipch = (struct ip_comp_hdr *)(skb->data+(iph->ihl<<2));
		spi = htonl(ntohs(ipch->cpi));
		break;
	default:
		return 0;
	}
	switch (icmp_hdr(skb)->type) {
	case ICMP_DEST_UNREACH:
		if (icmp_hdr(skb)->code != ICMP_FRAG_NEEDED)
			return 0;
	case ICMP_REDIRECT:
		break;
	default:
		return 0;
	}
	x = xfrm_state_lookup(net, skb->mark, (const xfrm_address_t *)&iph->daddr,
			      spi, protocol, AF_INET);
	if (!x)
		return 0;
	xi = xfrmi_lookup(net, x);
	if (!xi) {
		xfrm_state_put(x);
		return -1;
	}
	if (icmp_hdr(skb)->type == ICMP_DEST_UNREACH)
		ipv4_update_pmtu(skb, net, info, 0, protocol);
	else
		ipv4_redirect(skb, net, 0, protocol);
	xfrm_state_put(x);
	return 0;
}
static int xfrmi6_err(struct sk_buff *skb, struct inet6_skb_parm *opt,
		    u8 type, u8 code, int offset, __be32 info)
{
	const struct ipv6hdr *iph = (const struct ipv6hdr *)skb->data;
	struct net *net = dev_net(skb->dev);
	int protocol = iph->nexthdr;
	struct ip_comp_hdr *ipch;
	struct ip_esp_hdr *esph;
	struct ip_auth_hdr *ah;
	struct xfrm_state *x;
	struct xfrm_if *xi;
	__be32 spi;
	switch (protocol) {
	case IPPROTO_ESP:
		esph = (struct ip_esp_hdr *)(skb->data + offset);
		spi = esph->spi;
		break;
	case IPPROTO_AH:
		ah = (struct ip_auth_hdr *)(skb->data + offset);
		spi = ah->spi;
		break;
	case IPPROTO_COMP:
		ipch = (struct ip_comp_hdr *)(skb->data + offset);
		spi = htonl(ntohs(ipch->cpi));
		break;
	default:
		return 0;
	}
	if (type != ICMPV6_PKT_TOOBIG &&
	    type != NDISC_REDIRECT)
		return 0;
	x = xfrm_state_lookup(net, skb->mark, (const xfrm_address_t *)&iph->daddr,
			      spi, protocol, AF_INET6);
	if (!x)
		return 0;
	xi = xfrmi_lookup(net, x);
	if (!xi) {
		xfrm_state_put(x);
		return -1;
	}
	if (type == NDISC_REDIRECT)
		ip6_redirect(skb, net, skb->dev->ifindex, 0,
			     sock_net_uid(net, NULL));
	else
		ip6_update_pmtu(skb, net, info, 0, 0, sock_net_uid(net, NULL));
	xfrm_state_put(x);
	return 0;
}
static int xfrmi_change(struct xfrm_if *xi, const struct xfrm_if_parms *p)
{
	if (xi->p.link != p->link)
		return -EINVAL;
	xi->p.if_id = p->if_id;
	return 0;
}
static int xfrmi_update(struct xfrm_if *xi, struct xfrm_if_parms *p)
{
	struct net *net = dev_net(xi->dev);
	struct xfrmi_net *xfrmn = net_generic(net, xfrmi_net_id);
	int err;
	xfrmi_unlink(xfrmn, xi);
	synchronize_net();
	err = xfrmi_change(xi, p);
	xfrmi_link(xfrmn, xi);
	netdev_state_change(xi->dev);
	return err;
}
static void xfrmi_get_stats64(struct net_device *dev,
			       struct rtnl_link_stats64 *s)
{
	int cpu;
	for_each_possible_cpu(cpu) {
		struct pcpu_sw_netstats *stats;
		struct pcpu_sw_netstats tmp;
		int start;
		stats = per_cpu_ptr(dev->tstats, cpu);
		do {
			start = u64_stats_fetch_begin_irq(&stats->syncp);
			tmp.rx_packets = stats->rx_packets;
			tmp.rx_bytes   = stats->rx_bytes;
			tmp.tx_packets = stats->tx_packets;
			tmp.tx_bytes   = stats->tx_bytes;
		} while (u64_stats_fetch_retry_irq(&stats->syncp, start));
		s->rx_packets += tmp.rx_packets;
		s->rx_bytes   += tmp.rx_bytes;
		s->tx_packets += tmp.tx_packets;
		s->tx_bytes   += tmp.tx_bytes;
	}
	s->rx_dropped = dev->stats.rx_dropped;
	s->tx_dropped = dev->stats.tx_dropped;
}
static int xfrmi_get_iflink(const struct net_device *dev)
{
	struct xfrm_if *xi = netdev_priv(dev);
	return xi->phydev->ifindex;
}
static const struct net_device_ops xfrmi_netdev_ops = {
	.ndo_init	= xfrmi_dev_init,
	.ndo_uninit	= xfrmi_dev_uninit,
	.ndo_start_xmit = xfrmi_xmit,
	.ndo_get_stats64 = xfrmi_get_stats64,
	.ndo_get_iflink = xfrmi_get_iflink,
};
static void xfrmi_dev_setup(struct net_device *dev)
{
	dev->netdev_ops 	= &xfrmi_netdev_ops;
	dev->type		= ARPHRD_NONE;
	dev->hard_header_len 	= ETH_HLEN;
	dev->min_header_len	= ETH_HLEN;
	dev->mtu		= ETH_DATA_LEN;
	dev->min_mtu		= ETH_MIN_MTU;
	dev->max_mtu		= ETH_DATA_LEN;
	dev->addr_len		= ETH_ALEN;
	dev->flags 		= IFF_NOARP;
	dev->needs_free_netdev	= true;
	dev->priv_destructor	= xfrmi_dev_free;
	netif_keep_dst(dev);
}
static int xfrmi_dev_init(struct net_device *dev)
{
	struct xfrm_if *xi = netdev_priv(dev);
	struct net_device *phydev = xi->phydev;
	int err;
	dev->tstats = netdev_alloc_pcpu_stats(struct pcpu_sw_netstats);
	if (!dev->tstats)
		return -ENOMEM;
	err = gro_cells_init(&xi->gro_cells, dev);
	if (err) {
		free_percpu(dev->tstats);
		return err;
	}
	dev->features |= NETIF_F_LLTX;
	dev->needed_headroom = phydev->needed_headroom;
	dev->needed_tailroom = phydev->needed_tailroom;
	if (is_zero_ether_addr(dev->dev_addr))
		eth_hw_addr_inherit(dev, phydev);
	if (is_zero_ether_addr(dev->broadcast))
		memcpy(dev->broadcast, phydev->broadcast, dev->addr_len);
	return 0;
}
static int xfrmi_validate(struct nlattr *tb[], struct nlattr *data[],
			 struct netlink_ext_ack *extack)
{
	return 0;
}
static void xfrmi_netlink_parms(struct nlattr *data[],
			       struct xfrm_if_parms *parms)
{
	memset(parms, 0, sizeof(*parms));
	if (!data)
		return;
	if (data[IFLA_XFRM_LINK])
		parms->link = nla_get_u32(data[IFLA_XFRM_LINK]);
	if (data[IFLA_XFRM_IF_ID])
		parms->if_id = nla_get_u32(data[IFLA_XFRM_IF_ID]);
}
static int xfrmi_newlink(struct net *src_net, struct net_device *dev,
			struct nlattr *tb[], struct nlattr *data[],
			struct netlink_ext_ack *extack)
{
	struct net *net = dev_net(dev);
	struct xfrm_if_parms *p;
	struct xfrm_if *xi;
	xi = netdev_priv(dev);
	p = &xi->p;
	xfrmi_netlink_parms(data, p);
	if (!tb[IFLA_IFNAME])
		return -EINVAL;
	nla_strlcpy(p->name, tb[IFLA_IFNAME], IFNAMSIZ);
	xi = xfrmi_locate(net, p, 1);
	return PTR_ERR_OR_ZERO(xi);
}
static void xfrmi_dellink(struct net_device *dev, struct list_head *head)
{
	unregister_netdevice_queue(dev, head);
}
static int xfrmi_changelink(struct net_device *dev, struct nlattr *tb[],
			   struct nlattr *data[],
			   struct netlink_ext_ack *extack)
{
	struct xfrm_if *xi = netdev_priv(dev);
	struct net *net = dev_net(dev);
	xfrmi_netlink_parms(data, &xi->p);
	xi = xfrmi_locate(net, &xi->p, 0);
	if (IS_ERR_OR_NULL(xi)) {
		xi = netdev_priv(dev);
	} else {
		if (xi->dev != dev)
			return -EEXIST;
	}
	return xfrmi_update(xi, &xi->p);
}
static size_t xfrmi_get_size(const struct net_device *dev)
{
	return
		/* IFLA_XFRM_LINK */
		nla_total_size(4) +
		/* IFLA_XFRM_IF_ID */
		nla_total_size(4) +
		0;
}
static int xfrmi_fill_info(struct sk_buff *skb, const struct net_device *dev)
{
	struct xfrm_if *xi = netdev_priv(dev);
	struct xfrm_if_parms *parm = &xi->p;
	if (nla_put_u32(skb, IFLA_XFRM_LINK, parm->link) ||
	    nla_put_u32(skb, IFLA_XFRM_IF_ID, parm->if_id))
		goto nla_put_failure;
	return 0;
nla_put_failure:
	return -EMSGSIZE;
}
static struct net *xfrmi_get_link_net(const struct net_device *dev)
{
	struct xfrm_if *xi = netdev_priv(dev);
	return dev_net(xi->phydev);
}
static const struct nla_policy xfrmi_policy[IFLA_XFRM_MAX + 1] = {
	[IFLA_XFRM_LINK]	= { .type = NLA_U32 },
	[IFLA_XFRM_IF_ID]	= { .type = NLA_U32 },
};
static struct rtnl_link_ops xfrmi_link_ops __read_mostly = {
	.kind		= "xfrm",
	.maxtype	= IFLA_XFRM_MAX,
	.policy		= xfrmi_policy,
	.priv_size	= sizeof(struct xfrm_if),
	.setup		= xfrmi_dev_setup,
	.validate	= xfrmi_validate,
	.newlink	= xfrmi_newlink,
	.dellink	= xfrmi_dellink,
	.changelink	= xfrmi_changelink,
	.get_size	= xfrmi_get_size,
	.fill_info	= xfrmi_fill_info,
	.get_link_net	= xfrmi_get_link_net,
};
static void __net_exit xfrmi_destroy_interfaces(struct xfrmi_net *xfrmn)
{
	struct xfrm_if *xi;
	LIST_HEAD(list);
	xi = rtnl_dereference(xfrmn->xfrmi[0]);
	if (!xi)
		return;
	unregister_netdevice_queue(xi->dev, &list);
	unregister_netdevice_many(&list);
}
static int __net_init xfrmi_init_net(struct net *net)
{
	return 0;
}
static void __net_exit xfrmi_exit_net(struct net *net)
{
	struct xfrmi_net *xfrmn = net_generic(net, xfrmi_net_id);
	rtnl_lock();
	xfrmi_destroy_interfaces(xfrmn);
	rtnl_unlock();
}
static struct pernet_operations xfrmi_net_ops = {
	.init = xfrmi_init_net,
	.exit = xfrmi_exit_net,
	.id   = &xfrmi_net_id,
	.size = sizeof(struct xfrmi_net),
};
static struct xfrm6_protocol xfrmi_esp6_protocol __read_mostly = {
	.handler	=	xfrm6_rcv,
	.cb_handler	=	xfrmi_rcv_cb,
	.err_handler	=	xfrmi6_err,
	.priority	=	10,
};
static struct xfrm6_protocol xfrmi_ah6_protocol __read_mostly = {
	.handler	=	xfrm6_rcv,
	.cb_handler	=	xfrmi_rcv_cb,
	.err_handler	=	xfrmi6_err,
	.priority	=	10,
};
static struct xfrm6_protocol xfrmi_ipcomp6_protocol __read_mostly = {
	.handler	=	xfrm6_rcv,
	.cb_handler	=	xfrmi_rcv_cb,
	.err_handler	=	xfrmi6_err,
	.priority	=	10,
};
static struct xfrm4_protocol xfrmi_esp4_protocol __read_mostly = {
	.handler	=	xfrm4_rcv,
	.input_handler	=	xfrm_input,
	.cb_handler	=	xfrmi_rcv_cb,
	.err_handler	=	xfrmi4_err,
	.priority	=	10,
};
static struct xfrm4_protocol xfrmi_ah4_protocol __read_mostly = {
	.handler	=	xfrm4_rcv,
	.input_handler	=	xfrm_input,
	.cb_handler	=	xfrmi_rcv_cb,
	.err_handler	=	xfrmi4_err,
	.priority	=	10,
};
static struct xfrm4_protocol xfrmi_ipcomp4_protocol __read_mostly = {
	.handler	=	xfrm4_rcv,
	.input_handler	=	xfrm_input,
	.cb_handler	=	xfrmi_rcv_cb,
	.err_handler	=	xfrmi4_err,
	.priority	=	10,
};
static int __init xfrmi4_init(void)
{
	int err;
	err = xfrm4_protocol_register(&xfrmi_esp4_protocol, IPPROTO_ESP);
	if (err < 0)
		goto xfrm_proto_esp_failed;
	err = xfrm4_protocol_register(&xfrmi_ah4_protocol, IPPROTO_AH);
	if (err < 0)
		goto xfrm_proto_ah_failed;
	err = xfrm4_protocol_register(&xfrmi_ipcomp4_protocol, IPPROTO_COMP);
	if (err < 0)
		goto xfrm_proto_comp_failed;
	return 0;
xfrm_proto_comp_failed:
	xfrm4_protocol_deregister(&xfrmi_ah4_protocol, IPPROTO_AH);
xfrm_proto_ah_failed:
	xfrm4_protocol_deregister(&xfrmi_esp4_protocol, IPPROTO_ESP);
xfrm_proto_esp_failed:
	return err;
}
static void xfrmi4_fini(void)
{
	xfrm4_protocol_deregister(&xfrmi_ipcomp4_protocol, IPPROTO_COMP);
	xfrm4_protocol_deregister(&xfrmi_ah4_protocol, IPPROTO_AH);
	xfrm4_protocol_deregister(&xfrmi_esp4_protocol, IPPROTO_ESP);
}
static int __init xfrmi6_init(void)
{
	int err;
	err = xfrm6_protocol_register(&xfrmi_esp6_protocol, IPPROTO_ESP);
	if (err < 0)
		goto xfrm_proto_esp_failed;
	err = xfrm6_protocol_register(&xfrmi_ah6_protocol, IPPROTO_AH);
	if (err < 0)
		goto xfrm_proto_ah_failed;
	err = xfrm6_protocol_register(&xfrmi_ipcomp6_protocol, IPPROTO_COMP);
	if (err < 0)
		goto xfrm_proto_comp_failed;
	return 0;
xfrm_proto_comp_failed:
	xfrm6_protocol_deregister(&xfrmi_ah6_protocol, IPPROTO_AH);
xfrm_proto_ah_failed:
	xfrm6_protocol_deregister(&xfrmi_esp6_protocol, IPPROTO_ESP);
xfrm_proto_esp_failed:
	return err;
}
static void xfrmi6_fini(void)
{
	xfrm6_protocol_deregister(&xfrmi_ipcomp6_protocol, IPPROTO_COMP);
	xfrm6_protocol_deregister(&xfrmi_ah6_protocol, IPPROTO_AH);
	xfrm6_protocol_deregister(&xfrmi_esp6_protocol, IPPROTO_ESP);
}
static const struct xfrm_if_cb xfrm_if_cb = {
	.decode_session =	xfrmi_decode_session,
};
static int __init xfrmi_init(void)
{
	const char *msg;
	int err;
	pr_info("IPsec XFRM device driver\n");
	msg = "tunnel device";
	err = register_pernet_device(&xfrmi_net_ops);
	if (err < 0)
		goto pernet_dev_failed;
	msg = "xfrm4 protocols";
	err = xfrmi4_init();
	if (err < 0)
		goto xfrmi4_failed;
	msg = "xfrm6 protocols";
	err = xfrmi6_init();
	if (err < 0)
		goto xfrmi6_failed;
	msg = "netlink interface";
	err = rtnl_link_register(&xfrmi_link_ops);
	if (err < 0)
		goto rtnl_link_failed;
	xfrm_if_register_cb(&xfrm_if_cb);
	return err;
rtnl_link_failed:
	xfrmi6_fini();
xfrmi6_failed:
	xfrmi4_fini();
xfrmi4_failed:
	unregister_pernet_device(&xfrmi_net_ops);
pernet_dev_failed:
	pr_err("xfrmi init: failed to register %s\n", msg);
	return err;
}
static void __exit xfrmi_fini(void)
{
	xfrm_if_unregister_cb();
	rtnl_link_unregister(&xfrmi_link_ops);
	xfrmi4_fini();
	xfrmi6_fini();
	unregister_pernet_device(&xfrmi_net_ops);
}
module_init(xfrmi_init);
module_exit(xfrmi_fini);
MODULE_LICENSE("GPL");
MODULE_ALIAS_RTNL_LINK("xfrm");
MODULE_ALIAS_NETDEV("xfrm0");
MODULE_AUTHOR("Steffen Klassert");
MODULE_DESCRIPTION("XFRM virtual interface");
#endif /* w64k */