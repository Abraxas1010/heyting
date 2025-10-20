import HeytingLean.ProofWidgets.LoFViz.State
import HeytingLean.ProofWidgets.LoFViz.Kernel

namespace HeytingLean
namespace ProofWidgets
namespace LoFViz

lemma kernelSummary_ne_empty (s : State) :
    (KernelData.fromState s).summary ≠ "" := by
  simp [KernelData.fromState]

lemma certificates_have_default_message (s : State) :
    (KernelData.certificates (KernelData.fromState s)).messages ≠ #[] := by
  simp [KernelData.certificates]

lemma notes_nonempty (s : State) :
    (KernelData.fromState s).notes ≠ #[] := by
  simp [KernelData.fromState, KernelData.notes]

lemma fiberNotes_length (s : State) :
    (KernelData.fiberNotes (KernelData.fromState s)).length = 4 := by
  simp [KernelData.fromState, KernelData.fiberNotes]

end LoFViz
end ProofWidgets
end HeytingLean
