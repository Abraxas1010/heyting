namespace HeytingLean
namespace Crypto
namespace ZK

/--
Lightweight interface that any ZK backend can implement. It is intentionally
generic and compiled from the existing BoolLens VM in this codebase.
-/
structure Backend (F : Type) where
  Sys      : Type
  Assign   : Type
  compile  : ∀ {n}, HeytingLean.Crypto.Prog.Program n → Sys
  satisfies : Sys → Assign → Prop
  public   : Sys → Assign → List F

/--
Backend laws placeholder. Concrete backends can specialise this to state
soundness/completeness w.r.t. the BoolLens canonical run. We keep it as a
`Prop` container to avoid pulling additional dependencies into this module.
-/
structure Laws {F : Type} (B : Backend F) : Prop where
  sound : True
  complete : True

end ZK
end Crypto
end HeytingLean

