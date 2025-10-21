import Mathlib.Data.Rat.Init
import Mathlib.Tactic

namespace HeytingLean
namespace Crypto
namespace ZK

/-- Embed booleans into `ℚ` via `0/1`. -/
def boolToRat : Bool → ℚ
  | true => 1
  | false => 0

@[simp] lemma boolToRat_true : boolToRat true = 1 := rfl

@[simp] lemma boolToRat_false : boolToRat false = 0 := rfl

@[simp] lemma boolToRat_mul_self (b : Bool) :
    boolToRat b * boolToRat b = boolToRat b := by
  cases b <;> norm_num [boolToRat]

@[simp] lemma boolToRat_sq_sub (b : Bool) :
    boolToRat b * (boolToRat b - 1) = 0 := by
  cases b <;> norm_num [boolToRat]

end ZK
end Crypto
end HeytingLean
