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

@[simp] lemma boolToRat_and (x y : Bool) :
    boolToRat (x && y) = boolToRat x * boolToRat y := by
  cases x <;> cases y <;> norm_num [boolToRat]

lemma boolToRat_or (x y : Bool) :
    boolToRat (x || y) =
      boolToRat x + boolToRat y - boolToRat x * boolToRat y := by
  cases x <;> cases y <;> norm_num [boolToRat]

lemma boolToRat_imp (x y : Bool) :
    boolToRat ((! x) || y) =
      1 - boolToRat x + boolToRat x * boolToRat y := by
  cases x <;> cases y <;> norm_num [boolToRat]

@[simp] lemma boolToRat_not (x : Bool) :
    boolToRat (! x) = 1 - boolToRat x := by
  cases x <;> norm_num [boolToRat]

end ZK
end Crypto
end HeytingLean
