import Mathlib

namespace RepairCore

def moment (w a b : ℝ) (n : ℕ) : ℝ := w * a ^ n + (1 - w) * b ^ n

theorem variance_identity (w a b : ℝ) :
    moment w a b 2 - moment w a b 1 ^ 2 = w * (1 - w) * (a - b) ^ 2 := by
  simp only [moment]
  ring

theorem recurrence (w a b : ℝ) (n : ℕ) :
    moment w a b (n + 2) = (a + b) * moment w a b (n + 1) - a * b * moment w a b n := by
  simp only [moment, pow_succ]
  ring

theorem variance_positive (w a b : ℝ) (hw : 0 < w) (hw1 : w < 1) (hab : a ≠ b) :
    0 < moment w a b 2 - moment w a b 1 ^ 2 := by
  rw [variance_identity]
  exact mul_pos (mul_pos hw (sub_pos.mpr hw1)) (sq_pos_of_ne_zero (sub_ne_zero.mpr hab))

theorem sum_recovery (w a b : ℝ) (hw : 0 < w) (hw1 : w < 1) (hab : a ≠ b) :
    (moment w a b 3 - moment w a b 1 * moment w a b 2) /
      (moment w a b 2 - moment w a b 1 ^ 2) = a + b := by
  have hv : moment w a b 2 - moment w a b 1 ^ 2 ≠ 0 := ne_of_gt (variance_positive w a b hw hw1 hab)
  apply (div_eq_iff hv).2
  simp only [moment]
  ring

theorem product_recovery (w a b : ℝ) :
    (a + b) * moment w a b 1 - moment w a b 2 = a * b := by
  simp only [moment]
  ring

theorem unordered_from_sum_product (a b c d : ℝ) (hs : a + b = c + d)
    (hp : a * b = c * d) : (a = c ∧ b = d) ∨ (a = d ∧ b = c) := by
  have h : (a - c) * (a - d) = 0 := by
    calc
      (a - c) * (a - d) = a ^ 2 - a * (c + d) + c * d := by ring
      _ = a ^ 2 - a * (a + b) + a * b := by rw [← hs, ← hp]
      _ = 0 := by ring
  rcases mul_eq_zero.mp h with h | h
  · left
    constructor <;> linarith
  · right
    constructor <;> linarith

theorem recover_two_atom (w a b v c d : ℝ)
    (hw : 0 < w) (hw1 : w < 1) (hab : a ≠ b)
    (hv : 0 < v) (hv1 : v < 1) (hcd : c ≠ d)
    (h1 : moment w a b 1 = moment v c d 1)
    (h2 : moment w a b 2 = moment v c d 2)
    (h3 : moment w a b 3 = moment v c d 3) :
    (a = c ∧ b = d ∧ w = v) ∨ (a = d ∧ b = c ∧ w = 1 - v) := by
  have hs : a + b = c + d := by
    have h := sum_recovery w a b hw hw1 hab
    have h' := sum_recovery v c d hv hv1 hcd
    rw [h1, h2, h3] at h
    linarith
  have hp : a * b = c * d := by
    have h := product_recovery w a b
    have h' := product_recovery v c d
    rw [h1, h2, hs] at h
    linarith
  rcases unordered_from_sum_product a b c d hs hp with h | h
  · obtain ⟨hac, hbd⟩ := h
    have ht : (w - v) * (c - d) = 0 := by
      simp only [moment, pow_one] at h1
      rw [hac, hbd] at h1
      nlinarith [h1]
    have hwv : w = v := by
      rcases mul_eq_zero.mp ht with ht | ht
      · linarith
      · exact False.elim (hcd (sub_eq_zero.mp ht))
    exact Or.inl ⟨hac, hbd, hwv⟩
  · obtain ⟨had, hbc⟩ := h
    have ht : (w - (1-v)) * (d - c) = 0 := by
      simp only [moment, pow_one] at h1
      rw [had, hbc] at h1
      nlinarith [h1]
    have hwv : w = 1 - v := by
      rcases mul_eq_zero.mp ht with ht | ht
      · linarith
      · exact False.elim (hcd (sub_eq_zero.mp ht).symm)
    exact Or.inr ⟨had, hbc, hwv⟩

theorem distinct_weight_match (w₁ w₂ v₁ : ℝ)
    (h : w₁ ≠ w₂) (h11 : w₁ = v₁)
    (h21 : w₂ = v₁) : False := by
  exact h (h11.trans h21.symm)

theorem scalar_contrast (l lt t tt a aa : ℝ) (hl : l ≠ 0) (hlt : lt ≠ 0)
    (h : l * t - a = lt * tt - aa) :
    t = (lt / l) * tt + (a - aa) / l ∧ lt / l ≠ 0 := by
  constructor
  · field_simp
    nlinarith
  · exact div_ne_zero hlt hl

example : moment (1/3) 1 4 1 = 3 ∧ moment (1/3) 1 4 2 = 11 ∧
    moment (1/3) 1 4 3 = 43 := by
  norm_num [moment]

theorem repeated_weight_swap (a b : ℝ) (n : ℕ) :
    moment (1/2) a b n = moment (1/2) b a n := by
  simp only [moment]
  ring

example : moment (1/2) 1 4 1 = moment (1/2) 4 1 1 ∧
    (1 : ℝ) ≠ 4 := by
  constructor
  · exact repeated_weight_swap 1 4 1
  · norm_num

#print axioms variance_identity
#print axioms variance_positive
#print axioms sum_recovery
#print axioms product_recovery
#print axioms recover_two_atom
#print axioms distinct_weight_match
#print axioms scalar_contrast
#print axioms repeated_weight_swap
end RepairCore
