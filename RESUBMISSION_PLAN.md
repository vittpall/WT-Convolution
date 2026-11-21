# WTHaar-Net — Plan Check & Novelty Ideas for Resubmission

*Model check pass (Claude, 2026-11-16). Companion to the experiment sweep in `CIFAR10/exp/`
(PBS jobs 205518–205530 and N1 jobs 205705–205708, all finished; see §6–7). Paper source: `../WT-Convolution_Paper/DSP2027_WTHaarPaper_Arxiv/main.tex`.*

---

## 1. Why ECCV rejected it (compressed)

All three reviewers converged on the same three points, and all three survive the rebuttal:

1. **No consistent win.** 91.25 → 91.28 on CIFAR is noise; on Tiny-ImageNet the matched-protocol
   logs in this repo show HT *beating* Haar (74.11 vs ~71 @1-best, 90 ep, lr 0.1), i.e. the
   paper's Table 1 ranking does not reproduce under a fair protocol.
2. **"Why quantum at all?"** A 4-qubit, depth-8 circuit is trivially simulable; the network never
   actually consumes quantum outputs; sign ambiguity is waved away.
3. **Rigor gaps.** No seeds/significance, no pseudocode, no code release, naming inconsistencies,
   no larger circuits even in simulation (Wz7Z explicitly said ≤15 qubits is easily simulable —
   and left this "substantial concern unresolved").

Any resubmission must fix **all three**, not just add experiments.

---

## 2. Plan check: what the current sweep does and does not cover

### Covered by the queued CIFAR sweep (E1 + E3)
- ✅ 5 seeds × {ResNet-20, HT, Haar} × {1,2,3} paths under one protocol; Welch t-tests
  (`aggregate.py`). Answers "is 0.03 meaningful" either way.
- ✅ Confound isolation: the old comparison differed in *three* ways (transform, residual flag,
  HT scaling convention). The sweep varies exactly one factor at a time
  (`ht_p3_res`/`haar_p3_res`, `htpan_p3`).
- ✅ E3 decomposition levels L∈{1,2,3,full} — first test of the paper's own locality claim.
- ✅ Final-epoch reporting (removes the select-on-test-set flaw).

### NOT yet covered — gaps in the current plan
| Gap | Reviewer it answers | Status |
|---|---|---|
| Tiny-ImageNet rerun, unified script, 3 seeds | Lfqh, fC4E (headline result must reproduce) | not started; wait for E3 to pick L |
| Block-Haar / qubit-scaling curve (E2) | Wz7Z's #1 unresolved concern | not started |
| Quantum circuits beyond 4 qubits, simulated 6–12 qubits | Wz7Z explicitly | not started (Q1) |
| Network inference on *emulated quantum outputs* (shots × noise × sign recovery) | Lfqh "where is the quantum in the pipeline" | not started (Q3) |
| Principled sign recovery (Hadamard test), not "spatial coherence" | all three | not started (Q2) |
| Wall-clock latency (MACs alone convinced nobody) | Wz7Z | not started |
| One modern architecture | Wz7Z | not started (E5) |
| Pseudocode + architecture figure + code release | fC4E | writing task |

**Risk flagged:** the sweep decides the *headline* (user decision: accuracy stays headline iff Haar
wins). But even in the best case, CIFAR wins of ~0.1–0.3 pts will not carry a paper alone. The
novelty below is what makes it publishable regardless of which way E1 lands.

---

## 3. Novel ideas (ranked by payoff ÷ effort)

### N1. Learnable orthogonal wavelets via Givens rotations — *tested: no accuracy gain (see §7)*
The Haar butterfly `[1 1; 1 −1]/√2` is the θ=π/4 case of a 2×2 rotation
`R(θ) = [cosθ sinθ; sinθ −cosθ]`. Make **θ a trainable parameter per level (or per subband)**:

- **Classically**: a *learnable orthogonal wavelet* layer. Haar is the init; the network tunes its
  own filter bank. Orthogonality is preserved by construction (no penalty terms), so the inverse
  stays exact and the soft-thresholding = wavelet-shrinkage interpretation survives. Cost: +m
  scalars per transform — essentially free.
- **Quantumly**: R(θ) *is* an RY-type single-qubit gate. The transform circuit becomes a
  **variational quantum circuit whose optimal parameters are trained classically** and provably
  equal to the classical layer. This single move answers "why quantum / what is quantum here":
  the circuit is no longer a fixed re-implementation of a classical op — it is the *same
  parameterized family* in both worlds, with gradients flowing through the classical twin.
- Differentiates from both HT-Net (fixed transform) and LDW (learned but non-orthogonal,
  no quantum realization). This is the cleanest "contribution beyond prior work" statement.
- Implementation: ~20 lines on top of `exp/layers.py` (build the matrix from θ's each forward
  pass, or cache + re-build on update). **Do this immediately after E1 results land.**

### N2. Quantum-in-the-loop inference curve (Q3, upgraded)
Don't just validate one 4×4 patch. Take the *trained* CIFAR model, replace the forward transform
at inference with a simulated circuit including (a) finite shots, (b) an IBM-calibrated noise
model, (c) N3 sign recovery, and plot **test accuracy vs shots (10³–10⁵) × block size (4–32 px ⇒
4–10 qubits)**. One figure converts "illustrative" into "quantified cost of going quantum" and
covers Wz7Z's simulability demand (up to 10–12 qubits) without hardware time.

### N3. Sign recovery by a Wᵀ1 offset (corrected, implemented, simulated)
*Correction:* the first version of this idea (adding a constant DC offset) is wrong — every Haar
detail row sums to zero, so a constant only shifts the first coefficient. The version that works:
encode `x' = x + c·g` with `g = Wᵀ1`. Since W is orthogonal, `W x' = W x + c·1`, and
`c = ‖x‖₂ ≥ max_j |(Wx)_j|` is a classically known bound, so every measured amplitude is
non-negative and `sqrt(p_j)` is the signed coefficient plus a known constant. No extra qubits,
one classical subtraction. `QCHT/sign_recovery.py`, finite shots, CIFAR patches (relative L2 error
of the 2D coefficients; naive unsigned readout is 1.24 / 1.41, i.e. useless):

| shots | 4×4 (4 qubits) | 8×8 (6 qubits) |
|---|---|---|
| 10³ | 0.242 | 1.032 |
| 10⁴ | 0.079 | 0.313 |
| 10⁵ | 0.025 | 0.100 |
| 10⁶ | 0.008 | 0.032 |

Error ∝ 1/√shots; going from 4 to 6 qubits costs ~16× more shots at equal error. Caveat to state:
sign-wrong % stays high (4–15%) because near-zero coefficients flip; the relative error is the
meaningful metric. Ideal statevector probabilities + multinomial sampling; no gate noise yet.

### N4. Subband-structured parameters (Haar-specific design, E4)
Share scaling/threshold within each subband (LL, LH, HL, HH per level) instead of per-pixel maps:
parameters drop from O(N²) to O(log N), the layer becomes resolution-agnostic (no more pad-to-256
hack on Tiny-ImageNet — a quiet but real flaw in the current pipeline), and thresholds become
interpretable as per-scale denoising strengths. HT has no analogue of this — it *structurally*
differentiates Haar from Hadamard, which is exactly what reviewer Lfqh asked for.

### N5. Haar token mixer in a modern backbone (E5, sharpened)
Rather than yet another ResNet: drop the layer into a ViT/Mixer as the **token-mixing step**
(FNet does this with Fourier; WaveMix with wavelets, but neither is orthogonal-invertible with
learned shrinkage, and neither has a quantum twin). One experiment on DeiT-Tiny/CIFAR or
Tiny-ImageNet buys "modern architecture" + a second novelty claim at modest cost. If compute is
tight, MobileNetV2 at native 64×64 is the cheaper fallback.

### N6. Theory paragraph that costs nothing
Soft-thresholding in an orthonormal wavelet basis is the proximal operator of the ℓ₁ norm ⇒ each
path computes one step of learned wavelet shrinkage (Donoho–Johnstone); energy compaction of Haar
on piecewise-smooth images explains why fewer coefficients survive thresholding than in the
Hadamard basis. Three sentences + one citation each; grounds the robustness experiments (and
predicts the observed blur advantage) instead of describing them.

### N7. Honest quantum resource table
Encoding O(N) + readout O(N/ε²) shots vs classical O(N) FHT — state it, then scope the claim:
the contribution is a *quantum-compatible* (and with N1, *variational*) layer family, with the
crossover analysis left explicit. Reviewers punished the oversell, not the idea.

---

## 4. Decision tree after E1/E3 (jobs 205518–205530)

- **Haar ≥ HT (p<0.05) at any path count or level** → keep accuracy headline; N1 becomes the
  method's generalization ("Haar is the best fixed point of a learnable family"); run
  Tiny-ImageNet with the winning L; then N2/N3.
- **Haar ≈ HT** → *(this branch happened; N1 was then tested and also gave no gain, see §7)*.
- **Haar < HT on CIFAR but wins on Tiny-ImageNet rerun** → resolution-dependent story
  ("locality pays at scale"), supported by E3 levels and N4.

**Venue:** target is **DSP 2027** (IEEE International Conference on Digital Signal Processing,
paper folder `DSP2027_WTHaarPaper_Arxiv`). The block-transform / shot-cost framing (§9) fits a
signal-processing venue; ECCV is only the venue that rejected v1.

## 5. Immediate order of work
1. Wait out E1/E3 (queued) → `python CIFAR10/exp/aggregate.py`.
2. Implement N1 (learnable θ) in `exp/` — small diff, submit same 5-seed protocol.
3. N3 lemma + rewrite of the hardware section numbers (no compute needed).
4. Q1 generalized circuit (Fijany–Williams) + simulated 6–12 qubit validation table.
5. N2 inference-under-emulation figure.
6. Tiny-ImageNet unified rerun (3 seeds, best L, N4 variant).
7. N5 if budget remains; writing fixes throughout (pseudocode, naming, captions, code release).

---
## 6. Findings so far (2026-11-17)

**Circuit (Q1, done): `QCHT/haar_circuit.py`, `level_cost.py`.** n-qubit Haar circuit = per level
(H on lsb, then qubit shift), controlled on the already-finished top qubits being |0⟩. Verified
against the classical matrix to ≤3e-14 (1D up to 64 amplitudes; 2D 4×4 and 8×8).
**The paper's claim "constant O(1) depth, same Hadamard count as WHT" is false for full-depth Haar.**
Transpiled 1D cost (CX / depth, all-to-all, basis cx/rz/sx/x), HT is depth 3 at every size:

| N | L=1 | L=2 | L=3 | full |
|---|---|---|---|---|
| 8 | 0/3 | 8/21 | 14/36 | 14/36 |
| 16 | 0/3 | 15/37 | 61/125 | 74/150 |
| 32 | 0/3 | 22/53 | 110/214 | 267/540 |
| 64 | 0/3 | 29/69 | 159/305 | 714/1471 |

**This links to the CIFAR levels result (E3):** L=1 / L=2 / full give 90.90 / 90.92 / 90.87 final
accuracy (no significant difference), while the circuit cost drops from 714 CX to 0 / 29.
Proposed story: *a level-limited Haar layer is as accurate as full Haar and is the quantum-friendly
choice (constant depth at L=1, linear at L=2).* Caveat: at L=1 the circuit is as cheap as HT, so the
quantum argument is "no extra cost vs HT", not an advantage; the accuracy comparison to HT
(90.90 vs 90.48 orthonormal, but 91.05 for Pan's scaling) is what has to carry it.

**CIFAR sweep (E1/E3), 5 seeds, final-epoch:** Haar beats orthonormal HT at 3 paths by +0.39
(p=0.011, not multiple-comparison-corrected) but not the published-scaling HT (91.05 vs 90.87,
p=0.40). ResNet-20 91.64.

---
## 7. N1 result (2026-11-18): null on accuracy

5 seeds, 3 paths, final-epoch test accuracy (vs published-scaling HT 91.05 ± 0.41, fixed Haar+res 91.16 ± 0.31):

| Variant | Mean ± std | vs `htpan_p3` | vs fixed Haar (same residual) |
|---|---|---|---|
| `lhaar_p3` (angle/level) | 91.21 ± 0.30 | +0.16, p=0.49 | +0.34, p=0.065 |
| `lhaar_p3_res` | 91.09 ± 0.20 | +0.04, p=0.87 | −0.07, p=0.68 |
| `lhaarpos_p3` (angle/butterfly) | 91.13 ± 0.22 | +0.08, p=0.73 | +0.26, p=0.079 |
| `lhaarpos_p3_res` | 91.28 ± 0.12 | +0.23, p=0.29 | +0.12, p=0.46 |

No variant is significantly better than HT or fixed Haar. Learned angles stay near 45°: median
|θ−45°| is 0.8–2.3° with the residual, 1–3.8° without; means are larger (1.3–8°) because a minority of
angles move far (max 25–86°). Interpretation: on CIFAR-10 Haar is already near what this layer wants.
**Circuit caveat:** the per-level angle is a cheap RY(2θ)·Z in place of H (same CX count); the
per-butterfly angle is a multiplexed rotation (up to ~2^(k−1) CX), so it is a classical ablation only.

**Consequence:** no accuracy headline on CIFAR-10 (Haar ≈ learnable Haar ≈ HT-Pan, all ~0.5 below
ResNet-20). Remaining credible contributions: corrected/scalable circuit with level-limited Haar
(§6), sign recovery with shot-count curves, statistical rigor, and emulated-quantum inference (N2).
Open decision: spend GPU on a Tiny-ImageNet rerun (likely null) or reposition around
efficiency + hardware-aware design.

---
## 8. Running now (2026-11-18): Mini-ImageNet rerun + quantum-emulated inference

**Dataset correction:** the "Tiny-ImageNet" scripts load `timm/mini-imagenet` (100 classes, 224 px,
train 50k / val 10k / test 5k), not Tiny-ImageNet (200 classes, 64 px). The paper must say Mini-ImageNet.
**Unified rerun:** `TinyImageNet/unified/train.py`, one protocol, ResNet-50 (torchvision) / HT / Haar /
Haar-L2, 3 seeds, model selected on *validation*, reported on *test*. Params match the paper (23.71M / 20.78M).
Jobs 205893–205905.

**Quantum emulation:** `CIFAR10/exp/emulate.py` (job 205906). Every forward and inverse transform of the
trained CIFAR models is replaced by: block-wise amplitude encoding with the Wᵀ1 sign offset,
Binomial shot sampling (equal shot budget per feature map for all models), and optionally gate noise.
Gate noise uses the *exact* Aer channel (per-gate depolarizing, e1=2.5e-4, e2=3e-3, transpiled circuit)
for blocks ≤6 qubits, and a global-depolarizing fallback above that, which is accurate for HT (0 CX, checked
against Aer: 0.38% vs 0.39%) but **optimistic** for full-depth Haar. Mitigation: standard global-depolarizing.
Checks: exact limit reproduces logits to 1e-5; channels reproduce the ideal transform to 1e-16 at zero noise.

**Key structural point:** an L-level Haar is block-diagonal, i.e. it runs as independent 2^L×2^L patches on
2L qubits (L=2 ⇒ the paper's 4×4 / 4-qubit circuit, now actually used by the trained network). HT needs one
log₂(H·W)-qubit circuit per feature map. Shot noise per coefficient grows with circuit size, so at equal total
shots the patch-wise Haar should be much less noisy (≈√(N/N_b) smaller error) — the emulation tests this.

**Caveats found today:** (1) cost tables assume plain SWAPs are elided by qubit relabeling (transpiler opt-level 2),
which is legitimate but must be stated; (2) per-layer gate-noise residual after mitigation: 0.07% (L1, 2 qubits),
0.47% (L2, 4 qubits), 4–5% (L3 / 8×8, 6 qubits, 28 CX), so ≥6-qubit Haar blocks are already marginal on
current hardware; (3) amplitude encoding (state preparation) is not costed — with `initialize` its noise
dominates everything, and the paper must state that encoding is excluded from the comparison.

## 9. Final results (2026-11-19)

**Mini-ImageNet (3 seeds, final-epoch test top-1, model selected on validation)**

| Model | test top-1 |
|---|---|
| ResNet-50 | 73.01 ± 1.19 |
| Haar-L2 | 72.71 ± 1.22 |
| HT | 71.81 ± 1.33 |
| Haar (full) | 71.44 ± 0.43 |

All pairwise Welch p in 0.3–0.8: no significant differences. Haar-L2 matches ResNet-50 with 12% fewer parameters.

**Quantum emulation (CIFAR-10, 5 seeds, gate noise e1=2.5e-4, e2=3e-3), accuracy vs shots per feature map**

| Model | 1e4 | 1e5 | 1e6 | 1e7 | noiseless |
|---|---|---|---|---|---|
| Haar-L1 (2q blocks, 0 CX) | 89.2 | 90.8 | 90.9 | 90.9 | 90.9 |
| Haar-L2 (4q blocks, 2 CX) | 80.5 | 90.5 | 90.9 | 90.9 | 90.9 |
| Haar-L3 (6q blocks, 28 CX) | 17.0 | 87.3 | 89.9 | 89.9 | 90.8 |
| Haar full (10q) | 9.9 | 10.0 | 11.2 | 80.9 | 90.9 |
| HT | 10.0 | 11.7 | 83.8 | 90.2 | 90.5 |
| HT (Pan) | 10.0 | 11.5 | 83.7 | 90.6 | 91.1 |

Block Haar (L1/L2) needs ~100x fewer shots than HT / full Haar. Pending: block-HT control
(`ht_p3_L1/L2/L3`, job 206851) to test whether the advantage is from Haar or from blocking.
