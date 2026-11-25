# Learnings from Stage 4E: Sensitivity Sweeps & 5-Seed Replication

**Author:** Libor "Burny" Burian  
**Date:** 2026-11-25  

1. **Zero `float32` vs. `float64` Optimization Drift Even at Degree 81:**
   - Training `PolySortNet-Adaptive` in full `float64` matched `float32` to **6 decimal places** on both $L=3$ (deg 27: `0.033254`) and $L=4$ (deg 81: `0.013222`). Fixed-point Chebyshev anchoring keeps all intermediate wire activations cleanly inside $[-1, 1]$, preventing polynomial gradient overflow in single precision.
2. **Dramatic Seed-Stability Gap Between Structured and Unstructured PNNs:**
   - Across 5 independent seeds, 36-parameter `PolySortNet-Adaptive (m=2)` had seed SD `5.76e-7` at $n=8$ (`CV = 0.008%`) — **$9{,}078\times$ lower variance** than `PiNet-NCP-Skip` (`SD = 5.23e-3`), **$5{,}082\times$ lower** than `PolyMLP` (`2.93e-3`), and **$374\times$ lower** than `SetTransformer` (`2.16e-4`). Low-dimensional structured polynomial landscapes behave nearly convexly around Cheon initialization!
3. **Why Running Hyperparameter Sensitivity on Baselines Matters (Scientific Integrity):**
   - Sweeping `lr` on `PiNet-NCP-Skip` revealed that `lr=0.01` drops its $n=8$ MAE from `0.0529 -> 0.0423` (improving it 20% over our default `lr=2e-3`). Auditing baseline LRs prevented us from overclaiming `PiNet`'s weakness while still confirming a massive **$5.85\times$** gap vs. `PolySortNet-Adaptive (m=2)`.
4. **Windows Atomic-Write File-Lock Engineering Fix:**
   - On run `177 / 178`, `os.replace(tmp_live, live_state_path)` hit `[WinError 5] Access is denied` when colliding with the live dashboard server reading `live_dashboard_state.json`. Adding a 5-retry `PermissionError` backoff in `trainer.py` resolved the race cleanly, and our atomic checkpointing resumed seamlessly at run 177 with zero lost compute.
5. **5-Seed OOD Victory Over `SetTransformer`:**
   - Averaged across all 5 seeds at $n=8$, 36-parameter strictly pure-polynomial `PolySortNet-Adaptive (m=2)` outperformed 96,577-parameter softmax `SetTransformer` on **Near-Ties $10^{-3}$** (`0.00659` vs `0.00665`), **50% Duplicates** (`0.00660` vs `0.00737`), and **Shift $+100$** (`0.00726` vs `99.50`, $13{,}714\times$) — showing that algebraic symmetry beats raw attention capacity outside the training distribution.
