"""Staged experiment suite definitions matching reports/planned-runs.md.

Executes preregistered stages (4a, 4b, 4c, 4d, 4e) cleanly, skipping already
evaluated runs unless ``--force`` is passed, with optional live dashboard thread.
"""

from __future__ import annotations

import argparse
import json
from typing import List

from pnn_sort.config import DEFAULT_SEEDS, RESULTS_DIR, TrainConfig
from pnn_sort.dashboard import start_dashboard_thread
from pnn_sort.pipeline import run_experiment


def build_stage_configs(stage: str) -> List[TrainConfig]:
    cfgs: List[TrainConfig] = []

    if stage == "4a_controls":
        for n in (8, 16):
            cfgs.extend([
                TrainConfig(run_id=f"ctrl_identity_n{n}_s42", model_name="identity_control", n=n, epochs=0),
                TrainConfig(run_id=f"ctrl_linear_n{n}_s42", model_name="linear_control", n=n, epochs=0),
                TrainConfig(run_id=f"ctrl_rand_polysort_n{n}_s42", model_name="polysort_shared", n=n, epochs=0, model_kwargs={"init": "random"}),
                TrainConfig(run_id=f"ctrl_rand_pinet_n{n}_s42", model_name="pinet_ncp_skip", n=n, epochs=0),
                TrainConfig(run_id=f"ana_cheon_f1_L3_n{n}_s42", model_name="analytical_cheon_f1", n=n, epochs=0, model_kwargs={"num_layers": 3}),
                TrainConfig(run_id=f"ana_cheon_f2_L2_n{n}_s42", model_name="analytical_cheon_f2", n=n, epochs=0, model_kwargs={"num_layers": 2}),
                TrainConfig(run_id=f"ana_cheon_fg_n{n}_s42", model_name="analytical_cheon_fg", n=n, epochs=0),
                TrainConfig(run_id=f"ana_remez_L3_n{n}_s42", model_name="analytical_remez", n=n, epochs=0, model_kwargs={"num_layers": 3, "num_terms": 1, "delta": 0.10}),
                TrainConfig(run_id=f"ana_remez_L3_m2_n{n}_s42", model_name="analytical_remez", n=n, epochs=0, model_kwargs={"num_layers": 3, "num_terms": 2, "delta": 0.03}),
            ])

    elif stage == "4a_main":
        for n in (8, 16):
            cfgs.extend([
                TrainConfig(run_id=f"main_polysort_shared_n{n}_s42", model_name="polysort_shared", n=n, lr=3e-3, epochs=60),
                TrainConfig(run_id=f"main_polysort_shared_m2_n{n}_s42", model_name="polysort_shared_m2", n=n, lr=3e-3, epochs=60),
                TrainConfig(run_id=f"main_polysort_adaptive_n{n}_s42", model_name="polysort_adaptive", n=n, lr=3e-3, epochs=60),
                TrainConfig(run_id=f"main_polysort_adaptive_m2_n{n}_s42", model_name="polysort_adaptive_m2", n=n, lr=3e-3, epochs=60),
            ])

    elif stage == "4b":
        for n in (8, 16):
            cfgs.extend([
                TrainConfig(run_id=f"main_poly_pairwise_n{n}_s42", model_name="poly_pairwise_rank", n=n, lr=3e-3, epochs=60),
                TrainConfig(run_id=f"main_polysort_learnedwire_n{n}_s42", model_name="polysort_learned_wire", n=n, lr=2e-3, epochs=60),
                TrainConfig(run_id=f"main_deepsets_poly_n{n}_s42", model_name="deepsets_poly", n=n, lr=1e-3, epochs=60),
                TrainConfig(run_id=f"main_pinet_ncpskip_n{n}_s42", model_name="pinet_ncp_skip", n=n, lr=1e-3, epochs=60),
                TrainConfig(run_id=f"main_poly_mlp_n{n}_s42", model_name="poly_mlp", n=n, lr=1e-3, epochs=60),
            ])
        # Scaling study n=4 and n=32
        cfgs.extend([
            TrainConfig(run_id="scale_polysort_shared_n4_s42", model_name="polysort_shared", n=4, lr=3e-3, epochs=60),
            TrainConfig(run_id="scale_polysort_adaptive_n4_s42", model_name="polysort_adaptive", n=4, lr=3e-3, epochs=60),
            TrainConfig(run_id="scale_pinet_n4_s42", model_name="pinet_ncp_skip", n=4, lr=1e-3, epochs=60),
            TrainConfig(run_id="scale_poly_mlp_n4_s42", model_name="poly_mlp", n=4, lr=1e-3, epochs=60),
            TrainConfig(run_id="scale_deepsets_poly_n4_s42", model_name="deepsets_poly", n=4, lr=1e-3, epochs=60),
            TrainConfig(run_id="scale_polysort_shared_n32_s42", model_name="polysort_shared", n=32, lr=3e-3, epochs=40),
            TrainConfig(run_id="scale_polysort_adaptive_n32_s42", model_name="polysort_adaptive", n=32, lr=3e-3, epochs=40),
        ])

    elif stage == "4c":
        for n in (8, 16):
            cfgs.extend([
                TrainConfig(run_id=f"ctrl_shuffled_n{n}_s42", model_name="polysort_adaptive", n=n, lr=3e-3, epochs=30, shuffled_targets=True),
                TrainConfig(run_id=f"base_diffsort_logistic_n{n}_s42", model_name="diffsort_logistic", n=n, lr=2e-2, epochs=60),
                TrainConfig(run_id=f"base_diffsort_cauchy_n{n}_s42", model_name="diffsort_cauchy", n=n, lr=2e-2, epochs=60),
                TrainConfig(run_id=f"base_relu_mlp_n{n}_s42", model_name="relu_mlp", n=n, lr=1e-3, epochs=60),
                TrainConfig(run_id=f"base_set_transformer_n{n}_s42", model_name="set_transformer", n=n, lr=1e-3, epochs=60),
            ])

    elif stage == "4d":
        for n in (8, 16):
            # A1: Parameterization
            cfgs.extend([
                TrainConfig(run_id=f"abl_param_unconst_odd_n{n}_s42", model_name="polysort_custom", n=n, lr=3e-3, epochs=60,
                            model_kwargs={"shared": False, "num_layers": 3, "num_terms": 1, "mode": "unconstrained_odd_cheby", "init": "cheon"}),
                TrainConfig(run_id=f"abl_param_general_cheby_n{n}_s42", model_name="polysort_custom", n=n, lr=3e-3, epochs=60,
                            model_kwargs={"shared": False, "num_layers": 3, "num_terms": 1, "mode": "general_cheby", "init": "cheon"}),
                TrainConfig(run_id=f"abl_param_raw_monomial_n{n}_s42", model_name="polysort_custom", n=n, lr=3e-3, epochs=60,
                            model_kwargs={"shared": False, "num_layers": 3, "num_terms": 1, "mode": "raw_monomial", "init": "cheon"}),
            ])
            # A2: Iso-Degree Depth Separation (flat deg 9 vs comp L=2 d=3 deg 9 vs L=3 d=3 vs L=4 d=3)
            cfgs.extend([
                TrainConfig(run_id=f"abl_depth_L1_d9_n{n}_s42", model_name="polysort_custom", n=n, lr=3e-3, epochs=60,
                            model_kwargs={"shared": False, "num_layers": 1, "num_terms": 4, "mode": "fixed_point_cheby", "init": "cheon"}),
                TrainConfig(run_id=f"abl_depth_L2_d3_n{n}_s42", model_name="polysort_custom", n=n, lr=3e-3, epochs=60,
                            model_kwargs={"shared": False, "num_layers": 2, "num_terms": 1, "mode": "fixed_point_cheby", "init": "cheon"}),
                TrainConfig(run_id=f"abl_depth_L4_d3_n{n}_s42", model_name="polysort_custom", n=n, lr=2e-3, epochs=60,
                            model_kwargs={"shared": False, "num_layers": 4, "num_terms": 1, "mode": "fixed_point_cheby", "init": "cheon"}),
            ])
            # A3: Initialization
            cfgs.extend([
                TrainConfig(run_id=f"abl_init_identity_n{n}_s42", model_name="polysort_custom", n=n, lr=3e-3, epochs=60,
                            model_kwargs={"shared": False, "num_layers": 3, "num_terms": 1, "init": "identity"}),
                TrainConfig(run_id=f"abl_init_random_n{n}_s42", model_name="polysort_custom", n=n, lr=3e-3, epochs=60,
                            model_kwargs={"shared": False, "num_layers": 3, "num_terms": 1, "init": "random"}),
            ])
            # A4: Wiring Topology + Random-Wire Negative Control
            cfgs.extend([
                TrainConfig(run_id=f"abl_wire_oddeven_merge_n{n}_s42", model_name="polysort_custom", n=n, lr=3e-3, epochs=60,
                            model_kwargs={"shared": False, "wiring_type": "odd_even_merge", "num_layers": 3, "num_terms": 1}),
                TrainConfig(run_id=f"abl_wire_oddeven_trans_n{n}_s42", model_name="polysort_custom", n=n, lr=3e-3, epochs=60,
                            model_kwargs={"shared": False, "wiring_type": "odd_even_trans", "num_layers": 3, "num_terms": 1}),
                TrainConfig(run_id=f"abl_wire_random_fixed_n{n}_s42", model_name="polysort_custom", n=n, lr=3e-3, epochs=60,
                            model_kwargs={"shared": False, "wiring_type": "random_fixed", "num_layers": 3, "num_terms": 1, "wiring_seed": 777}),
            ])
            # A5: Loss Function
            cfgs.extend([
                TrainConfig(run_id=f"abl_loss_huber_n{n}_s42", model_name="polysort_adaptive", n=n, lr=3e-3, epochs=60, loss_name="huber"),
                TrainConfig(run_id=f"abl_loss_mse_mono_n{n}_s42", model_name="polysort_adaptive", n=n, lr=3e-3, epochs=60, loss_name="mse", mono_weight=0.1),
                TrainConfig(run_id=f"abl_loss_mse_mono_w1_n{n}_s42", model_name="polysort_adaptive", n=n, lr=3e-3, epochs=60, loss_name="mse", mono_weight=0.1, w1_weight=0.1),
            ])
            # A6: Non-Polynomial Scale-Normalization Boundary Ablation
            cfgs.append(
                TrainConfig(run_id=f"abl_scale_norm_n{n}_s42", model_name="polysort_scale_norm", n=n, lr=3e-3, epochs=60,
                            model_kwargs={"shared": False, "num_layers": 3, "num_terms": 1})
            )

    elif stage == "4e_seeds":
        # Seeds 123, 456, 789, 1024 (seed 42 already run in stages 4a-4c!)
        core_specs = [
            ("main_polysort_shared", "polysort_shared", 3e-3),
            ("main_polysort_adaptive", "polysort_adaptive", 3e-3),
            ("main_polysort_adaptive_m2", "polysort_adaptive_m2", 3e-3),
            ("main_poly_pairwise", "poly_pairwise_rank", 3e-3),
            ("main_deepsets_poly", "deepsets_poly", 1e-3),
            ("main_pinet_ncpskip", "pinet_ncp_skip", 1e-3),
            ("main_poly_mlp", "poly_mlp", 1e-3),
            ("base_diffsort_cauchy", "diffsort_cauchy", 2e-2),
            ("base_relu_mlp", "relu_mlp", 1e-3),
            ("base_set_transformer", "set_transformer", 1e-3),
        ]
        for n in (8, 16):
            for seed in DEFAULT_SEEDS[1:]:
                for prefix, mname, lr in core_specs:
                    cfgs.append(TrainConfig(run_id=f"{prefix}_n{n}_s{seed}", model_name=mname, n=n, seed=seed, lr=lr, epochs=60))

    elif stage == "4e_sens":
        # Hyperparameter & precision sensitivity sweep on n=8
        for lr in (3e-4, 1e-3, 1e-2):
            cfgs.append(TrainConfig(run_id=f"sens_polysort_lr_{lr:g}_n8_s42", model_name="polysort_adaptive", n=8, lr=lr, epochs=60))
            cfgs.append(TrainConfig(run_id=f"sens_pinet_lr_{lr:g}_n8_s42", model_name="pinet_ncp_skip", n=8, lr=lr, epochs=60))
        for wd in (0.0, 1e-2):
            cfgs.append(TrainConfig(run_id=f"sens_polysort_wd_{wd:g}_n8_s42", model_name="polysort_adaptive", n=8, lr=3e-3, weight_decay=wd, epochs=60))
        for bs in (128, 2048):
            cfgs.append(TrainConfig(run_id=f"sens_polysort_bs_{bs}_n8_s42", model_name="polysort_adaptive", n=8, lr=3e-3, batch_size=bs, epochs=60))
        # Float64 training precision verification on L=3 and L=4
        cfgs.append(TrainConfig(run_id="sens_polysort_L3_f64_n8_s42", model_name="polysort_adaptive", n=8, lr=3e-3, epochs=60, dtype="float64"))
        cfgs.append(TrainConfig(run_id="sens_polysort_L4_f64_n8_s42", model_name="polysort_custom", n=8, lr=2e-3, epochs=60, dtype="float64",
                                model_kwargs={"shared": False, "num_layers": 4, "num_terms": 1}))

    else:
        raise ValueError(f"Unknown stage: {stage}")

    return cfgs


def main() -> None:
    parser = argparse.ArgumentParser(description="Execute preregistered staged experiment runs.")
    parser.add_argument("--stage", required=True, choices=["4a_controls", "4a_main", "4b", "4c", "4d", "4e_seeds", "4e_sens"])
    parser.add_argument("--force", action="store_true", help="Re-run even if eval_report.json already exists")
    parser.add_argument("--dashboard-port", type=int, default=0, help="If >0, start live web monitor on this port")
    parser.add_argument("--only", default="", help="Comma-separated run_id substring filter")
    args = parser.parse_args()

    if args.dashboard_port > 0:
        start_dashboard_thread(port=args.dashboard_port)
        print(f"Live accuracy dashboard started at http://127.0.0.1:{args.dashboard_port}", flush=True)

    cfgs = build_stage_configs(args.stage)
    filters = [s.strip() for s in args.only.split(",") if s.strip()]
    if filters:
        cfgs = [c for c in cfgs if any(f in c.run_id for f in filters)]

    print(f"=== Running stage '{args.stage}' ({len(cfgs)} runs) ===", flush=True)
    for i, cfg in enumerate(cfgs, 1):
        eval_file = RESULTS_DIR / "runs" / cfg.run_id / "eval_report.json"
        if eval_file.exists() and not args.force:
            rep = json.loads(eval_file.read_text(encoding="utf-8"))
            t = rep["test"]
            print(f"[{i}/{len(cfgs)}] SKIP (done) {cfg.run_id}: MAE={t['mae']:.6f} Mono={t['mono_rate']*100:.2f}%", flush=True)
            continue

        # Zero-shot only needed on variable-length models (and skip on multi-seed repeats beyond s42 to keep eval fast)
        do_zs = cfg.seed == 42
        res = run_experiment(cfg, eval_zero_shot=do_zs, eval_float64=(cfg.seed == 42))
        t = res["eval"]["test"]
        tr = res["train"]
        print(
            f"[{i}/{len(cfgs)}] DONE {cfg.run_id} ({tr['train_seconds']:.1f}s) | "
            f"MAE={t['mae']:.6f} | RMSE={t['rmse']:.6f} | Mono={t['mono_rate']*100:.2f}% | "
            f"PairAcc={t['pair_acc']*100:.2f}% | Tau={t['kendall_tau']:.5f} | R_stab={res['eval']['r_stable']}",
            flush=True,
        )


if __name__ == "__main__":
    main()
