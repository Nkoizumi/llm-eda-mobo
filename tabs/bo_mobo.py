"""BO / MOBO tab — Bayesian Optimization on the joint MTL surrogate.

Single target → qLogExpectedImprovement.
Multi-target → qLogExpectedHypervolumeImprovement (MOBO).
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from tabs._shared import PLOT_THEME

try:
    import torch  # noqa: F401
    import botorch  # noqa: F401
    import gpytorch  # noqa: F401
    _BOTORCH_AVAILABLE = True
except ImportError:
    _BOTORCH_AVAILABLE = False


def render(*, target_cols: list[str]) -> None:
    st.markdown(
        '<p class="tab-header">🎯 Bayesian Optimization (BO / MOBO)</p>',
        unsafe_allow_html=True,
    )

    if not _BOTORCH_AVAILABLE:
        st.error(
            "BoTorch is not installed. Install with:\n\n"
            "```bash\npip install botorch gpytorch torch\n```\n"
            "(Already in requirements.txt — heavy dependency, install only "
            "if you intend to use BO/MOBO.)"
        )
        return

    mtl_bo = st.session_state.get("mtl_results")
    if mtl_bo is None:
        st.info(
            "Run the **Multitask Learning** tab first. BO/MOBO uses "
            "the trained data (X, Y) and the joint surrogate. "
            "(GPs are fit fresh here on the same X, Y.)"
        )
        return

    import torch
    from botorch.models import SingleTaskGP, ModelListGP
    from botorch.fit import fit_gpytorch_mll
    from gpytorch.mlls import (
        ExactMarginalLogLikelihood,
        SumMarginalLogLikelihood,
    )
    from botorch.models.transforms import Normalize, Standardize
    from botorch.optim import optimize_acqf

    X_bo      = np.asarray(mtl_bo["X"], dtype=np.float64)
    Y_bo_raw  = np.asarray(mtl_bo["Y"], dtype=np.float64)
    feat_nm_b = list(mtl_bo["feature_names"])
    t_cols_b  = list(mtl_bo["target_cols"])
    n_targets = len(t_cols_b)

    # Sidebar target list changed since MTL was last run → the
    # stored mtl_results is stale. Warn rather than silently
    # using a different target set than the sidebar advertises.
    if t_cols_b != list(target_cols):
        st.warning(
            f"Sidebar targets `{', '.join(target_cols)}` do not "
            f"match the last Multitask Learning run "
            f"(`{', '.join(t_cols_b)}`). BO/MOBO below uses the "
            "MTL run's targets — re-run the **Multitask Learning** "
            "tab to refresh."
        )

    st.caption(
        f"Surrogate source: joint **{mtl_bo['base_choice']}** model "
        f"with {len(feat_nm_b)} feature(s) and {n_targets} target(s) "
        f"on {X_bo.shape[0]} training rows."
    )
    if mtl_bo["task"] == "classification":
        st.warning(
            "BO on classification targets uses the label-encoded "
            "integer class indices as the objective. "
            "*Maximize* pushes candidates toward higher class indices."
        )

    # ── 1. Direction per target ──────────────────────────────
    st.markdown("#### 1. Optimization direction per target")
    if "bo_directions" not in st.session_state or \
       st.session_state.get("_bo_dirs_for") != t_cols_b:
        st.session_state.bo_directions = {
            t: "maximize" for t in t_cols_b
        }
        st.session_state["_bo_dirs_for"] = list(t_cols_b)

    dir_cols = st.columns(min(3, max(1, n_targets)))
    for ti, t_col in enumerate(t_cols_b):
        with dir_cols[ti % len(dir_cols)]:
            st.session_state.bo_directions[t_col] = st.radio(
                f"`{t_col}`",
                ["maximize", "minimize"],
                index=0 if st.session_state.bo_directions.get(t_col, "maximize") == "maximize" else 1,
                key=f"bo_dir_{t_col}",
                horizontal=True,
            )
    directions = [st.session_state.bo_directions[t] for t in t_cols_b]
    direction_signs = np.array(
        [1.0 if d == "maximize" else -1.0 for d in directions]
    )

    # Y oriented so higher = better in all targets
    Y_bo = Y_bo_raw * direction_signs[None, :]

    # ── 2. Pareto front / leaderboard of existing data ───────
    st.markdown("---")
    st.markdown("#### 2. Existing data view")

    def _pareto_mask(Yo):
        """Return boolean mask of Pareto-optimal rows (higher = better)."""
        n = Yo.shape[0]
        keep = np.ones(n, dtype=bool)
        for i in range(n):
            if not keep[i]:
                continue
            # j dominates i if j ≥ i everywhere and > i somewhere
            dom = (
                np.all(Yo >= Yo[i], axis=1)
                & np.any(Yo > Yo[i], axis=1)
            )
            dom[i] = False
            if dom.any():
                keep[i] = False
        return keep

    # Pull predicted Y values of the last BO/MOBO batch, if any —
    # used to overlay candidates on the existing-data view.
    bo_cands_df = st.session_state.get("bo_candidates")
    cand_preds = None
    if bo_cands_df is not None:
        pred_cols = [f"pred_{t}" for t in t_cols_b]
        if all(c in bo_cands_df.columns for c in pred_cols):
            cand_preds = bo_cands_df[pred_cols].to_numpy(dtype=float)

    if n_targets == 1:
        # Sorted leaderboard
        t = t_cols_b[0]
        ascending = directions[0] == "minimize"
        ldb = pd.DataFrame({"Source": ["data"] * X_bo.shape[0],
                            t: Y_bo_raw[:, 0]})
        for fi, fn in enumerate(feat_nm_b):
            ldb[fn] = X_bo[:, fi]

        # Append BO candidates with their predicted target
        if cand_preds is not None and cand_preds.shape[0]:
            cand_rows = pd.DataFrame({
                "Source": ["BO suggestion"] * cand_preds.shape[0],
                t: cand_preds[:, 0],
            })
            for fi, fn in enumerate(feat_nm_b):
                cand_rows[fn] = bo_cands_df[fn].to_numpy()
            ldb = pd.concat([ldb, cand_rows], ignore_index=True)

        ldb_sorted = ldb.sort_values(t, ascending=ascending).reset_index(drop=True)
        arrow = "↓" if ascending else "↑"
        st.markdown(
            f"**Sorted leaderboard for `{t}` ({directions[0]} {arrow})** — top rows are best."
        )
        if cand_preds is not None:
            st.caption(
                f"Includes {cand_preds.shape[0]} BO suggestion(s) "
                "interleaved with existing data based on predicted target."
            )

        # Highlight BO suggestion rows
        def _highlight_src(row):
            if row["Source"] == "BO suggestion":
                return ["background-color:#3d2a00;color:#ffa657"] * len(row)
            return [""] * len(row)
        st.dataframe(
            ldb_sorted.head(20).style.apply(_highlight_src, axis=1),
            use_container_width=True,
        )
    else:
        # Pareto front view (multi-target)
        pareto_keep = _pareto_mask(Y_bo)
        n_pareto = int(pareto_keep.sum())
        st.markdown(
            f"**{n_pareto} Pareto-optimal point(s)** out of "
            f"{Y_bo.shape[0]} based on the chosen directions."
        )

        pareto_mask_arr = pareto_keep
        dom_idx    = np.where(~pareto_mask_arr)[0]
        pareto_idx = np.where(pareto_mask_arr)[0]

        if n_targets == 2:
            fig_p = go.Figure()
            fig_p.add_trace(go.Scatter(
                x=Y_bo_raw[dom_idx, 0], y=Y_bo_raw[dom_idx, 1],
                mode="markers", name="Dominated",
                marker=dict(color="#6e7681", size=8, opacity=0.7),
            ))
            fig_p.add_trace(go.Scatter(
                x=Y_bo_raw[pareto_idx, 0], y=Y_bo_raw[pareto_idx, 1],
                mode="markers", name="Pareto",
                marker=dict(color="#56d364", size=10,
                            line=dict(width=1, color="#0d2d1c")),
            ))
            if cand_preds is not None and cand_preds.shape[0]:
                fig_p.add_trace(go.Scatter(
                    x=cand_preds[:, 0], y=cand_preds[:, 1],
                    mode="markers+text",
                    name="BO suggestion (predicted)",
                    marker=dict(color="#f0883e", size=14,
                                symbol="star",
                                line=dict(width=1, color="#3d2a00")),
                    text=[f"#{i+1}" for i in range(cand_preds.shape[0])],
                    textposition="top center",
                    textfont=dict(color="#ffa657", size=10),
                ))
            fig_p.update_layout(
                title="Existing data — Pareto front (BO suggestions overlaid)"
                      if cand_preds is not None
                      else "Existing data — Pareto front",
                xaxis_title=f"{t_cols_b[0]} ({directions[0]})",
                yaxis_title=f"{t_cols_b[1]} ({directions[1]})",
                **PLOT_THEME,
            )
            st.plotly_chart(fig_p, use_container_width=True)

        elif n_targets == 3:
            fig_p = go.Figure()
            fig_p.add_trace(go.Scatter3d(
                x=Y_bo_raw[dom_idx, 0], y=Y_bo_raw[dom_idx, 1],
                z=Y_bo_raw[dom_idx, 2],
                mode="markers", name="Dominated",
                marker=dict(color="#6e7681", size=4, opacity=0.6),
            ))
            fig_p.add_trace(go.Scatter3d(
                x=Y_bo_raw[pareto_idx, 0], y=Y_bo_raw[pareto_idx, 1],
                z=Y_bo_raw[pareto_idx, 2],
                mode="markers", name="Pareto",
                marker=dict(color="#56d364", size=6,
                            line=dict(width=0.5, color="#0d2d1c")),
            ))
            if cand_preds is not None and cand_preds.shape[0]:
                fig_p.add_trace(go.Scatter3d(
                    x=cand_preds[:, 0], y=cand_preds[:, 1],
                    z=cand_preds[:, 2],
                    mode="markers+text",
                    name="BO suggestion (predicted)",
                    marker=dict(color="#f0883e", size=8,
                                symbol="diamond",
                                line=dict(width=1, color="#3d2a00")),
                    text=[f"#{i+1}" for i in range(cand_preds.shape[0])],
                    textfont=dict(color="#ffa657", size=10),
                ))
            fig_p.update_layout(
                title="Existing data — 3D Pareto (BO suggestions overlaid)"
                      if cand_preds is not None
                      else "Existing data — 3D Pareto",
                scene=dict(
                    xaxis_title=f"{t_cols_b[0]} ({directions[0]})",
                    yaxis_title=f"{t_cols_b[1]} ({directions[1]})",
                    zaxis_title=f"{t_cols_b[2]} ({directions[2]})",
                ),
                height=550,
                **PLOT_THEME,
            )
            st.plotly_chart(fig_p, use_container_width=True)

        if cand_preds is not None:
            st.caption(
                f"Orange star/diamond markers = {cand_preds.shape[0]} BO suggestion(s) "
                "plotted at the GP's predicted target values."
            )

        # Pareto rows table
        pf_rows = pd.DataFrame(
            np.hstack([Y_bo_raw[pareto_keep], X_bo[pareto_keep]]),
            columns=t_cols_b + feat_nm_b,
        )
        st.markdown("**Pareto-optimal rows:**")
        st.dataframe(pf_rows, use_container_width=True)

    # ── 3. Per-feature bounds ─────────────────────────────────
    st.markdown("---")
    st.markdown("#### 3. Per-feature search bounds")
    st.caption(
        "Defaults to the observed min/max per feature. Edit cells to "
        "widen or narrow the search space before BO."
    )
    st.warning(
        f"**Feature space:** values below are in the "
        f"`{mtl_bo.get('x_space', 'training')}` space used by the "
        f"MTL surrogate, **not** your original units. "
        f"BO candidates downloaded later are in the same space — "
        f"inverse-transform before applying as new experiments.\n\n"
        f"**Extrapolation:** bounds wider than observed data → "
        f"the GP extrapolates; candidates often pile up at the "
        f"bound edges where uncertainty is highest."
    )

    default_bounds = pd.DataFrame({
        "Feature":  feat_nm_b,
        "Min":      X_bo.min(axis=0).astype(float),
        "Max":      X_bo.max(axis=0).astype(float),
    })
    edited_bounds = st.data_editor(
        default_bounds,
        key="bo_bounds_editor",
        use_container_width=True,
        num_rows="fixed",
        column_config={
            "Feature": st.column_config.TextColumn(disabled=True),
            "Min":     st.column_config.NumberColumn(format="%.6g"),
            "Max":     st.column_config.NumberColumn(format="%.6g"),
        },
    )

    # ── 4. Candidate count + Run ──────────────────────────────
    st.markdown("---")
    st.markdown("#### 4. Run BO / MOBO")
    n_candidates = st.slider(
        "Number of candidate points (q)",
        min_value=1, max_value=20, value=5, key="bo_q",
    )
    num_restarts = st.slider(
        "Acquisition optimizer restarts",
        min_value=4, max_value=32, value=10, key="bo_restarts",
        help="More restarts = better acquisition optimum, slower.",
    )
    raw_samples = st.select_slider(
        "Acquisition initial raw samples",
        options=[128, 256, 512, 1024, 2048],
        value=512, key="bo_raw_samples",
    )

    bo_label = "MOBO" if n_targets > 1 else "BO"
    if st.button(f"Run {bo_label}", key="btn_run_bo", type="primary"):
        try:
            # Validate bounds
            lows  = edited_bounds["Min"].to_numpy(dtype=np.float64)
            highs = edited_bounds["Max"].to_numpy(dtype=np.float64)
            if np.any(highs <= lows):
                bad = [
                    feat_nm_b[i]
                    for i in range(len(feat_nm_b))
                    if highs[i] <= lows[i]
                ]
                st.error(
                    f"Min must be < Max for every feature. "
                    f"Bad: {bad}"
                )
                st.stop()

            bounds_t = torch.tensor(
                np.stack([lows, highs]), dtype=torch.double
            )
            train_X_t = torch.tensor(X_bo, dtype=torch.double)
            train_Y_t = torch.tensor(Y_bo, dtype=torch.double)

            with st.spinner(
                f"Fitting GP surrogate(s) and optimizing "
                f"{bo_label} acquisition…"
            ):
                if n_targets == 1:
                    # qLogEI handles both q=1 and q>1 batch suggestions.
                    from botorch.acquisition.logei import (
                        qLogExpectedImprovement,
                    )
                    from botorch.sampling.normal import SobolQMCNormalSampler

                    gp = SingleTaskGP(
                        train_X_t, train_Y_t,
                        input_transform=Normalize(
                            d=train_X_t.shape[1], bounds=bounds_t,
                        ),
                        outcome_transform=Standardize(m=1),
                    )
                    mll = ExactMarginalLogLikelihood(
                        gp.likelihood, gp
                    )
                    fit_gpytorch_mll(mll)

                    best_f = train_Y_t.max().item()
                    sampler_bo = SobolQMCNormalSampler(
                        sample_shape=torch.Size([128]),
                    )
                    acqf = qLogExpectedImprovement(
                        model    = gp,
                        best_f   = best_f,
                        sampler  = sampler_bo,
                    )

                    cand, _ = optimize_acqf(
                        acq_function = acqf,
                        bounds       = bounds_t,
                        q            = int(n_candidates),
                        num_restarts = int(num_restarts),
                        raw_samples  = int(raw_samples),
                    )
                    cand_np = cand.detach().cpu().numpy()
                    with torch.no_grad():
                        post_mean = gp.posterior(cand).mean.detach().cpu().numpy().reshape(-1, 1)
                else:
                    # MOBO
                    from botorch.acquisition.multi_objective.logei import (
                        qLogExpectedHypervolumeImprovement,
                    )
                    from botorch.utils.multi_objective.box_decompositions.non_dominated import (
                        FastNondominatedPartitioning,
                    )
                    from botorch.sampling.normal import SobolQMCNormalSampler

                    gp_models = []
                    for ti in range(n_targets):
                        gp_i = SingleTaskGP(
                            train_X_t,
                            train_Y_t[:, ti:ti+1],
                            input_transform=Normalize(
                                d=train_X_t.shape[1],
                                bounds=bounds_t,
                            ),
                            outcome_transform=Standardize(m=1),
                        )
                        gp_models.append(gp_i)
                    model_list = ModelListGP(*gp_models)
                    mll = SumMarginalLogLikelihood(
                        model_list.likelihood, model_list,
                    )
                    fit_gpytorch_mll(mll)

                    Y_range = (
                        train_Y_t.max(dim=0).values
                        - train_Y_t.min(dim=0).values
                    ).clamp_min(1e-9)
                    ref_point_t = (
                        train_Y_t.min(dim=0).values
                        - 0.1 * Y_range
                    )

                    partitioning = FastNondominatedPartitioning(
                        ref_point=ref_point_t, Y=train_Y_t,
                    )
                    sampler = SobolQMCNormalSampler(
                        sample_shape=torch.Size([128]),
                    )
                    # ref_point as list[float] (NOT tensor) — the tensor form
                    # silently breaks qLogEHVI in some BoTorch versions; see
                    # feedback memory `feedback-botorch-ref-point`.
                    acqf = qLogExpectedHypervolumeImprovement(
                        model        = model_list,
                        ref_point    = ref_point_t.tolist(),
                        partitioning = partitioning,
                        sampler      = sampler,
                    )

                    cand, _ = optimize_acqf(
                        acq_function = acqf,
                        bounds       = bounds_t,
                        q            = int(n_candidates),
                        num_restarts = int(num_restarts),
                        raw_samples  = int(raw_samples),
                    )
                    cand_np = cand.detach().cpu().numpy()
                    with torch.no_grad():
                        pm_cols = []
                        for gp_i in gp_models:
                            pm_cols.append(
                                gp_i.posterior(cand)
                                    .mean.detach().cpu().numpy()
                                    .reshape(-1)
                            )
                        post_mean = np.column_stack(pm_cols)

            # Invert direction sign so predicted targets read
            # in user-original scale (max stays max, min stays min)
            post_mean_user = post_mean * direction_signs[None, :]

            cand_df = pd.DataFrame(
                cand_np, columns=feat_nm_b
            )
            for ti, t_col in enumerate(t_cols_b):
                cand_df[f"pred_{t_col}"] = post_mean_user[:, ti]

            st.session_state.bo_candidates = cand_df
            st.success(
                f"{bo_label} suggested **{len(cand_df)}** candidate "
                f"point(s). See the table below."
            )
            # The "Existing data view" block above reads bo_candidates
            # from session_state and overlays them on the Pareto plot —
            # but it already ran with the OLD value during this script
            # pass. Rerun once so the overlay picks up the new batch
            # immediately instead of only after the user's next
            # interaction.
            st.rerun()

        except Exception as e:
            st.error(f"{bo_label} failed: {e}")
            import traceback
            with st.expander("Traceback"):
                st.code(traceback.format_exc())

    # ── Display latest candidates ─────────────────────────────
    if st.session_state.get("bo_candidates") is not None:
        st.markdown("#### Suggested candidate points")
        st.dataframe(
            st.session_state.bo_candidates,
            use_container_width=True,
        )
        st.download_button(
            label="⬇️ Download candidates (CSV)",
            data=st.session_state.bo_candidates
                .to_csv(index=False).encode("utf-8"),
            file_name="bo_candidates.csv",
            mime="text/csv",
        )
