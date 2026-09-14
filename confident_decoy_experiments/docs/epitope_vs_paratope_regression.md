# What drives the DockQ gain from seed scaling: epitope, or paratope?

n = 148 antibody–antigen interfaces (FoldBench). Every quantity is a **gain** from seed scaling, and
both arms are ipTM-selected — no oracle — so this describes what a user actually receives.

- **DockQ gain** = DockQ of the 50-seed ipTM pick − median DockQ of the 50 per-seed ipTM picks
- **epitope / paratope gain** = same construction on 5 Å contact-set Jaccard against the crystal,
  antigen side and antibody side respectively

Source: `artifacts/abag_selection_protocol.csv`; figures from
`scripts/plot_gain_epitope_vs_dockq.py`.

## Model

DockQ gain = intercept + a·(epitope gain) + b·(paratope gain), fitted by least squares.

| model | R² | adjusted R² | 10-fold cross-validated R² | residual sum of squares | intercept | a (epitope) | b (paratope) |
|---|---|---|---|---|---|---|---|
| intercept only | 0.000 | 0.000 | — | 12.09 | +0.167 | — | — |
| epitope only | 0.821 | 0.820 | 0.819 | 2.17 | +0.030 | +0.874 | — |
| paratope only | 0.395 | 0.391 | 0.370 | 7.31 | +0.098 | — | +0.912 |
| epitope + paratope | 0.827 | 0.825 | 0.823 | 2.09 | +0.028 | +0.813 | +0.147 |

The intercept-only row is the null model: predict the mean gain (+0.167) for every interface. Its
residual sum of squares, 12.09, is the total variation the other models are trying to remove.

## Adding the paratope buys +0.006

R² rises 0.821 → 0.827 and the residual sum of squares falls 2.17 → 2.09. Adjusted R² moves by
+0.005, cross-validated R² by +0.004. The second predictor is not earning its parameter.

## Variance partition

| | share of DockQ-gain variance |
|---|---|
| unique to epitope | **0.432** |
| unique to paratope | **0.006** |
| shared by the two | 0.389 |

The predictors correlate at r = 0.626, so most of the paratope's apparent 0.395 is borrowed from the
epitope rather than its own.

## The coefficient shift confirms it

Fitted alone the paratope slope is +0.912; fitted jointly it collapses to **+0.147**. The epitope
slope barely moves (+0.874 → +0.813). A predictor whose coefficient collapses when a correlated
variable is introduced is a proxy, not a driver — the paratope looked predictive only through its
correlation with epitope gain.

Mechanistically this is expected: when the antibody finds the correct antigen surface its CDRs end up
in the right place as a consequence. Among the 38 interfaces with a DockQ gain above 0.2, mean
epitope gain is +0.562 versus +0.274 for the paratope.

## Why the paratope has so little to say

At the 1-seed baseline the paratope Jaccard is already 0.594 ± 0.202, with only **1%** of interfaces
below 0.1. The epitope starts at 0.435 ± 0.343 with **30%** below 0.1. The model nearly always has
the antibody face roughly right and nearly always has the antigen face wrong, so only the epitope has
room to make the 0 → 1 jumps that carry the signal.

## Recommendation

Report the single-predictor epitope model: **gain in epitope Jaccard explains 82% of the variance in
DockQ gain (n = 148, slope 0.87), versus 40% for paratope gain.** The joint model adds a decimal
place and a second parameter for no explanatory gain; the variance partition is the cleaner way to
state that the paratope is not doing the work.

Avoid phrasing it as "epitope improvement causes 82% of the DockQ improvement" — R² is symmetric and
supports an explanatory claim, not a causal decomposition. The directional reading rests on the
mechanism (a pose on the wrong antigen surface cannot score well, and epitope Jaccard is measured on
the antigen alone and is not a DockQ component).
