# Decomposing where DockQ gain comes from: site, orientation, register

Status: PROPOSED, not run. All inputs are already on disk (the OST JSONs behind
`artifacts/abag_selection_protocol.csv`), so this is CPU-only.

## The question

Seed scaling raises ipTM-selected DockQ on ab-ag by +0.167 (n=148). We know **82% of that gain is
explained by gain in epitope Jaccard** (`scripts/plot_gain_epitope_vs_dockq.py`, slope 0.87,
r² = 0.82). That leaves the rest unexplained, and it leaves open what happens on interfaces where the
model was already on roughly the right patch.

Ebrahim's framing: DockQ can improve two ways — (1) the model finds the epitope/paratope, or
(2) it was already on largely the right patch but the residues are matched up wrong. Agreed, with
(2) split in two, because they imply different fixes.

## The three categories

| | what is wrong | fix it would need |
|---|---|---|
| **1. site** | wrong patch entirely | better search / coverage |
| **2a. orientation** | right patches, antibody rotated or flipped against the antigen | rigid-body correction |
| **2b. register** | right patches, right orientation, residue pairings offset by a few positions | local refinement |

Orientation vs register, concretely: take the correct pose. Spin the antibody 90° while keeping it on
the same patch — the surfaces still touch but the wrong parts of the antibody face the antigen
(CDR-H3 where CDR-L1 should be). That is orientation. Now instead slide the antibody three residues
along the surface, keeping it facing the right way — like buttoning a shirt one hole off. That is
register.

## Why epitope Jaccard cannot see either

Epitope Jaccard is a set overlap computed on the ANTIGEN side only. A flipped antibody contacting the
correct antigen residues scores 1.0. So the 82% result says nothing about 2a or 2b, and any
interface that fails only in those ways is invisible to it.

## Predicted metric signature

| | epitope Jaccard | fnat (correct residue PAIRS) | iRMSD |
|---|---|---|---|
| site wrong | low | low | large |
| orientation wrong | can be high | very low | large |
| register wrong | high | low | small |

The discriminating quantity is **iRMSD**: a flipped antibody cannot be superposed onto the native
without a large residual, whereas a three-residue slide almost can.

CAVEAT: this signature is a prediction, not a verified separation. It is plausible that 2a and 2b
blur together on real data. The measurement is what decides it.

## The measurement

For each of the 148 interfaces, decompose the DockQ gain between the 1-seed and 50-seed
ipTM-selected poses into its three DockQ components — fnat, f(iRMSD), f(LRMSD) — since
DockQ = (fnat + f(iRMSD) + f(LRMSD))/3 with f(x) = 1/(1+(x/d)²), d = 1.5 Å (iRMSD), 8.5 Å (LRMSD).
Cross that against the epitope Jaccard gain already computed.

## The target population

**The 18 interfaces that gained epitope Jaccard but LOST DockQ.** By construction those can only be
2a or 2b — the model found a better antigen surface and still produced a worse structure. They are
also the population where "search harder at the epitope you already found" would pay off, which is
the intervention the sequential-repulsion plan is aiming at.

The count is stable at 17–18 across both baseline conventions, so it is a real population and not an
artefact of how the 1-seed baseline is defined.
