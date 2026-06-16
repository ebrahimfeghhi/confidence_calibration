"""Render 7txw A|C predictions: a good seed vs a bad seed vs ground truth.

Loads GT, seed_19 (DockQ~0.95) and seed_0 (DockQ~0.03), superposes everything
on the antibody (chain A), and colors the antigen (chain C) so the two docking
poses can be compared visually.

Run with the isolated PyMOL venv:
    /home/ebrahim/.venvs/pymol/bin/python plot/render_7txw_poses.py
"""
import os
from pymol2 import PyMOL

PREDROOT = "/home/ebrahim/esm/foldbench_predictions/esmfold2fast_tts/7txw"
GT = "/home/ebrahim/FoldBench/examples/ground_truths/7txw-assembly1.cif"
GOOD = f"{PREDROOT}/seed_19/predictions/7txw_sample_0.cif"   # DockQ 0.948
BAD = f"{PREDROOT}/seed_0/predictions/7txw_sample_0.cif"     # DockQ 0.029
OUTDIR = "/home/ebrahim/tts_abag/analysis/figs"

os.makedirs(OUTDIR, exist_ok=True)

p = PyMOL()
p.start()
cmd = p.cmd

# Use label_asym_id so all objects share chain naming A,B = antibody; C = antigen.
# (GT's auth_asym_id is H,L,A which would mislabel the antigen as "A".)
cmd.set("cif_use_auth", 0)

cmd.load(GT, "gt")
cmd.load(GOOD, "good")
cmd.load(BAD, "bad")

cmd.remove("not polymer")
cmd.hide("everything")
cmd.show("cartoon")

# Superpose predictions onto GT using the ANTIBODY (chain A) only; the whole
# object moves, so the antigen (chain C) lands wherever the model docked it.
r_good = cmd.align("good and chain A", "gt and chain A")
r_bad = cmd.align("bad and chain A", "gt and chain A")
print(f"align good->gt on chain A: RMSD={r_good[0]:.2f} A over {r_good[1]} atoms")
print(f"align bad ->gt on chain A: RMSD={r_bad[0]:.2f} A over {r_bad[1]} atoms")

# Antibody shown once (from GT); antigens colored per source.
cmd.color("grey70", "gt and chain A+B")
cmd.color("green", "gt and chain C")        # native antigen pose
cmd.color("marine", "good and chain C")     # good seed (DockQ 0.95)
cmd.color("firebrick", "bad and chain C")   # bad seed  (DockQ 0.03)
cmd.hide("cartoon", "(good or bad) and chain A+B")  # declutter overlaid antibodies

# Mark the native epitope (antigen residues contacting the antibody) as sticks.
cmd.select("epitope", "byres (gt and chain C within 5 of (gt and chain A))")
cmd.show("sticks", "epitope")
cmd.color("orange", "epitope")

cmd.bg_color("white")
cmd.set("ray_opaque_background", 1)
cmd.set("antialias", 2)
cmd.set("ray_shadows", 0)
cmd.set("cartoon_transparency", 0.15, "gt and chain A+B")
cmd.deselect()

W, H = 1700, 1300

# --- Image 1: overview, both poses ---
cmd.orient("gt or (good and chain C) or (bad and chain C)")
cmd.ray(W, H)
cmd.png(f"{OUTDIR}/7txw_overview.png", dpi=150)
print("wrote 7txw_overview.png")

# --- Image 2: same, rotated 90 deg for depth ---
cmd.turn("y", 90)
cmd.ray(W, H)
cmd.png(f"{OUTDIR}/7txw_overview_y90.png", dpi=150)
print("wrote 7txw_overview_y90.png")

# --- Image 3: zoom on native site, GT (green) vs good seed (blue) only ---
cmd.hide("everything", "bad")
cmd.show("cartoon", "gt and chain A+B")
cmd.set("cartoon_transparency", 0.6, "gt and chain A+B")
cmd.orient("(gt and chain C) or (good and chain C)")
cmd.zoom("(gt and chain C) or (good and chain C)", 5)
cmd.ray(W, H)
cmd.png(f"{OUTDIR}/7txw_native_zoom.png", dpi=150)
print("wrote 7txw_native_zoom.png")

p.stop()
