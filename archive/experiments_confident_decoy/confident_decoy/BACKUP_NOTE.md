# Confident-decoy experiments — scored-results backup

Code moved to `tts_abag/confident_decoy_experiments/` (in git) on 2026-08-07.
This directory now holds ONLY the artifact JSONs — the scored results.

Live data (NOT in git): `/home/ebrahim/confident_decoy_experiments/{artifacts,predictions}` (87 GB).
Every headline number is re-tallyable from the JSONs here.

Re-sync (JSONs only — do NOT sync scripts/configs, they live in git now):
```bash
rsync -a --include='*.json' --exclude='*' \
  /home/ebrahim/confident_decoy_experiments/artifacts/ \
  experiments/confident_decoy/artifacts/
```

NOT backed up (regenerable, GPU-expensive): `artifacts/*_ost/` score caches,
`artifacts/sweep_out/` prediction CIFs, `Protenix/foldbench_abag/predictions_template/`
free-run pools, `/home/ebrahim/mmcif/*.cif` registered templates.
