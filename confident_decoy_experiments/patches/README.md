# Protenix patches used by this project

## `protenix_unmask.patch`
Applied to a COPY of the Protenix code tree (`/home/ebrahim/Protenix_unmask`, 12 MB: `protenix/`,
`configs/`, `runner/`). `/home/ebrahim/Protenix` is never modified — a running sweep spawns a fresh
inference process per job, so editing the shared tree would silently change arms mid-run.

Two changes:

1. **`pairformer.py:1013` — unmask cross-chain template features.** Stock Protenix (from AF3) sets
   `multichain_mask = (asym_id[:,None] == asym_id[None,:])` and multiplies the template distogram,
   unit vectors and both frame masks by it, so a template can only transmit INTRA-chain geometry.
   The patch forces the mask to ones. Also logs `UNMASKED TEMPLATES ACTIVE ... mean 1.0000` once per
   process, because a masked and an unmasked run are otherwise indistinguishable from their output.

2. **`protenix.py` — `_mc_log` helper** instrumenting the MC-dropout decision.
   NOTE: it is wired into the MULTI-model-seed call site only. With the default `N_model_seed=1`
   inference takes the single-seed branch, so the log never fires. The dropout condition was instead
   verified by resolving the config directly:
       `--mc_dropout_apply_rate 0` -> 0.0, and `random.random() < 0` is always False.

### Reproduce
    cp -r /home/ebrahim/Protenix/{protenix,configs,runner} /home/ebrahim/Protenix_unmask/
    cd /home/ebrahim/Protenix_unmask && patch -p0 < .../patches/protenix_unmask.patch

### Run against it
    PYTHONPATH=/home/ebrahim/Protenix_unmask LAYERNORM_TYPE=openfold \
      python runner/inference.py ...     # cwd must be /home/ebrahim/Protenix_unmask
`PYTHONPATH` is REQUIRED: `python runner/inference.py` puts the SCRIPT's directory on `sys.path`,
not the cwd, so `import protenix` otherwise resolves to the editable install at
`/home/ebrahim/Protenix` and the patch silently does nothing.
