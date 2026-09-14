# Why does scaling the number of seeds help for antibody–antigen complexes?

## 1. Research question

Test-time scaling by running multiple random seeds substantially improves antibody–antigen structure
prediction, and this effect appears specific to antibody–antigen interfaces. AlphaFold3 reports the
fraction of acceptable poses (DockQ > 0.23) rising from ~40% to ~60% when scaling from 1 to 1000 seeds
on an antibody–antigen benchmark, while reporting no comparable benefit for other protein–protein
complexes, nor for protein–DNA or protein–RNA interfaces. The same pattern holds across models: on
FoldBench ABAG, ESMFold2 improves from 0.49 to 0.64 and ESMFold2 Fast from 0.45 to 0.64, and
Protenix-v2 from 0.56 to 0.74, while the corresponding gain on FoldBench protein–protein is below 0.04.

We reproduce the effect on Protenix. Across 161 labelled antibody–antigen interfaces (104 complexes,
40,249 scored poses), going from 1 seed to 50 raises oracle DockQ from 0.426 to 0.697, and the pass
rate of the pose selected by interface pair-ipTM from 50.0% to 72.7%.

A seed changes two things at inference: which rows of the MSA are subsampled at each recycling
iteration, and the dropout masks. Notably, AlphaFold3 reports that reducing from five diffusion
samples per seed to one has little effect on results, indicating that the diffusion sampler is not the
principal source of the benefit.

**Why does scaling the number of seeds help so much for antibody–antigen complexes specifically, and
not for other biomolecular interfaces?**
