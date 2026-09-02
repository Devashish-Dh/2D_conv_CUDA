# 2D convolution on CUDA (HPCA assignment)

GPU 2D convolution with a naive baseline and room for progressive kernels (coalescing, on-chip memory, tiling, concurrency). Course work for HPCA; the vendor-style assignment brief is in [`README FOR ASSIGNMENT.md`](README%20FOR%20ASSIGNMENT.md). The original starter tree is under [`hpca-gpu-assignment-2025-RAW/`](hpca-gpu-assignment-2025-RAW/).

## Report and archives

- [Report (PDF)](docs/reports/report.pdf)
- [Assignment zip](docs/archives/assignment-27231.zip)

Nsight Compute / Systems captures: [`ncu_rep/`](ncu_rep/), [`nsys_rep/`](nsys_rep/). Notes and plots: [`results_obs_etc/`](results_obs_etc/). Working sources: [`program_files/`](program_files/).

## Build and run (assignment driver)

From the assignment tree (see the full brief for flags):

```bash
make clean && make
./gpu_conv --n=16 --h=2048 --w=2048 --k=11 --impl=naive --iters=10 --verify
```

Implement variants in `src/conv_kernels.cu` as described in the assignment README. Needs an NVIDIA GPU and a CUDA toolkit that matches the Makefile.
