# gaffa

CUDA/C++ and Python library for pulsar data processing and fast-folding
algorithm (FFA) periodicity searches.

## Environment

```bash
conda env create -f environment.yml
conda activate gaffa
source env/dev.sh
```

The system layer should provide only the NVIDIA driver. The conda environment
provides Python 3.12, CUDA Toolkit 12.8, GCC/G++ 13, CMake, Ninja, and Conan.

## Development

```bash
just install
just test-all
```
