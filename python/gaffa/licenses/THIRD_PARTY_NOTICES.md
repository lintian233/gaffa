# Third-party runtime notices

The CUDA-enabled Gaffa wheel may bundle the following dynamically linked
runtime components:

- Loki 0.0.3, MIT License.
- fmt 12.0.0, MIT License.
- FFTW 3.3.10, GNU General Public License v2 or later.
- HDF5 1.14.6, HDF5 License.
- yaml-cpp 0.8.0, MIT License; linked statically into the command-line tool.

The wheel does not bundle the NVIDIA kernel driver, CUDA runtime, cuFFT, or
cuRAND. Those components are supplied by the target CUDA 12 installation and
remain subject to NVIDIA's applicable license terms.
