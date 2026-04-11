#!/bin/bash
# SNPE root
SNPE_ROOT=/home/ubuntu/qairt/2.28.0.241029

# Runtime shared libraries
export LD_LIBRARY_PATH=${SNPE_ROOT}/lib/aarch64-ubuntu-gcc9.4:$LD_LIBRARY_PATH
# Binaries (snpe-net-run, qnn-net-run, etc.)
export PATH=${SNPE_ROOT}/bin/aarch64-ubuntu-gcc9.4:$PATH

# Hexagon DSP skel libraries (required for --use_dsp / HTP backend)
export ADSP_LIBRARY_PATH="${SNPE_ROOT}/lib/hexagon-v68/unsigned/;/usr/lib/rfsa/adsp;/dsp;."