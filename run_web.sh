#!/bin/bash

echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
echo "ADSP_LIBRARY_PATH: $ADSP_LIBRARY_PATH"

# Add project directories to Python path
export PYTHONPATH="${PWD}:${PWD}/Application:${PYTHONPATH}"
# python --version

base_script_dir="./Application"
python3.10 "${base_script_dir}/web.py" \
    --camera 2 \
    --db-path "${base_script_dir}/face_database" \
    --scrfd-dlc "./SCRFD/Model/scrfd.dlc" \
    --arcface-dlc "./ArcFace/Model/arcface_quantized_6490.dlc" \
    --threshold 0.5 \
    --skip-frames 2 \
    --runtime DSP \
    --port 8080 # Port for the web interface (default: 5000)
