export SNPE_ROOT=/home/ubuntu/qairt/2.28.0.241029
export LD_LIBRARY_PATH=${SNPE_ROOT}/lib/aarch64-ubuntu-gcc9.4:$LD_LIBRARY_PATH
export PATH=${SNPE_ROOT}/bin/aarch64-ubuntu-gcc9.4:$PATH
export ADSP_LIBRARY_PATH="${SNPE_ROOT}/lib/hexagon-v68/unsigned/;/usr/lib/rfsa/adsp;/dsp;."

base_script_dir="./Face Detection + Face Recognition"
python "${base_script_dir}/web.py" \
    --db-path "${base_script_dir}/face_database" \
    --scrfd-dlc "./SCRFD (Face Detection)/Model/scrfd.dlc" \
    --arcface-dlc "./ArcFace (Face Recognition)/Model/arcface_quantized_6490.dlc" \
    --threshold 0.9 \
    --skip-frames 2 \
    --runtime DSP \
    --port 5000 # Port for the web interface (default: 5000)