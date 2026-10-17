#!/bin/bash
set -eu
root=/data/vllm-final-gpu
prior=/data/vllm-gpu-followup
cd /data/vllm-chunked-local
export PYTHONPATH="$prior:." HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=2 VLLM_WORKER_MULTIPROC_METHOD=spawn VLLM_USE_FLASHINFER_SAMPLER=0
stage() {
 /data/vllm-cpu-venv/bin/python "$root/stage.py" "$1"
 echo "STAGE $1"
}
run() {
 name=$1; shift
 set +e
 /data/vllm-gpu-venv/bin/python "$prior/run_inference.py" "$@" --output "$root/$name.json" > "$root/$name.log" 2>&1
 status=$?
 set -e
 echo "$status" > "$root/$name.exit"
 echo "$name exit=$status"
 test "$status" = 0
}
cpu() {
 name=$1; shift
 set +e
 VLLM_TARGET_DEVICE=cpu CUDA_VISIBLE_DEVICES='' /data/vllm-cpu-venv/bin/python -m pytest "$@" -q > "$root/$name.log" 2>&1
 status=$?
 set -e
 echo "$status" > "$root/$name.exit"
 echo "$name exit=$status"
 test "$status" = 0
}
stage fix
cpu fix-cpu tests/v1/kv_connector/unit/offloading_connector/test_config.py tests/v1/kv_offload/test_factory.py
model="$prior/tiny-llama4-kv1-full"
run full-base --tp 1 --model "$model" --full-attention-control
run full-producer-tp1 --tp 1 --model "$model" --full-attention-control --offload --record-divergences --filesystem "$root/full-1to2"
run full-read-tp2 --tp 2 --model "$model" --full-attention-control --offload --filesystem "$root/full-1to2" --expect-external --teacher-reference "$root/full-base.json"
run full-producer-tp2 --tp 2 --model "$model" --full-attention-control --offload --record-divergences --filesystem "$root/full-2to1"
run full-read-tp1 --tp 1 --model "$model" --full-attention-control --offload --filesystem "$root/full-2to1" --expect-external --teacher-reference "$root/full-base.json"
stage feature
cpu feature-cpu tests/v1/kv_connector/unit/offloading_connector/test_config.py tests/v1/kv_connector/unit/offloading_connector/test_canonical_mapping.py tests/v1/kv_offload/test_file_mapper.py tests/v1/kv_connector/unit/offloading_connector/test_worker.py tests/v1/kv_connector/unit/offloading_connector/test_scheduler.py tests/v1/kv_offload/test_factory.py
model="$prior/tiny-llama4-kv1"
run replica-producer-tp1 --tp 1 --model "$model" --offload --record-divergences --filesystem "$root/replica-1to2"
run replica-read-tp2 --tp 2 --model "$model" --offload --filesystem "$root/replica-1to2" --expect-external --teacher-reference "$prior/replica-base.json"
run replica-producer-tp2 --tp 2 --model "$model" --offload --record-divergences --filesystem "$root/replica-2to1"
run replica-read-tp1 --tp 1 --model "$model" --offload --filesystem "$root/replica-2to1" --expect-external --teacher-reference "$prior/replica-base.json"
run sharded-producer-tp1 --tp 1 --offload --record-divergences --filesystem "$root/sharded-1to2"
run sharded-read-tp2 --tp 2 --offload --filesystem "$root/sharded-1to2" --expect-external --teacher-reference "$prior/base-tp1.json"
run sharded-producer-tp2 --tp 2 --offload --record-divergences --filesystem "$root/sharded-2to1"
run sharded-read-tp1 --tp 1 --offload --filesystem "$root/sharded-2to1" --expect-external --teacher-reference "$prior/base-tp1.json"
set +e
/data/vllm-gpu-venv/bin/python /data/vllm-closeout/stress_inference.py --tp 2 --model "$model" --offload --filesystem "$root/replica-1to2" --expect-external --teacher-reference "$prior/replica-base.json" --output "$root/stress-tp2.json" > "$root/stress-tp2.log" 2>&1
status=$?
set -e
echo "$status" > "$root/stress-tp2.exit"
echo "stress-tp2 exit=$status"
test "$status" = 0
run read-after-stress-tp1 --tp 1 --model "$model" --offload --filesystem "$root/replica-1to2" --expect-external --teacher-reference "$prior/replica-base.json"
echo 'ALL_RUNS_COMPLETED'
