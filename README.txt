Canonical KV offload validation, 2026-10-17

Tested source: fix a99a46153a3b13034339b32942773b44d0cb651d;
feature 3172b52d123acf1bea4f78313089d58354e9ac3a.
Base/native extensions: 8e25bbca19f0605b347e103dde4bf5355d049e72.
Hardware: 2 x NVIDIA RTX 5090 (32 GiB each), Linux x86_64.
See environment.json for package versions and source-manifest.json for hashes.

Fix: 135 CPU tests passed; 5 GPU processes exited 0.
Feature: 702 CPU tests passed, 2 skipped; 10 GPU processes exited 0.
analysis.json records transfer counts, hits and all-vocabulary logprob deltas.
Tolerance: 0.005. Fixed teacher-forced contexts: 240; vocabulary: 512.
All cross-TP reader cold/reload phases hit 240/240 contexts with real FS reads
and CPU-to-GPU transfers. Pressure cycles intentionally permit misses.

Scripts are archived verbatim from the test host, not a turnkey installer.
run-final.sh records commands and host-local /data paths. Its stage helper
installed the corresponding source snapshot; instead use separate checkouts
at the two tested commits and a compatible vLLM installation. CPU and GPU
Python environments were separate. Adapt /data paths to your machine.
reproduction/ contains checkpoint generators and inference/probe helpers.
For the pure-full-attention control, copy the KV1 checkpoint and change its
config no_rope_layers to [0,0,0,0], retaining identical weight files.
Generate offload-disabled TP1 references using run_inference.py; use their
JSON outputs as --teacher-reference. The sharded baseline's cold phase also
supplies teacher-base-tp1-r1.json used by analyze-final.py.
Place gpu_probe.py and run_inference.py on PYTHONPATH for worker/stress imports.
run-final.sh lists producer/read pairs and stress commands, including CPU suites.
The analyzer expects raw per-run JSONs, exit codes and verified manifests.
Raw full-vocabulary outputs and weights are retained locally but are not in
this compact evidence branch. No model-quality or performance claim is made.

AI assistance: Codex helped prepare implementation, validation and this archive.
