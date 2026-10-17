"""GPU generation and forced CPU KV reload, with transfer evidence."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time


def main():
    from vllm import LLM, SamplingParams
    from vllm.config import KVTransferConfig
    parser = argparse.ArgumentParser()
    parser.add_argument('--tp', type=int, required=True)
    parser.add_argument('--dtype', default='float16', choices=['float16', 'bfloat16'])
    parser.add_argument('--offload', action='store_true')
    parser.add_argument('--output', required=True)
    parser.add_argument('--filesystem')
    parser.add_argument('--expect-external', action='store_true')
    parser.add_argument('--teacher-reference')
    parser.add_argument('--model', default='/data/vllm-gpu-validation/tiny-llama4')
    parser.add_argument('--batch-tokens', type=int, default=64)
    parser.add_argument('--record-divergences', action='store_true')
    parser.add_argument('--serial', action='store_true')
    parser.add_argument('--full-attention-control', action='store_true')
    args = parser.parse_args()
    kwargs = {}
    if args.serial:
        kwargs['async_scheduling'] = False
    if args.offload:
        extra = {'spec_name': 'TieringOffloadingSpec',
                 'canonical_layout': True, 'cpu_bytes_to_use': 64 << 20}
        if args.filesystem:
            extra['secondary_tiers'] = [{'type': 'fs', 'root_dir': args.filesystem,
                                        'n_read_threads': 2, 'n_write_threads': 2}]
        kwargs['kv_transfer_config'] = KVTransferConfig(
            kv_connector='OffloadingConnector', kv_role='kv_both',
            kv_connector_extra_config=extra)
    llm = LLM(model=args.model,
              skip_tokenizer_init=True, dtype=args.dtype,
              tensor_parallel_size=args.tp, enforce_eager=True,
              max_model_len=768, max_num_seqs=1 if args.serial else 4,
              max_num_batched_tokens=args.batch_tokens,
              enable_chunked_prefill=True, enable_prefix_caching=True,
              block_size=16, gpu_memory_utilization=0.15,
              kv_cache_memory_bytes=64 << 20, seed=61886,
              disable_log_stats=False,
              max_logprobs=512,
              worker_extension_cls='gpu_probe.WorkerProbe',
              kernel_config={'moe_backend': 'triton'},
              attention_config={'backend': 'TRITON_ATTN'}, **kwargs)
    worker_info = llm.collective_rpc('describe_kv_worker')
    if args.full_attention_control:
        assert all(set(w['specs']) == {'FullAttentionSpec'} for w in worker_info)
    else:
        assert all('ChunkedLocalAttentionSpec' in w['specs'] for w in worker_info), worker_info
    if args.offload:
        assert all(w['canonical_layout'] and w['all_layers_portable'] for w in worker_info)
    sampling = SamplingParams(temperature=0, max_tokens=16, ignore_eos=True, logprobs=-1)
    lengths = [63, 64, 65, 127, 128, 129, 255, 256, 257,
               383, 384, 385, 511, 512, 513]
    prompts = [{'prompt_token_ids': [3 + ((i * 37 + n) % 509) for i in range(n)]}
               for n in lengths]
    if args.teacher_reference:
        reference = json.loads(Path(args.teacher_reference).read_text())['cold']
        prompts = []
        lengths = []
        for row in reference:
            n = row['length']
            prefix = [3 + ((i * 37 + n) % 509) for i in range(n)]
            for step in range(16):
                prompts.append({'prompt_token_ids': prefix + row['tokens'][:step]})
                lengths.append(n + step)
        sampling = SamplingParams(temperature=0, max_tokens=1, ignore_eos=True, logprobs=-1)

    def generate():
        results = llm.generate(prompts, sampling, use_tqdm=False)
        return [{'length': n, 'tokens': list(r.outputs[0].token_ids),
                 'cached': r.num_cached_tokens,
                 'logprobs': [{str(k): v.logprob for k, v in step.items()}
                              for step in r.outputs[0].logprobs]}
                for n, r in zip(lengths, results)]

    initial_metrics = [asdict(m) for m in llm.get_metrics()]
    cold = generate()
    deadline = time.monotonic() + 60
    while not llm.reset_prefix_cache(reset_connector=False):
        assert time.monotonic() < deadline, 'KV offload did not drain'
        llm.generate([{'prompt_token_ids': [7, 8, 9]}],
                     SamplingParams(temperature=0, max_tokens=1), use_tqdm=False)
        time.sleep(0.05)
    before = [asdict(m) for m in llm.get_metrics()]
    warm = generate()
    # Flush metric reporting and asynchronous transfer completion.
    llm.generate([{'prompt_token_ids': [10, 11, 12]}],
                 SamplingParams(temperature=0, max_tokens=1), use_tqdm=False)
    after = [asdict(m) for m in llm.get_metrics()]
    result = {'tp': args.tp, 'dtype': args.dtype, 'offload': args.offload, 'workers': worker_info,
              'cold': cold, 'reload': warm, 'metrics_before': before,
              'metrics_after': after, 'metrics_initial': initial_metrics,
              'filesystem': args.filesystem, 'teacher_reference': args.teacher_reference,
              'model': args.model, 'batch_tokens': args.batch_tokens,
              'serial': args.serial}
    Path(args.output).write_text(json.dumps(result, indent=2))
    if args.teacher_reference:
        max_delta = 0.0
        divergences = []
        for index, (a, b) in enumerate(zip(cold, warm)):
            pa, pb = a['logprobs'][0], b['logprobs'][0]
            assert len(pa) == len(pb) == 512
            max_delta = max(max_delta, max(abs(pa[k] - pb[k]) for k in pa))
            if a['tokens'] != b['tokens']:
                ta, tb = str(a['tokens'][0]), str(b['tokens'][0])
                margin = max(abs(pa[ta] - pa[tb]), abs(pb[ta] - pb[tb]))
                divergences.append({'index': index, 'top_token_margin': margin})
                assert margin <= 0.015, divergences[-1]
        assert max_delta <= 0.03, max_delta
        print('TEACHER_FORCED', len(cold), 'max_all_vocab_logprob_delta', max_delta,
              'near_tie_argmax_changes', divergences, flush=True)
    elif not args.record_divergences:
        assert [r['tokens'] for r in cold] == [r['tokens'] for r in warm], 'reload token mismatch'
    else:
        print('AUTOREGRESSIVE_DIVERGENCES', [i for i, (a, b) in
              enumerate(zip(cold, warm)) if a['tokens'] != b['tokens']], flush=True)
    if args.offload:
        def load_bytes(metrics):
            return sum(m.get('value', 0) for m in metrics
                       if m['name'].startswith('vllm:kv_offload_load_bytes'))
        assert load_bytes(after) > load_bytes(before), 'no measured CPU-to-GPU reload'
        assert any(r['cached'] for r in warm), 'no cache hits after GPU cache reset'
        if args.expect_external:
            assert load_bytes(before) > load_bytes(initial_metrics), 'no cross-process reload'
            assert any(r['cached'] for r in cold), 'no cross-process cache hits'
    print('PASS', args.tp, args.offload, worker_info, flush=True)


if __name__ == '__main__':
    main()
