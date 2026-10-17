class WorkerProbe:
    def describe_kv_worker(self):
        from vllm.distributed.kv_transfer import get_kv_transfer_group
        from vllm.distributed.parallel_state import get_tensor_model_parallel_rank
        groups = self.model_runner.kv_cache_config.kv_cache_groups
        result = {'rank': get_tensor_model_parallel_rank(),
                  'specs': [type(g.kv_cache_spec).__name__ for g in groups],
                  'spec_details': [repr(g.kv_cache_spec) for g in groups],
                  'total_kv_heads': self.vllm_config.model_config.hf_text_config.num_key_value_heads,
                  'tp_size': self.vllm_config.parallel_config.tensor_parallel_size}
        if self.vllm_config.kv_transfer_config is not None:
            spec = get_kv_transfer_group().connector_worker.spec
            result.update(canonical_layout=spec.config.canonical_layout,
                          all_layers_portable=spec.all_layers_portable)
        return result
