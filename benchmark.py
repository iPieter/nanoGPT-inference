"""
Benchmark inference engines.

Measures:
- Time to first token (TTFT): prefill latency
- Inter-token latency (ITL): average decode latency per token
- Throughput: tokens/sec (prefill and decode phases)
- Batch scaling: throughput vs batch size

Usage:
    python benchmark.py --engine baseline
    python benchmark.py --engines baseline kv_cache --compare
    python benchmark.py --batch_sizes 1 2 4 8 --plot
"""

import os
import time
import json
import random
import torch
import numpy as np
from contextlib import nullcontext
from model import GPTConfig, GPT
import inference

# -----------------------------------------------------------------------------
# Configuration
engines = inference.list_engines().keys()  # which engines to benchmark (can be multiple)
#engines = ['sampling']
compare = False  # compare multiple engines side-by-side
init_from = 'gpt2'  # 'resume' or gpt2 variant
out_dir = 'out'
prompt_lengths = [32]  # list of prompt lengths to test
batch_sizes = [2**i for i in range(10)]  # list of batch sizes to test, default 1 to 512
max_new_tokens = 100  # tokens to generate per sample
num_runs = 5  # number of runs to average over
warmup_runs = 10  # warmup runs (not counted)
device = 'cuda'  # 'cpu', 'cuda', 'cuda:0', etc.
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
compile = False  # compile model for speed
seed = 1337
randomize_order = True  # randomize engine execution order to avoid bias
cooldown_seconds = 10  # seconds to wait between engines (e.g., 180 for 3 min)
save_results = True  # save results to JSON
results_dir = 'benchmark_results'  # where to save results
plot_dir = 'plots'  # where to save plots
profile_engines = []  # list of engines to profile (e.g., ['fp8', 'sampling'])
profile_dir = 'profiles'  # where to save profiler traces
exec(open('configurator.py').read())
# -----------------------------------------------------------------------------

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

device_type = 'cuda' if 'cuda' in device else 'cpu'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

print("="*80)
print("nanoGPT Inference Benchmark")
print("="*80)

# Load model
print(f"\nLoading model: {init_from}")
if init_from == 'resume':
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    gptconf = GPTConfig(**checkpoint['model_args'])
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
elif init_from.startswith('gpt2'):
    model = GPT.from_pretrained(init_from, dict(dropout=0.0))
elif init_from.startswith('empty-'):
    model = GPT.init_empty_weights(init_from, dict(dropout=0.0))



model.eval()
model.to(device)

if compile:
    print("Compiling model with torch.compile()...")
    model = torch.compile(model)

print(f"Model config: {model.config.n_layer}L, {model.config.n_head}H, {model.config.n_embd}D")
print(f"Device: {device}, dtype: {dtype}")
print(f"Runs per config: {num_runs} (+ {warmup_runs} warmup)")
print()


class BenchmarkResult:
    """Container for benchmark measurements."""

    def __init__(self, engine_name, prompt_length, batch_size, max_new_tokens):
        self.engine_name = engine_name
        self.prompt_length = prompt_length
        self.batch_size = batch_size
        self.max_new_tokens = max_new_tokens

        # Raw measurements (per run)
        self.ttft_samples = []  # time to first token
        self.itl_samples = []   # inter-token latencies
        self.total_time_samples = []
        self.memory_allocated = []
        self.memory_reserved = []

    def add_run(self, ttft, itl, total_time, mem_alloc=0, mem_res=0):
        """Add measurements from one run."""
        self.ttft_samples.append(ttft)
        self.itl_samples.append(itl)
        self.total_time_samples.append(total_time)
        self.memory_allocated.append(mem_alloc)
        self.memory_reserved.append(mem_res)

    def compute_stats(self):
        """Compute aggregate statistics."""
        return {
            'engine': self.engine_name,
            'prompt_length': self.prompt_length,
            'batch_size': self.batch_size,
            'max_new_tokens': self.max_new_tokens,

            # Time to first token (ms)
            'ttft_mean': np.mean(self.ttft_samples) * 1000,
            'ttft_std': np.std(self.ttft_samples) * 1000,
            'ttft_min': np.min(self.ttft_samples) * 1000,
            'ttft_max': np.max(self.ttft_samples) * 1000,

            # Inter-token latency (ms)
            'itl_mean': np.mean(self.itl_samples) * 1000,
            'itl_std': np.std(self.itl_samples) * 1000,

            # Total time (s)
            'total_time_mean': np.mean(self.total_time_samples),
            'total_time_std': np.std(self.total_time_samples),

            # Throughput
            'prefill_tokens_per_sec': self.prompt_length * self.batch_size / np.mean(self.ttft_samples),
            'decode_tokens_per_sec': self.batch_size / np.mean(self.itl_samples),
            'overall_tokens_per_sec': (self.prompt_length + self.max_new_tokens) * self.batch_size / np.mean(self.total_time_samples),

            # Memory (GB)
            'memory_allocated_gb': np.mean(self.memory_allocated) if self.memory_allocated else 0,
            'memory_reserved_gb': np.mean(self.memory_reserved) if self.memory_reserved else 0,
        }

    def print_summary(self):
        """Print human-readable summary."""
        stats = self.compute_stats()
        print(f"\n{'='*80}")
        print(f"Engine: {stats['engine']}")
        print(f"Config: prompt={stats['prompt_length']}, batch={stats['batch_size']}, "
              f"generate={stats['max_new_tokens']}")
        print(f"{'='*80}")
        print(f"Time to first token:    {stats['ttft_mean']:7.2f} ± {stats['ttft_std']:.2f} ms")
        print(f"Inter-token latency:    {stats['itl_mean']:7.2f} ± {stats['itl_std']:.2f} ms")
        print(f"Total time:             {stats['total_time_mean']:7.3f} ± {stats['total_time_std']:.3f} s")
        print(f"{'-'*80}")
        print(f"Prefill throughput:     {stats['prefill_tokens_per_sec']:7.1f} tokens/s")
        print(f"Decode throughput:      {stats['decode_tokens_per_sec']:7.1f} tokens/s")
        print(f"Overall throughput:     {stats['overall_tokens_per_sec']:7.1f} tokens/s")
        if stats['memory_allocated_gb'] > 0:
            print(f"{'-'*80}")
            print(f"Memory allocated:       {stats['memory_allocated_gb']:7.2f} GB")
            print(f"Memory reserved:        {stats['memory_reserved_gb']:7.2f} GB")


def benchmark_engine(engine_name, prompt_length, batch_size, max_new_tokens, num_runs, warmup_runs):
    """Benchmark a single engine configuration."""

    # Create inference engine
    Engine = inference.get_engine(engine_name)
    engine = Engine(model)

    # Create prompt (batch_size x prompt_length)
    prompt = torch.randint(
        0, model.config.vocab_size,
        (batch_size, prompt_length),
        device=device,
        dtype=torch.long
    )

    print("prompt", prompt.shape)

    result = BenchmarkResult(engine_name, prompt_length, batch_size, max_new_tokens)

    # Warmup (FP8 needs more warmup for torch.compile)
    print(f"  Warming up ({warmup_runs} runs)...", end='', flush=True)
    with torch.no_grad():
        with ctx:
            for _ in range(warmup_runs):
                _ = engine.generate(prompt, max_new_tokens, temperature=1.0, top_k=20, top_p=0.9)
                if device_type == 'cuda':
                    torch.cuda.synchronize()
    print(" done")

    # Benchmark runs
    print(f"  Running benchmark ({num_runs} runs)...", end='', flush=True)

    # Profiler setup (will be used only on first run if enabled)
    profiler = None
    profile_path = None
    if engine_name in profile_engines:
        os.makedirs(profile_dir, exist_ok=True)
        profile_path = os.path.join(profile_dir, f"profile_{engine_name}_bs{batch_size}.json")
        print(f" (profiling first run)")

    for run in range(num_runs):
        # Enable profiler only for first run
        if run == 0 and engine_name in profile_engines:
            profiler = torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                record_shapes=True,
                profile_memory=True,
                with_stack=True,
                with_flops=True,
            )
            profiler.__enter__()
        if device_type == 'cuda':
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()

        with torch.no_grad():
            with ctx:
                t_start = time.perf_counter()

                # Generate tokens
                output = engine.generate(prompt, max_new_tokens, temperature=1.0, top_k=20, top_p=0.9)

                if device_type == 'cuda':
                    torch.cuda.synchronize()
                t_end = time.perf_counter()

        total_time = t_end - t_start

        # Get timing breakdown from engine
        engine_stats = engine.get_stats()

        if 'ttft' in engine_stats and 'itl' in engine_stats:
            # Engine provides accurate per-token measurements
            ttft = engine_stats['ttft']
            itl = engine_stats['itl']
        elif 'prefill_time' in engine_stats and 'decode_time' in engine_stats:
            # Engine provides phase timings
            ttft = engine_stats['prefill_time']
            decode_time = engine_stats['decode_time']
            itl = decode_time / max_new_tokens if max_new_tokens > 0 else 0
        else:
            # Engine doesn't report timing breakdown - this shouldn't happen
            # if engines follow the base class contract
            print(f"\nWarning: Engine {engine_name} doesn't report timing stats!")
            ttft = 0.0
            itl = 0.0

        # Memory
        mem_alloc = torch.cuda.max_memory_allocated() / 1e9 if device_type == 'cuda' else 0
        mem_res = torch.cuda.max_memory_reserved() / 1e9 if device_type == 'cuda' else 0

        result.add_run(ttft, itl, total_time, mem_alloc, mem_res)

        # Close profiler after first run
        if run == 0 and profiler is not None:
            profiler.__exit__(None, None, None)

    print(" done")

    # Save and print profiler results if enabled
    if profiler is not None:
        profiler.export_chrome_trace(profile_path)
        print(f"  Profiler trace saved to {profile_path}")

        # Print summary
        print("\n" + "="*80)
        print(f"Top 10 GPU time consumers for {engine_name}:")
        print("="*80)
        print(profiler.key_averages().table(sort_by="cuda_time_total", row_limit=10))
        print("="*80 + "\n")

    return result


# Run benchmarks
all_results = []

for prompt_length in prompt_lengths:
    for batch_size in batch_sizes:
        print(f"\n{'='*80}")
        print(f"Batch size: {batch_size}, Prompt length: {prompt_length}")
        print(f"{'='*80}")

        # Randomize engine order for each batch size to avoid bias
        engine_list = list(engines)
        if randomize_order:
            random.shuffle(engine_list)
            print(f"Randomized engine order: {', '.join(engine_list)}")

        for idx, engine_name in enumerate(engine_list):
            print(f"\n  Benchmarking engine: {engine_name}")

            try:
                result = benchmark_engine(
                    engine_name,
                    prompt_length,
                    batch_size,
                    max_new_tokens,
                    num_runs,
                    warmup_runs
                )

                result.print_summary()
                all_results.append(result)

            except torch.OutOfMemoryError as e:
                print(f"\n  ERROR: Engine {engine_name} ran out of memory!")
                print(f"  {str(e)}")
                print(f"  Skipping this engine and continuing...\n")
                # Clear CUDA cache to free up memory for next engine
                if device_type == 'cuda':
                    torch.cuda.empty_cache()
                continue
            except Exception as e:
                print(f"\n  ERROR: Engine {engine_name} failed with exception!")
                print(f"  {type(e).__name__}: {str(e)}")
                print(f"  Skipping this engine and continuing...\n")
                # Clear CUDA cache just in case
                if device_type == 'cuda':
                    torch.cuda.empty_cache()
                continue

            # Cooldown between engines to let GPU temperature stabilize
            if cooldown_seconds > 0:
                print(f"\n  Cooling down for {cooldown_seconds}s before next engine...")
                time.sleep(cooldown_seconds)

# Save results
results_file = None
if save_results:
    os.makedirs(results_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(results_dir, f"benchmark_{timestamp}.json")

    results_data = {
        'config': {
            'model': init_from,
            'device': device,
            'dtype': dtype,
            'compile': compile,
            'num_runs': num_runs,
            'warmup_runs': warmup_runs,
        },
        'results': [r.compute_stats() for r in all_results]
    }

    with open(results_file, 'w') as f:
        json.dump(results_data, f, indent=2)

    print(f"\n{'='*80}")
    print(f"Results saved to: {results_file}")
    print(f"{'='*80}")

# Comparison table (if multiple engines)
if compare and len(engines) > 1:
    print(f"\n{'='*80}")
    print("COMPARISON")
    print(f"{'='*80}")

    # Group by (prompt_length, batch_size)
    configs = {}
    for result in all_results:
        key = (result.prompt_length, result.batch_size)
        if key not in configs:
            configs[key] = []
        configs[key].append(result)

    for (prompt_len, batch_sz), results in configs.items():
        print(f"\nPrompt length: {prompt_len}, Batch size: {batch_sz}")
        print(f"{'-'*80}")
        print(f"{'Engine':<20} {'TTFT (ms)':<15} {'ITL (ms)':<15} {'Decode tok/s':<15}")
        print(f"{'-'*80}")

        for result in results:
            stats = result.compute_stats()
            print(f"{stats['engine']:<20} "
                  f"{stats['ttft_mean']:>7.2f} ± {stats['ttft_std']:<4.2f} "
                  f"{stats['itl_mean']:>7.2f} ± {stats['itl_std']:<4.2f} "
                  f"{stats['decode_tokens_per_sec']:>10.1f}")

        # Calculate speedups vs first engine
        if len(results) > 1:
            baseline_stats = results[0].compute_stats()
            print(f"{'-'*80}")
            print("Speedup vs first engine:")
            for result in results[1:]:
                stats = result.compute_stats()
                ttft_speedup = baseline_stats['ttft_mean'] / stats['ttft_mean']
                itl_speedup = baseline_stats['itl_mean'] / stats['itl_mean']
                throughput_speedup = stats['decode_tokens_per_sec'] / baseline_stats['decode_tokens_per_sec']
                print(f"  {stats['engine']:<18} "
                      f"TTFT: {ttft_speedup:.2f}x, "
                      f"ITL: {itl_speedup:.2f}x, "
                      f"Throughput: {throughput_speedup:.2f}x")

print(f"\n{'='*80}")
print("Benchmark complete!")
if results_file:
    print(f"\nTo generate plots, run:")
    print(f"  python plot_benchmark.py {results_file}")
    print(f"  or: python plot_benchmark.py --latest")
print(f"{'='*80}\n")
