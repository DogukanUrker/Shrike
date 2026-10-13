# Independent check: BenchKit

Two more benchmarks, run with [BenchKit](https://github.com/DogukanUrker/BenchKit) (a different harness, different prompts,
none of this data was used in healing). 32 parallel requests on the same stock llama-server, RTX 3060.

| file | code-verifier acc | agree with d1-3B | GPQA acc | agree with d1-3B | dec/s (32 parallel) |
|---|---|---|---|---|---|
| d1-3B Q8_0 (Liquid) | 51.8% | 100% | 33.8% | 100% | 12.6 / 16.8 |
| Shrike-3B-IQ4_XS | 51.8% | 98.4% | 29.8% | 86.9% | 13.6 / 18.4 |
| Shrike-3B-IQ2_M | 52.4% | 94.6% | 32.8% | 68.2% | 11.2 / 15.3 |
| Shrike-3B-IQ2_XXS | 49.6% | 92.2% | 28.3% | 44.9% | 12.8 / 17.3 |
| Shrike-3B-IQ1_S | 49.4% | 84.4% | 23.2% | 32.3% | 12.1 / 16.6 |

- **Agreement on code-verifier matches our evals** (98 / 95 / 92 / 84% vs 98 / 94 / 91 / 86%) on a domain we never touched.
- **On hard reasoning, d1-3B itself is near chance:** code-verifier is balanced yes/no (50%) and d1 answers "yes" 93% of the
  time; GPQA has 4 options (25%). When the original is guessing, its answers flip easily, so agreement drops (GPQA) and the
  accuracy differences between files are noise (IQ2_M vs IQ4_XS: paired p = 0.39 on GPQA, 0.73 on code-verifier).
  Use d1 / Shrike for classification, routing and images, not for graduate-level reasoning.
- **On a GPU, lower bits are smaller, not faster.** Decisions are prefill only, which is compute-bound when batched, and
  the 1-2 bit kernels cost more compute to unpack. IQ4_XS is the fastest file; the win of the small files is memory.

