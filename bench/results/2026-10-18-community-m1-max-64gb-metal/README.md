# Community benchmark on Apple M1 Max 64 GB (Metal streaming engine)

Measured on 2026-10-18 by DarkJoll. What was tested: a settings/tuning matrix on the
Metal streaming engine at a short prompt (~160 tokens) and a 35,778-token prompt, on
one machine and one model. Main finding: on this machine the tuning knobs that target
expert misses (`STRATA_ROUTE_TAIL_SKIP`), KV bandwidth (`--kv k8v4`) and prompt chunk
size (`STRATA_METAL_PROMPT_CHUNK`) are all **neutral** (within run-to-run noise),
because the model's experts are already resident in the arena. Main limitation: single
machine, single model/quant, no recall/needle test, no 4K/32K/128K standard sweep, no
time-to-first-token capture.

## Hardware and software

- GPU and VRAM: Apple M1 Max integrated GPU, no discrete VRAM (unified memory).
  CPU: Apple M1 Max, 10 cores. Installed RAM: 64 GiB unified (= 68.7 GB decimal, the
  figure the engine prints at start-up). Storage: APPLE SSD AP1024R, 1 TB (experts are
  streamed from this SSD). PCIe link: not applicable (SoC).
- OS: macOS 27.0 (build 26A428). Driver/CUDA/ROCm: not applicable (Metal backend).
- Strata commit: `477ac13c` (v0.1.45-metal base) plus two local commits:
  `ad92d055` (Metal prompt path no longer kills the engine on prompts > 64 tokens) and
  `eddd769b` (Metal build fix). Engine version: `0.1.45-metal`. Source build
  (`build-metal`, Metal backend).
- Background workloads and power limits: the server was idle between runs; a separate
  desktop app sent some unrelated requests during the session (those are excluded from
  the per-run numbers below, which come from the engine's own request lines).

## Model and configuration

- Model: Qwen3.8-Flash-Next, IQ3_S quantization, GGUF split into two files (the expert
  tensors are streamed in place). Custom native-experts pack and `expert-profile.bin`.
- Vision encoder: enabled (`--vision`). MTP draft layer from `mtp-q2_0.gguf` (rt).
- Context: 262,144 tokens. KV: default (the `k8v4` variant was tested, see results).
  Expert cache: `auto` (the start-up line reports 17,940 slots, but on Metal those
  device copies are "given back"; the real resident set is the arena: 12,945 slots,
  28.44 GB, out of 24,576 ranked experts). Prefill: `auto`, prompt chunk 2048 (4096
  tested, see results). `--spec 2` (the minimum: native IQ packs reject `--spec 1` at
  start-up), `--mtp`, `--mmap-experts`, `--vram-reserve-mib 1728`.

```text
strata --serve --pack <pack>/iq3_s --native <model>-00001-of-00002.gguf \
  --ple-gguf <model>-00002-of-00002.gguf --expert-profile <repo>/data/expert-profile.bin \
  --expert-cache auto --prefill auto --spec 2 --suffix-draft 0 --max-context 262144 \
  --mmap-experts --mtp <mtp>/rt --vram-reserve-mib 1728 --vision
env: STRATA_METAL_STREAM=1, STRATA_METAL_PROMPT_CHUNK=2048
```

## Method

Two workloads, sent through the HTTP `/v1/chat/completions` API with `temperature 0`
(deterministic; the engine's "drafts accepted" count was identical, 190 of 322, across
every configuration, so the input is reproducible):

- **short**: a ~160-token Italian story prompt, output capped at 1,024 tokens. The
  engine serves it in two windows, so one request logs two prompt lines (160 tokens,
  then a 690-token continuation that reuses the 160-token prefix); ~1,007 tokens
  generated in total. The decode figure below is the warm continuation line.
- **bigctx**: a 35,778-token prompt, 512 tokens generated. Run 1 after each restart is
  cold (empty KV, experts re-read); runs 2-3 reuse the 35,773-token prefix.

Three runs per configuration. Decode tok/s and prompt tok/s are the engine's own
figures from its request line (`generated ... (N tok/s)` and `read ... (N tok/s)`),
not generated-tokens divided by total request time. Cold prefill is the fresh-prompt
line; decode is reported for the warm (prefix-reused) runs. Each configuration was a
separate engine restart with one setting changed from the base.

## Results

Decode tok/s (warm, 512 generated at 35,778 tokens) and cold prompt tok/s:

| Configuration | Prompt tok/s (cold, 35,778 fresh) | Decode tok/s (warm, 35,778) | Decode tok/s (warm, short ~1k) |
| --- | ---: | ---: | ---: |
| base: spec 2 + MTP + chunk 2048 + reserve 1728, KV default, ROUTE_TAIL_SKIP default 7 | 274.3 | 21.4 / 21.4 | 21.0-21.2 (median 21.2) |
| + `--kv k8v4` | 273.8 | 21.4 / 21.5 | 21.2-21.5 (median 21.3) |
| + `STRATA_ROUTE_TAIL_SKIP=0` | 244.8 / 276.4 | 21.3-21.5 | 20.1-21.4 (median 21.1) |
| + `STRATA_ROUTE_TAIL_SKIP=10` | 242.3 | 21.1 / 21.1 | 20.9-21.2 (median 21.0) |
| + `STRATA_METAL_PROMPT_CHUNK=4096` | 250.9 | 21.1 / 21.2 | not measured |

Every configuration lands in the same band: decode ~21 tok/s at 35,778 tokens and ~21
tok/s at ~1k, cold prompt ~245-276 tok/s. The spread between configurations is the same
as the run-to-run spread within one configuration, so all three knobs are neutral here.

Why they are neutral: the arena holds 12,945 of the 24,576 ranked experts resident
(28.44 GB), so there are no expert misses for `ROUTE_TAIL_SKIP` to skip, and the draft
acceptance is identical across runs. At 35,778 tokens the cold prompt is bound by
streaming experts from the SSD, not by the matrix-unit chunk size, so 2048 vs 4096 does
not move it.

Per-run engine lines are in `results.json`; the start-up block (arena, footprint, RAM)
is in `engine-log-excerpt.txt`.

## Correctness and limitations

- No recall/needle test, no HumanEval, no tool-call or image check: this is a speed
  report only, it does not establish answer quality.
- No time-to-first-token and no total-latency capture; no concurrency (`--batch` off).
- No 4,096 / 32,768 / 128,000 standard sweep; only ~160 and 35,778 tokens were run.
- One machine, one model/quant (IQ3_S), one engine build (0.1.45-metal with the two
  local commits above). Numbers are not comparable to other reports' headline figures.
- Not tested on this machine (deliberately left alone): `STRATA_METAL_PROMPT_EXPERTS=v3`
  and `STRATA_METAL_ROUTE=1` (not used on M1 Max), the arena `LEAVE_SHARE` plateau, and
  `--spec 1` (rejected at start-up: native IQ packs need `--spec >= 2`).
- The "0 MiB VRAM free" warning reappeared at 120k+ context with `--vram-reserve-mib
  1728` (the log suggests 2240); not chased in this report, flagged for follow-up.
