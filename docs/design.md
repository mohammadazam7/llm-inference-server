# Design: llm-inference-server

Status: draft. Author: Mohammad Azam. Last updated: Nov 16, 2026.

## Context

Generating text takes one forward pass per token. A server that runs each request alone
leaves most of its capacity idle and makes every other client wait. Production servers
batch requests continuously, keep attention state in a managed cache and run reduced
precision weights. This project implements those techniques for small models at a size
where each can be tested and measured on its own.

## Goals

- A naive baseline server and a load-test harness, built first, so every later change
  has a number to beat.
- Continuous batching: requests join and leave a running batch at decoding-step
  boundaries.
- A paged KV cache: fixed-size blocks, a free list, no copying when a sequence grows.
- INT8 quantization with its effect on speed, memory and output quality reported.
- Token streaming over server-sent events, with cancellation.
- Serving checkpoints produced by tiny-llm.

## Non-goals

- GPU kernels or multi-node serving.
- Large models: the largest model served is a few million parameters.
- Authentication, quotas and multi-tenancy beyond a simple request limit.

## Design

**Model runtime.** A narrow interface: given a batch of sequences and their cache
handles, return next-token logits for each. The first implementation wraps a small
PyTorch decoder; the scheduler and cache never import the model directly, so they are
tested with a fake runtime that returns scripted tokens.

**Scheduler.** A loop over decoding steps. At each step it admits waiting requests up to
a batch limit, runs one forward pass for all active sequences, emits one token per
sequence, and retires sequences that reached a stop token or their length limit.
Admission happens every step, which is what makes the batching continuous.

**KV cache.** Keys and values live in fixed-size blocks drawn from one preallocated pool.
A sequence holds a list of block indices. Growing a sequence takes a block from the free
list; finishing returns its blocks. Memory use is bounded by the pool, and a request that
cannot get a block waits instead of failing the others.

**Quantization.** Post-training INT8 for linear layers, per-channel scales, with the
float model kept as the reference. The report compares tokens per second, memory, and
the change in validation loss and in greedy outputs.

**API.** A completions endpoint with a streaming mode (server-sent events). A client
disconnect cancels the request and frees its cache blocks at the next step.

**Load-test harness.** Replays a schedule of arrivals (constant rate, bursts, Poisson)
with seeded prompt and output lengths, and reports time to first token, tokens per
second, and p50 and p95 request latency. The same schedule runs against the baseline and
each improvement.

## Alternatives considered

- **Static batching** (wait for a full batch, run it to completion). Simple, but a short
  request waits for the longest one in its batch; it is kept as a second baseline.
- **A contiguous cache per sequence.** Easy to write, but growing a sequence means
  reallocating and copying, and memory fragments under mixed lengths.
- **Building on an existing serving framework.** Faster to a demo, but the scheduler and
  cache are what this project sets out to show.

## Testing plan

- Scheduler, with a fake runtime: every request gets exactly its tokens in order, a late
  arrival joins without disturbing running sequences, cancellation frees resources,
  and outputs equal the one-at-a-time baseline for the same seeds.
- KV cache: blocks are never shared between live sequences, the free list is conserved,
  and exhaustion makes a request wait.
- Quantization: outputs stay within a stated tolerance of the float model.
- API: the stream is well-formed server-sent events and ends with a final event.

## Open questions

- Whether continuous batching helps on a CPU as much as on a GPU, where a batch is
  nearly free; the first load test answers this, and the README will report it either
  way.
- The block size for the cache: small blocks waste less memory, large blocks mean fewer
  lookups.
