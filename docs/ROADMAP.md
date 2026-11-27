# Roadmap

Target: v1.0 between **Jan 8 and Jan 10, 2027**. Each milestone ends with a pull
request, a tagged release and a changelog entry.

- [x] **M0 (Nov 16): foundation.** Repository, CI, design doc, roadmap.
- [x] **M1 (Nov 17 to 27): baseline and measurement.** The model runtime interface, a
  naive one-request-at-a-time server, token streaming over server-sent events, and the
  load-test harness (time to first token, tokens per second, p50 and p95).
- [x] **M2 (Nov 26): continuous batching.** A step-level scheduler that admits
  and retires requests while the batch runs, tested with a fake runtime and measured
  against the baseline and static batching.
- [ ] **M3 (Dec 10 to 19): paged KV cache.** Fixed-size blocks from a shared pool with a
  free list, bounded memory, and back-pressure when the pool is full.
- [ ] **M4 (Dec 20 to 30): quantization.** INT8 weights with speed, memory and quality
  compared with the float model.
- [ ] **M5 (Dec 31 to Jan 6): tiny-llm checkpoints and the app.** Loading and serving
  models trained by tiny-llm, and the Token Highway web app showing requests moving
  through the scheduler.
- [ ] **M6 (Jan 7 to 10): v1.0.** Results with the hardware stated, runbook, final
  design doc, release.
