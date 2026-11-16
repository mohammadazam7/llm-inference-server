# llm-inference-server

An inference server for small transformer language models that shows, with measurements,
what the standard serving techniques buy: continuous batching, a paged key-value cache,
INT8 quantization and token streaming. Every technique is compared against a naive
server that handles one request at a time, on the same hardware and the same load.

> **Status:** design stage. The design doc and roadmap are written; the naive baseline
> and the load-test harness come first, because nothing else can be measured without
> them. No throughput or latency figure appears here until the harness has produced it.

## Why

Serving a language model is a scheduling problem as much as a modelling one. A request
needs many forward passes, requests arrive at different times and finish at different
lengths, and each one carries a growing cache. Handling them one at a time wastes most
of the hardware. The techniques that fix this are well known and usually met inside
large systems where they are hard to isolate. Here each one is a small, tested module
with a before-and-after number.

## How it will work

```mermaid
flowchart LR
    C[Clients] -->|HTTP, SSE| A[API<br/>completions, streaming,<br/>cancellation]
    A --> Q[Request queue]
    Q --> S[Continuous batching scheduler<br/>admit new requests each step,<br/>retire finished ones]
    S --> M[Model runtime<br/>float or INT8]
    K[(Paged KV cache<br/>fixed-size blocks,<br/>free list)] <--> M
    M -->|one token per sequence per step| S
    S -->|tokens as produced| A
    L[Load-test harness<br/>arrival patterns, TTFT,<br/>tokens per second, p50 and p95] --> A
```

| Part | What it does |
|---|---|
| Baseline | One request at a time, start to finish; the number everything else is compared with |
| Continuous batching | New requests join the running batch at the next decoding step instead of waiting for it to finish |
| Paged KV cache | Attention keys and values stored in fixed-size blocks from a shared pool, so sequences of different lengths share memory without copying |
| Quantization | INT8 weights with measured effect on speed, memory and output quality |
| Streaming | Server-sent events, one token at a time, with cancellation when the client disconnects |
| Load-test harness | Replays arrival patterns and reports time to first token, tokens per second and latency percentiles |

In its last milestones the server loads and serves checkpoints trained by
[tiny-llm](https://github.com/mohammadazam7/tiny-llm).

## Honest limits

There is no GPU here. All measurements are on a CPU with small models, and each results
table states the hardware. The relative gains from batching and caching carry over to
larger systems; the absolute numbers do not.

## Documentation

- [Design doc](docs/design.md): goals, non-goals, design, alternatives, testing plan
- [Roadmap](docs/ROADMAP.md): milestones and dates

## Development

```bash
python -m venv .venv && source .venv/bin/activate
make install
make check
```

See [CONTRIBUTING.md](CONTRIBUTING.md) and [SECURITY.md](SECURITY.md).

## License

MIT
