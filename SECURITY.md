# Security policy

## Reporting a problem

Please report security issues privately through GitHub's
[security advisories](https://github.com/mohammadazam7/llm-inference-server/security/advisories/new)
rather than in a public issue. Expect an acknowledgement within a few days.

## Scope

The server is a research and teaching system. It has no authentication, and it should not
be exposed to an untrusted network. Model checkpoints are loaded with
`weights_only=True`; only serve checkpoints you produced or trust. Request size and
output length are bounded so one client cannot hold the whole cache pool.
