# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[semantic versioning](https://semver.org/).

## [Unreleased]

### Added
- `POST /v1/generate` with token streaming over server-sent events, served one
  request at a time; a disconnected client frees the model at the next token.
  `llm-serve` runs it with the simulated model paced in real time.
- Project foundation: package layout, CI, contribution and security guides, design doc
  and roadmap.
