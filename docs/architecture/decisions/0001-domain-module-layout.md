# ADR 0001: Domain module layout

Status: Accepted

## Context

RustGPT's model implementation originally used one flat Rust file per domain.
That made `main.rs` own CLI parsing, data loading, model construction, training,
and interaction, while each model file mixed public types with implementation
details. The public module paths are already used by tests and potential
consumers, so restructuring must not change them.

## Decision

Each model responsibility is a domain directory whose `mod.rs` is its facade.
Facades privately declare child category modules and re-export only the existing
public surface. Categories are created only when the responsibility exists:

- `interfaces.rs` owns structs, enums, and traits;
- `logic.rs` owns inherent and trait implementations;
- `constants.rs` owns genuinely shared configuration values;
- `tests.rs` owns private, implementation-sensitive tests.

Sibling implementation modules receive `pub(super)` access only where Rust's
module boundary requires it. Existing public fields remain public. The crate
root keeps compatibility module declarations and re-exports, including the
shared constants.

The binary has private `cli` and `application` facades. `main.rs` is limited to
parsing the mode, loading data, building the model, and running the selected
flow. Public library domains remain available at their original paths, such as
`llm::self_attention::SelfAttention` and `llm::llm::{LLM, Layer}`.

## Consequences

- Readers can enter any domain through one curated facade and then choose its
  interface, implementation, or private tests.
- Internal file locations change, but public Rust paths and fields do not.
- New category files are added only for real responsibilities; empty symmetry
  files and miscellaneous utility modules are prohibited.
- Cross-domain and public-consumer tests remain in top-level `tests/`.
