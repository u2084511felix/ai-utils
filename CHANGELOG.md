# Changelog

All notable changes to `ai-utils` will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.1.0]

### Added

- Pre-commit hooks for TOML/YAML validation, merge conflict detection, and the
  full pytest suite, with a development dependency and setup instructions.
- Offline pytest suite covering models, pricing, budgets, clients, generation,
  embeddings, file patches, prompts, schema helpers, and audio requests.
- Mocked provider clients, blocked network connections, and temporary-file
  fixtures so tests run without API credentials or external services.
- Regression tests documenting existing defects as ordinary failures.
- Coverage-guided tests for structured-output orchestration, schema imports,
  refactoring delegation, client logging and failures, web search, and patch
  dispatch and recovery. Additional regression tests capture dropped request
  options, incorrect client boundaries, and failed patches recorded as successful.
- Package-only branch coverage configuration so reports exclude test code.
- Regression coverage for Responses API input forwarding, retained output and
  token usage, patch application and undo through client replies, and provider
  error propagation.
- Pytest configuration, test and coverage instructions in the README, and ignore
  rules for coverage artifacts.

### Changed

- Split tests that compare tuples of unrelated results into separate tests.
  Each new test checks one result with one assertion.
- Revise the pytest suite to use public calls, fixture setup, one assertion per
  test, named expected values, and GIVEN/WHEN/THEN test descriptions.
- Split tests with multiple actions into separate cases. Move prior calls for
  conversation, budget, and patch tests into fixtures.
- Removed all 16 `xfail` decorators so the 24 affected test cases report normal
  failures and block pre-commit until the underlying defects are fixed.

### Fixed

- Correct the model-switch test to request a non-reasoning model after a
  reasoning model.
- Model enum construction so supported model names and legacy aliases are valid
  string enum members.
- Pydantic module configuration to accept SDK client types.
- `Generate` initialization by removing the incompatible dataclass decorator
  from the Pydantic model.
- Token-cost arithmetic and currency conversion, preserving per-million price
  units, optional cached-input rates, and discounts.
- Client parsing of empty choices and Responses API text and token usage,
  retaining the provider response for downstream patch operations.
- Responses API tool forwarding and support for the `input` keyword, rejecting
  requests that also supply `prompt`.
- Failed patches incorrectly recorded as successful, and patch handling of
  wrapped client replies and provider errors.
- Stale reasoning settings when switching to a non-reasoning model.
- Token counting with a named encoding; updated the tokenizer fixture to keep
  the corrected lookup offline.
- Default prompt template lookup and explicit template path handling.
- Speech preset initialization while preserving explicit model overrides.
- WebSearch construction, tool overrides, and deprecated Pydantic serialization.
- Structured-output defaults for token limits, client selection, and system
  instructions.

## [Unreleased]

### Changed

- Replace the budget API with a public protocol and one factory. Add shared
  token capacity, cancellation, and read-only spend values.
- Track confirmed cost and unresolved estimates separately. Reject foreign
  reservations and repeated settlement.
- Add one library retry policy for synchronous and asynchronous client calls.
  Disable SDK retries and reject injected clients that permit nested retries.
- Validate named JSON Schema wrappers and Pydantic models before history changes.
  Support JSON Schema Draft 2020-12 and Draft 7.
- Correct cached-input pricing and preserve cached usage during reconciliation.
- Require explicit usage on successful replies. Retain parsed Pydantic output.
- Repair schema-generation response handling and structured-output keyword calls.
- Remove import-time client creation.

### Added

- Add migration contract notes and offline checks for budget windows, retry
  ownership, cancellation, schema validation, and reported usage.
- Add a separate paid OpenAI suite. Require the explicit `--openai-live` flag.
  Keep these checks outside normal tests and pre-commit.

### Fixed

- Restore OpenAI, Google, and Brave selection in `set_vendor`. Remove the
  OpenAI-only restriction from the client factory.
- Update budget and retry fixtures to replace the pure wait functions and clock.
  Remove calls to the deleted waiter and clock constructor arguments.
- Reject boolean token counts before integer matching. Permit zero-second waits
  and check cancellation without a sleep.
- Update wait function notes and remove deleted waiter arguments from docstrings.
- Fix test discovery after the retry policy became a protocol. Keep Pydantic
  from applying a runtime instance check to the client policy field.
- Use the public retry policy factory in tests and examples. Reject boolean
  retry counts and non-finite wait limits in the Pydantic policy model.
