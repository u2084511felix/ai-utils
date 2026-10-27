# Changelog

All notable changes to `ai-utils` will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

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

- Removed all 16 `xfail` decorators so the 24 affected test cases report normal
  failures and block pre-commit until the underlying defects are fixed.

### Fixed

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
