# Changelog

All notable changes to `ai-utils` will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added

- Offline pytest suite covering models, pricing, budgets, clients, generation,
  embeddings, file patches, prompts, schema helpers, and audio requests.
- Mocked provider clients, blocked network connections, and temporary-file
  fixtures so tests run without API credentials or external services.
- Strict expected-failure tests documenting existing defects. Validation reports
  180 passing tests, 15 expected failures, and 84% statement coverage.
- Pytest configuration, test and coverage instructions in the README, and ignore
  rules for coverage artifacts.

### Fixed

- Model enum construction so supported model names and legacy aliases are valid
  string enum members.
- Pydantic module configuration to accept SDK client types.
- `Generate` initialization by removing the incompatible dataclass decorator
  from the Pydantic model.
