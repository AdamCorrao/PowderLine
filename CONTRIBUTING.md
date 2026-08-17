# Contributing to PowderLine

Thanks for your interest! This page is the map; the detail lives in the linked
guides.

## Dev setup

```bash
git clone https://github.com/nsls2/PowderLine.git
cd PowderLine
pixi install        # installs everything, including GSAS-II
pixi run test       # full suite — run it before and after your change
```

[pixi](https://pixi.sh) is the only tool you install yourself. See the
[README](README.md) for platforms and the [Quickstart](docs/quickstart.md) for
a first refinement run.

## Where things are

- **[docs/DEVELOPMENT.md](docs/DEVELOPMENT.md)** — developer guide:
  architecture, domain concepts, GSAS-II integration, how to add a refinement
  parameter, and a `## Contributing` section with expectations for changes.
- **[docs/known_issues.md](docs/known_issues.md)** — the known-issues register
  (`KI-NN` ids, referenced from code and docs): check it before "fixing" a
  quirk. See also [docs/regression-tolerance.md](docs/regression-tolerance.md)
  (why numeric comparisons are tolerance-based) and
  [docs/cross-platform-guide.md](docs/cross-platform-guide.md).
- **[examples/CONTRIBUTING_EXAMPLES.md](examples/CONTRIBUTING_EXAMPLES.md)** —
  how to contribute a new example recipe (examples drive schema evolution).

## Ground rules

- Branch off `main`; keep commits reviewable and self-contained.
- Enable the pre-commit guard once per clone:
  `git config core.hooksPath scripts/hooks`. It blocks accidental commits of
  data files outside the sanctioned dirs, `.gpx` binaries, and the private
  `_dev/` workspace.
- Run `pixi run test` before opening a PR. Some regression cases need a working
  GSAS-II refinement — compare failures against a baseline run, don't assume
  they're yours.
- After editing `src/powderline/kicker.py`: run `pixi run update-code-hash` and
  `pixi run gsas-server restart`.
- `examples/**/output/` is tracked on purpose (regression references) — don't
  regenerate it casually.
- Questions and bug reports: open a GitHub issue.
