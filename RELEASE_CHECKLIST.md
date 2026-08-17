# PowderLine v0.1.0 public-release checklist

Maintainer steps to take PowderLine public. Work through in order; items
marked **[admin]** need GitHub org permissions beyond repo creation. Delete
this file via a normal PR once the release is done.

## 1. Retire the old private repo

- [ ] **[admin]** Rename `NSLS2/PowderLine` → `NSLS2/PowderLineV0` (requires an
      org admin; repo *renames* are not covered by repo-creation rights).
- [ ] Push a final commit to `PowderLineV0`'s README:
      *"This is an archived version of PowderLine pre public release.
      Archived \<date\>."* Optionally archive the repo in GitHub settings
      (read-only).
- [ ] Delete the `AdamCorrao/PowderLine` fork (everything is preserved in
      `PowderLineV0`). Retire or re-point any local clones whose `origin`
      targets it.

## 2. Create and push the new repos (creation rights suffice)

- [ ] Create **private** `NSLS2/PowderLine` (empty — no README/license/
      gitignore, so the push is a clean fast-forward).
- [ ] From the prepared snapshot clone (`PowderLine-public`):
      `git remote add origin git@github.com:NSLS2/PowderLine.git`
      then `git push -u origin main` and `git push origin v0.1.0`.
- [ ] Create **private** `NSLS2/PowderLine-devkit`; push the prepared devkit
      staging repo the same way. Fork it to your account if you want a
      personal remote.
- [ ] In your PowderLine working tree, wire the devkit as the gitignored
      private workspace: `git clone git@github.com:NSLS2/PowderLine-devkit.git _dev`
- [ ] Enable the pre-commit guard in every clone:
      `git config core.hooksPath scripts/hooks`

## 3. Pre-flight on the new `NSLS2/PowderLine` (still private)

- [ ] Confirm the LICENSE copyright line with BNL (institutional/BSA
      requirements for lab-developed software), then align the
      `docs/conf.py` `copyright` string with the answer.
- [ ] `pixi install && pixi run test` from a fresh clone of the new remote
      (confirms nothing depended on the old clone's local state).
- [ ] Enable branch protection on `main` (PRs required; no force pushes).
- [ ] **Publishing rule** (document for all maintainers): publishing is a
      fast-forward push / PR of a clean `main` — NEVER merge a branch whose
      history contains data or `_dev/` material ("strip-and-merge" leaks
      ancestry). Data and `_dev/` must never be `git add`-ed at all.

## 4. Docs

- [ ] Import the project on readthedocs.org — slug must be **`powderline`**
      (the README badge and doc links hardcode it). Point RTD at the new
      `NSLS2/PowderLine`; the build config is already in `.readthedocs.yaml`.
- [ ] Verify the RTD build succeeds and pages render (Developer section should
      show DEVELOPMENT, known_issues, regression-tolerance,
      cross-platform-guide).
- [ ] Remove the README note "the hosted documentation goes live when
      PowderLine becomes public" once RTD is live.

## 5. Paper linkage

- [ ] When the PowderLine arXiv preprint is posted: fill the
      `REPLACE-BEFORE-PUBLIC` `identifiers:` block in `CITATION.cff` and add
      the link to the README.
- [ ] DRX_33 data citation (`doi:10.26434/chemrxiv.15003271/v1`) is already in
      the example DESCRIPTIONs — update it to the final published-article DOI
      when it appears.

## 6. Go public

- [ ] Flip `NSLS2/PowderLine` to **public**.
- [ ] Create the GitHub Release for tag `v0.1.0` (title "PowderLine v0.1.0";
      notes: what PowderLine does, engines, schema 0.26.0, link to docs).
- [ ] Confirm the RTD badge in the README renders green.
- [ ] Delete this `RELEASE_CHECKLIST.md` via a normal PR.

## 7. Post-release (optional / when ready)

- [ ] CI (GitHub Actions), options by cost: (a) GSAS-II-free subset (schema +
      TOPAS tests) on plain pip, cheap; (b) full linux-64 `pixi run test` via
      `prefix-dev/setup-pixi` with `cache: true` keyed on `pixi.lock` —
      recommended target; (c) a win-64 matrix leg (tests need no `tc.exe`).
- [ ] `CHANGELOG.md` (Keep-a-Changelog), starting from v0.1.0.
      `docs/SCHEMA_HISTORY.md` remains the recipe-schema changelog — the two
      version streams stay independent (app semver vs. schema version).
- [ ] Lint: adopt ruff minimally (pixi dep + `lint` task,
      `select = ["E4","E7","E9","F","I","UP"]`, skip E501).
- [ ] `CODE_OF_CONDUCT.md` (Contributor Covenant 2.1) with a real, monitored
      contact email.
- [ ] README badges beyond RTD: license + Python version (static shields.io);
      CI badge only if/when CI lands.
- [ ] Versioning reminders for future releases: bump
      `src/powderline/__init__.py` `__version__` (single source for hatch +
      Sphinx), and mirror it in `pixi.toml` `[workspace] version` and
      `CITATION.cff` `version`; tag `vX.Y.Z`. The schema version
      (`EXPECTED_SCHEMA_VERSION` in `schema.py`) evolves independently — never
      couple the two.
- [ ] Deferred with rationale: `[project.scripts]` console entry points
      (revisit at first PyPI release), SECURITY.md (server binds localhost),
      mypy (GSAS-II untyped), Zenodo DOI (omitted by decision; can be minted
      later from any tagged release if a journal requires an archived
      software DOI).
