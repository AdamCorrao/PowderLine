"""TOPAS v7 INP generation for PowderLine.

This subpackage translates a validated ``GSASII_Rietveld`` recipe (schema 0.26.0)
into a syntactically valid, semantically faithful TOPAS v7 ``.INP`` + ``.xye``
pair. It imports **zero** GSAS-II code: TOPAS INP generation and the round-trip
must run on Windows, where GSAS-II does not install cleanly.

Public entry point: :func:`powderline.topas.writer.write_topas_inp`.

Citation notation used throughout this subpackage: ``findings §X`` / ``plan §Y``
/ ``D-numbers`` cite the internal TOPAS design record (empirical findings and
decision log kept with the maintainers' development notes); they label *why* a
formula or behavior is the way it is and appear verbatim in some emitted INP
headers, so they are stable identifiers - do not renumber them.
"""

from __future__ import annotations

from .errors import TopasTranslationError
from .writer import render_topas, write_topas_inp

__all__ = ["write_topas_inp", "render_topas", "TopasTranslationError"]
