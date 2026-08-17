"""Shared exception type for the TOPAS translation path.

Kept in its own module so both ``symmetry`` and ``writer`` can raise it without
a circular import. Re-exported from ``powderline.topas`` for convenience.
"""

from __future__ import annotations


class TopasTranslationError(ValueError):
    """A recipe cannot be faithfully translated to a TOPAS v7 INP.

    Raised only for genuinely untranslatable input: an unsupported
    ``schema_name``, an unrecognised or unresolved space-group symbol, a
    rhombohedral ``:R`` setting, or SPF ``use_instrument_profile=true``.
    Symmetry-breaking refine flags (refined ``Uaniso``, symmetry-fixed or
    symmetry-coupled coordinates, symmetry-fixed angles) instead **warn + emit**
    -- the engine is permissive and does not arbitrate recipe correctness
    (design-options ``D3-update``).
    """
