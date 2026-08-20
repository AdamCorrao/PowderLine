# LaB6 easydiffraction example

Derived from the stock `example_LaB6` with modifications to match easydiffraction's supported feature set:

- Background peaks removed (`single_peaks` removed from `background`)
- Broadening parameter Z fixed (`refine_flag` set to `False`)
- Lattice parameter `a` freed for refinement (`refine_flag` set to `True`)

The easydiffraction engine does not support background peak refinement or Z broadening refinement, so this example demonstrates a compatible workflow.

## Running

```python
import json
from powderline.engine import run

recipe = json.load(open("examples/example_LaB6_easydiff/input.json"))
output_dir = "output"
result = run(recipe, output_dir, engine="easydiffraction")
```

Expected refinement quality: Rwp ~19.8% (higher than GSAS-II due to missing SH/L axial-divergence asymmetry model and background peaks), lattice parameter `a` refined to ~4.1575 Å (NIST SRM 660 reference: 4.15682 Å).
