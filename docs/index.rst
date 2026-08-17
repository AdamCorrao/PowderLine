.. PowderLine documentation master file

PowderLine Documentation
========================

PowderLine automates crystallographic refinement using GSAS-II software.
It takes a JSON "recipe" describing refinement parameters and produces
standardized output reports.

.. toctree::
   :maxdepth: 2
   :caption: User Guide:

   quickstart
   getting-started
   integration-guide
   mp-simulation
   TROUBLESHOOTING
   SCHEMA_HISTORY
   api

.. toctree::
   :maxdepth: 1
   :caption: Developer:

   DEVELOPMENT
   known_issues
   regression-tolerance
   cross-platform-guide

Quickstart
----------

New to PowderLine? See :doc:`quickstart` for the shortest path from a fresh clone
to a finished refinement, on Linux and Windows.

Getting Started
---------------

See :doc:`getting-started` for the full recipe structure, output files, and the
Python API.

Using PowderLine in Other Projects
-----------------------------------

See :doc:`integration-guide` for instructions on using PowderLine as a library
in other local projects, including pixi environment templates and example scripts.

Materials Project Simulation
----------------------------

See :doc:`mp-simulation` for generating simulated diffraction patterns from
Materials Project structures with ``pixi run mp-simulate``.

Developer Guide
---------------

See :doc:`DEVELOPMENT` for comprehensive developer documentation including
architecture, domain concepts, and contribution guidelines.

API Reference
-------------

See :doc:`api` for detailed function and class documentation.

Troubleshooting
---------------

See :doc:`TROUBLESHOOTING` for common errors and solutions.

Schema History
--------------

See :doc:`SCHEMA_HISTORY` for the recipe-schema changelog and migration guidance.

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
