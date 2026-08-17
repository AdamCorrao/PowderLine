"""Build PowderLine simulation recipes from crystal structures.

This module converts crystal structure data (from Materials Project or other sources)
into PowderLine JSON recipes configured for simulation mode (refinement_cycles=1).

Simulations run through the GSAS-II engine, so all defaults follow GSAS-II
conventions and units (NOT TOPAS, which differs for polarization, axial
divergence, and the instrumental broadening parameters):

- wavelength: Angstroms
- isotropic_size: microns
- isotropic_strain: delta-d/d x 10^-6
- LG_eta: Lorentzian/Gaussian mixing (1 = Lorentzian, 0 = Gaussian)
- Uiso: Angstrom^2
"""

from typing import Dict, Any, Optional, List
import numpy as np

from ._status import WARN


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge ``override`` into ``base`` (returns a new dict)."""
    merged = dict(base)
    for key, value in override.items():
        if (key in merged and isinstance(merged[key], dict)
                and isinstance(value, dict)):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


# Markers for leaf types in _DEFAULTS_SCHEMA
_NUM = 'number'
_STR = 'string'
_NUM_LIST = 'number list'

# The complete allowed shape of a user-supplied simulation_defaults dict.
# Must mirror _get_hardcoded_defaults and .powderline_config.yaml.example.
_DEFAULTS_SCHEMA: Dict[str, Any] = {
    'instrument_defaults': {
        'wavelength': _NUM,
        'polarization': _NUM,
        'zero_shift': _NUM,
        'axial_divergence': _NUM,
        'broadening': {'U': _NUM, 'V': _NUM, 'W': _NUM,
                       'X': _NUM, 'Y': _NUM, 'Z': _NUM},
    },
    'phase_defaults': {
        'scale': _NUM,
        'Uiso': _NUM,
        'occupancy': _NUM,
        'size_broadening': {'model': _STR, 'isotropic_size': _NUM,
                            'LG_eta': _NUM},
        'strain_broadening': {'model': _STR, 'isotropic_strain': _NUM,
                              'LG_eta': _NUM},
    },
    'data_range': {'min_2theta': _NUM, 'max_2theta': _NUM, 'step': _NUM},
    'background': {'chebyshev_coefficients': _NUM_LIST},
}

_SEE_EXAMPLE = "See .powderline_config.yaml.example for the supported layout."


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _validate_user_defaults(user: Any,
                            schema: Dict[str, Any] = _DEFAULTS_SCHEMA,
                            path: str = 'simulation_defaults') -> None:
    """Strictly validate a user-supplied simulation_defaults dict.

    Unknown keys at any level and invalid values are rejected with a
    ValueError naming the offending key path, so config mistakes (typos,
    the old flat layout, empty YAML keys that parse as null) fail here with
    a clear message instead of surfacing deep inside GSAS-II.
    """
    if not isinstance(user, dict):
        raise ValueError(
            f"Invalid config: '{path}' must be a mapping of settings, "
            f"got {user!r}. {_SEE_EXAMPLE}"
        )
    for key, value in user.items():
        key_path = f"{path}.{key}"
        if key not in schema:
            raise ValueError(
                f"Invalid config: unknown key '{key_path}'. Allowed keys "
                f"here: {', '.join(schema)}. {_SEE_EXAMPLE}"
            )
        expected = schema[key]
        if isinstance(expected, dict):
            _validate_user_defaults(value, expected, key_path)
        elif expected == _NUM:
            if not _is_number(value):
                hint = (" (an empty YAML key parses as null — set a value "
                        "or remove the key)") if value is None else ""
                raise ValueError(
                    f"Invalid config: '{key_path}' must be a number, "
                    f"got {value!r}{hint}. {_SEE_EXAMPLE}"
                )
        elif expected == _STR:
            if not isinstance(value, str):
                raise ValueError(
                    f"Invalid config: '{key_path}' must be a string, "
                    f"got {value!r}. {_SEE_EXAMPLE}"
                )
        elif expected == _NUM_LIST:
            if (not isinstance(value, list) or not value
                    or not all(_is_number(v) for v in value)):
                raise ValueError(
                    f"Invalid config: '{key_path}' must be a non-empty "
                    f"list of numbers, got {value!r}. {_SEE_EXAMPLE}"
                )


class SimulationRecipeBuilder:
    """Builder for PowderLine simulation recipes (GSAS-II conventions)."""

    def __init__(self, simulation_defaults: Optional[Dict[str, Any]] = None):
        """Initialize recipe builder.

        Args:
            simulation_defaults: Dictionary with default simulation parameters
                               (instrument_defaults, phase_defaults, data_range,
                               background). Partial dicts are allowed — missing
                               keys fall back to the hardcoded defaults.
                               Validation is strict: unknown keys (including
                               the retired flat layout) and invalid values
                               raise ValueError naming the offending key.

        Raises:
            ValueError: If simulation_defaults contains unknown keys or
                invalid values.
        """
        user_defaults = simulation_defaults or {}
        _validate_user_defaults(user_defaults)
        self.defaults = _deep_merge(self._get_hardcoded_defaults(), user_defaults)

    def _get_hardcoded_defaults(self) -> Dict[str, Any]:
        """Fallback defaults if no config provided (GSAS-II units)."""
        return {
            'instrument_defaults': {
                'wavelength': 0.4133,      # Angstroms (30 keV)
                'polarization': 0.99,      # GSAS-II 'Polariz.'
                'zero_shift': 0.0,         # degrees 2theta ('Zero')
                'axial_divergence': 0.001, # GSAS-II 'SH/L'
                # TCH pseudo-Voigt instrumental broadening (GSAS-II U,V,W,X,Y,Z)
                'broadening': {'U': 1.0, 'V': -0.5, 'W': 0.1,
                               'X': 0.0, 'Y': 0.0, 'Z': 0.0},
            },
            'phase_defaults': {
                'scale': 1.0,
                # Structural fallbacks — applied ONLY when the structure source
                # does not provide a value (a warning is emitted per phase)
                'Uiso': 0.01,       # Angstrom^2
                'occupancy': 1.0,
                'size_broadening': {
                    'model': 'isotropic',
                    'isotropic_size': 10.0,  # microns (~no size broadening)
                    'LG_eta': 1.0,           # 1 = Lorentzian, 0 = Gaussian
                },
                'strain_broadening': {
                    'model': 'isotropic',
                    'isotropic_strain': 0.0,  # delta-d/d x 10^-6
                    'LG_eta': 1.0,
                },
            },
            'data_range': {'min_2theta': 1.0, 'max_2theta': 80.0, 'step': 0.05},
            'background': {'chebyshev_coefficients': [10.0, 0.0, 0.0]},
        }

    def _generate_dummy_xrd_data(self) -> Dict[str, Any]:
        """Generate dummy XRD data arrays for simulation.

        GSAS-II requires observed data arrays even in simulation mode.
        We generate a uniform grid with dummy intensities.

        Raises:
            ValueError: If data_range parameters are invalid
        """
        data_range = self.defaults['data_range']

        # Validate data range parameters
        min_2theta = data_range['min_2theta']
        max_2theta = data_range['max_2theta']
        step = data_range['step']

        if min_2theta >= max_2theta:
            raise ValueError(f"min_2theta ({min_2theta}) must be < max_2theta ({max_2theta})")

        if step <= 0:
            raise ValueError(f"step ({step}) must be > 0")

        # Check for excessive number of points (>50k could cause memory issues)
        num_points = int((max_2theta - min_2theta) / step)
        if num_points > 50000:
            raise ValueError(
                f"Data range would generate {num_points} points (>50k limit). "
                f"Increase step size (currently {step}°) or reduce 2θ range."
            )

        tth = np.arange(min_2theta, max_2theta, step).tolist()

        # Dummy intensities (will be ignored, y_calc is what matters)
        Itth = [1.0] * len(tth)

        # Uniform weights
        Itth_weights = [1.0] * len(tth)

        return {
            'tth': tth,
            'Itth': Itth,
            'Itth_weights': Itth_weights,
            'filename': 'simulation_dummy_data'
        }

    def _build_instrument_initialization(self) -> List[Dict[str, Any]]:
        """Build instrument initialization parameters (Iparm1, Iparm2)."""
        inst = self.defaults['instrument_defaults']
        wavelength = inst['wavelength']
        broadening = inst['broadening']

        Iparm1 = {
            'Type': ['PXC', 'PXC', False],
            'Lam': [wavelength, wavelength, False],
            'Polariz.': [inst['polarization'], inst['polarization'], False],
            'U': [broadening['U'], broadening['U'], False],
            'V': [broadening['V'], broadening['V'], False],
            'W': [broadening['W'], broadening['W'], False],
            'X': [broadening['X'], broadening['X'], False],
            'Y': [broadening['Y'], broadening['Y'], False],
            'Z': [broadening['Z'], broadening['Z'], False],
            'Zero': [inst['zero_shift'], inst['zero_shift'], False],
            'SH/L': [inst['axial_divergence'], inst['axial_divergence'], False],
            'Azimuth': [0.0, 0.0, False],
            'Bank': [1, 1, False]
        }

        Iparm2 = {}

        return [Iparm1, Iparm2]

    def _build_structure_from_mp_data(self, mp_structure_data: Dict[str, Any], phase_name: str) -> Dict[str, Any]:
        """Build PowderLine phase structure from Materials Project data.

        Uses the structure data provided by MP API directly, avoiding CIF parsing.
        Structural fallbacks (Uiso, occupancy) are applied ONLY to sites that do
        not provide a value, with a warning naming the affected atoms.
        """
        # Use MP-provided data (unit_cell, space_group, sites already extracted)
        if not mp_structure_data.get('unit_cell') or not mp_structure_data.get('space_group') or not mp_structure_data.get('sites'):
            raise ValueError(
                "MP structure data missing required fields. "
                "Ensure unit_cell, space_group, and sites are present."
            )

        phase_defaults = self.defaults['phase_defaults']
        default_uiso = phase_defaults['Uiso']
        default_occupancy = phase_defaults['occupancy']

        # Convert MP sites to PowderLine atom format
        atoms = {}
        missing_uiso = []
        missing_occupancy = []
        for site in mp_structure_data['sites']:
            label = site['label']
            # A key that is absent OR explicitly None both count as missing
            uiso = site.get('Uiso')
            occupancy = site.get('occupancy')
            if uiso is None:
                missing_uiso.append(label)
                uiso = default_uiso
            if occupancy is None:
                missing_occupancy.append(label)
                occupancy = default_occupancy
            atoms[label] = {
                'element': site['element'],
                'x': site['x'],
                'y': site['y'],
                'z': site['z'],
                'occupancy': occupancy,
                'Uiso': uiso,
                'ADP': 'Uiso',
                'Multiplicity': None
            }

        if missing_uiso:
            print(f"{WARN}  Phase '{phase_name}': no Uiso in structure source for "
                  f"{len(missing_uiso)} atom(s) ({', '.join(missing_uiso)}); "
                  f"default {default_uiso} Å² applied.")
        if missing_occupancy:
            print(f"{WARN}  Phase '{phase_name}': no occupancy in structure source "
                  f"for {len(missing_occupancy)} atom(s) "
                  f"({', '.join(missing_occupancy)}); "
                  f"default {default_occupancy} applied.")

        return {
            'phase_name': phase_name,
            'space_group': mp_structure_data['space_group'],
            'unit_cell': mp_structure_data['unit_cell'],
            'atoms': atoms
        }

    def build_recipe_from_mp_structure(
        self,
        mp_structure_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Build complete PowderLine simulation recipe from MP structure data.

        Args:
            mp_structure_data: Structure data from MPInterface.get_structure()

        Returns:
            Complete PowderLine recipe dictionary (schema 0.26.0)

        Note:
            Sample identity (name, description) is managed externally by the
            caller (e.g. via directory names or a database). Schema 0.25 removed
            the ``sample_name`` field from the recipe format.
        """
        formula = mp_structure_data['formula']
        material_id = mp_structure_data['material_id']

        # Build structure from MP data (no CIF parsing needed)
        phase_name = formula
        structure = self._build_structure_from_mp_data(mp_structure_data, phase_name)

        wavelength = self.defaults['instrument_defaults']['wavelength']
        energy_kev = 12.398 / wavelength
        phase_defaults = self.defaults['phase_defaults']
        size_defaults = phase_defaults['size_broadening']
        strain_defaults = phase_defaults['strain_broadening']

        # Build recipe
        recipe = {
            'schema_name': 'GSASII_Rietveld',
            'schema_version': '0.26.0',
            'payload': {
                # Dummy XRD data (required for GSAS-II, but not used in simulation)
                'xrd_data': self._generate_dummy_xrd_data(),

                # Instrument configuration
                'instrument': {
                    'description': (f'Simulated synchrotron '
                                    f'(λ={wavelength:.4f} Å, ~{energy_kev:.0f} keV)'),
                    'initialization': self._build_instrument_initialization(),
                    'parameterization': None  # No instrument refinement in simulation
                },

                # Background (flat, locked for simulation)
                'background': {
                    'chebyshev': {
                        'num_coefficients': len(self.defaults['background']['chebyshev_coefficients']),
                        'coefficients': self.defaults['background']['chebyshev_coefficients'],
                        'refine_flag': False  # CRITICAL: locked for simulation
                    },
                    'single_peaks': None
                },

                # Phase definition
                'phases': {
                    phase_name: {
                        'structure': structure,
                        'parameterization': {
                            'scale': [phase_defaults['scale'], False, None, None],  # Locked
                            'unit_cell': {
                                'a': [None, False, None, None],
                                'b': [None, False, None, None],
                                'c': [None, False, None, None],
                                'alpha': [None, False, None, None],
                                'beta': [None, False, None, None],
                                'gamma': [None, False, None, None]
                            },
                            'atoms': {},  # Empty dict — no atom refinement in simulation
                            'peak_broadening': {
                                'size_broadening': {
                                    'model': size_defaults['model'],
                                    'isotropic_size': [size_defaults['isotropic_size'], False, None, None],
                                    'LG_eta': [size_defaults['LG_eta'], False, None, None]
                                },
                                'strain_broadening': {
                                    'model': strain_defaults['model'],
                                    'isotropic_strain': [strain_defaults['isotropic_strain'], False, None, None],
                                    'LG_eta': [strain_defaults['LG_eta'], False, None, None]
                                }
                            }
                        }
                    }
                },

                # Simulation mode: 1 cycle, no refinement
                'refinement_controls': {
                    'refinement_cycles': 1
                },

                # Fit range covers full generated data range
                'fit_range': [
                    self.defaults['data_range']['min_2theta'],
                    self.defaults['data_range']['max_2theta']
                ]
            }
        }

        return recipe
