"""Tests for peak broadening model support (Sprint 2).

Verifies that:
1. Schema validation catches unsupported broadening models
2. Isotropic models work correctly
3. NotImplementedError raised for uniaxial/ellipsoidal/generalized models
"""

import pytest
from pydantic import ValidationError
from powderline.schema import RecipeModel, SizeBroadening, StrainBroadening


class TestSizeBroadeningModels:
    """Test size broadening model validation."""

    def test_isotropic_size_model_valid(self):
        """Verify isotropic size model validates successfully."""
        size = SizeBroadening(
            model="isotropic",
            isotropic_size=[100.0, True, None, None],
            LG_eta=[0.5, False, None, None]
        )
        assert size.model == "isotropic"
        # RefinementParameter is tuple type, so comparisons use tuples
        assert size.isotropic_size == (100.0, True, None, None)

    def test_uniaxial_size_model_raises_not_implemented(self):
        """Verify uniaxial size model raises NotImplementedError."""
        with pytest.raises(NotImplementedError, match="Uniaxial size broadening"):
            SizeBroadening(
                model="uniaxial",
                uniaxial_equatorial=[100.0, True, None, None],
                uniaxial_axial=[150.0, True, None, None],
                hkl_direction=[0, 0, 1]
            )

    def test_ellipsoidal_size_model_raises_not_implemented(self):
        """Verify ellipsoidal size model raises NotImplementedError."""
        with pytest.raises(NotImplementedError, match="Ellipsoidal size broadening"):
            SizeBroadening(
                model="ellipsoidal",
                S11=[1.0, True, None, None],
                S22=[1.0, True, None, None],
                S33=[1.0, True, None, None]
            )

    def test_invalid_size_model_rejected(self):
        """Verify invalid model names are rejected by Literal type."""
        with pytest.raises(ValidationError, match="Input should be 'isotropic', 'uniaxial' or 'ellipsoidal'"):
            SizeBroadening(model="invalid_model")

    def test_default_size_model_is_isotropic(self):
        """Verify default model is isotropic."""
        size = SizeBroadening()
        assert size.model == "isotropic"


class TestStrainBroadeningModels:
    """Test strain broadening model validation."""

    def test_isotropic_strain_model_valid(self):
        """Verify isotropic strain model validates successfully."""
        strain = StrainBroadening(
            model="isotropic",
            isotropic_strain=[0.001, True, None, None],
            LG_eta=[0.8, False, None, None]
        )
        assert strain.model == "isotropic"
        # RefinementParameter is tuple type, so comparisons use tuples
        assert strain.isotropic_strain == (0.001, True, None, None)

    def test_uniaxial_strain_model_raises_not_implemented(self):
        """Verify uniaxial strain model raises NotImplementedError."""
        with pytest.raises(NotImplementedError, match="Uniaxial strain broadening"):
            StrainBroadening(
                model="uniaxial",
                uniaxial_equatorial=[0.001, True, None, None],
                uniaxial_axial=[0.002, True, None, None],
                hkl_direction=[1, 1, 0]
            )

    def test_generalized_strain_model_raises_not_implemented(self):
        """Verify generalized (Stephens) strain model raises NotImplementedError."""
        with pytest.raises(NotImplementedError, match="Generalized.*Stephens.*strain broadening"):
            StrainBroadening(
                model="generalized",
                stephens_parameters={"S400": [0.001, True, None, None]}
            )

    def test_invalid_strain_model_rejected(self):
        """Verify invalid model names are rejected by Literal type."""
        with pytest.raises(ValidationError, match="Input should be 'isotropic', 'uniaxial' or 'generalized'"):
            StrainBroadening(model="ellipsoidal")  # Not valid for strain

    def test_default_strain_model_is_isotropic(self):
        """Verify default model is isotropic."""
        strain = StrainBroadening()
        assert strain.model == "isotropic"


class TestBroadeningInRecipe:
    """Test broadening models in full recipe validation."""

    def test_recipe_with_isotropic_broadening_valid(self):
        """Verify recipe with isotropic broadening validates."""
        recipe = {
            "schema_name": "GSASII_Rietveld",
            "schema_version": "0.26.0",
            "payload": {
                "xrd_data": {"tth": [10], "Itth": [100], "Itth_weights": [1]},
                "instrument": {
                    "description": "test",
                    "initialization": [{"Type": ["PXC", "PXC", False]}, {}]
                },
                "phases": {
                    "TestPhase": {
                        "structure": {
                            "phase_name": "TestPhase",
                            "space_group": "Pm-3m",
                            "unit_cell": {
                                "a": 4.1569, "b": 4.1569, "c": 4.1569,
                                "alpha": 90.0, "beta": 90.0, "gamma": 90.0
                            },
                            "atoms": {}
                        },
                        "parameterization": {
                            "peak_broadening": {
                                "size_broadening": {
                                    "model": "isotropic",
                                    "isotropic_size": [100.0, True, None, None],
                                    "LG_eta": [0.5, False, None, None]
                                },
                                "strain_broadening": {
                                    "model": "isotropic",
                                    "isotropic_strain": [0.001, True, None, None],
                                    "LG_eta": [0.8, False, None, None]
                                }
                            }
                        }
                    }
                },
                "refinement_controls": {
                    "refinement_cycles": 1
                }
            }
        }

        model = RecipeModel.model_validate(recipe)
        size = model.payload.phases["TestPhase"].parameterization.peak_broadening.size_broadening
        strain = model.payload.phases["TestPhase"].parameterization.peak_broadening.strain_broadening

        assert size.model == "isotropic"
        assert strain.model == "isotropic"

    def test_recipe_with_uniaxial_size_raises_error(self):
        """Verify recipe with uniaxial size model raises NotImplementedError."""
        recipe = {
            "schema_name": "GSASII_Rietveld",
            "schema_version": "0.26.0",
            "payload": {
                "xrd_data": {"tth": [10], "Itth": [100], "Itth_weights": [1]},
                "instrument": {
                    "description": "test",
                    "initialization": [{"Type": ["PXC", "PXC", False]}, {}]
                },
                "phases": {
                    "TestPhase": {
                        "structure": {
                            "phase_name": "TestPhase",
                            "space_group": "Pm-3m",
                            "unit_cell": {
                                "a": 4.1569, "b": 4.1569, "c": 4.1569,
                                "alpha": 90.0, "beta": 90.0, "gamma": 90.0
                            },
                            "atoms": {}
                        },
                        "parameterization": {
                            "peak_broadening": {
                                "size_broadening": {
                                    "model": "uniaxial",
                                    "uniaxial_equatorial": [100.0, True, None, None],
                                    "uniaxial_axial": [150.0, True, None, None],
                                    "hkl_direction": [0, 0, 1]
                                }
                            }
                        }
                    }
                },
                "refinement_controls": {
                    "refinement_cycles": 1
                }
            }
        }

        with pytest.raises(NotImplementedError, match="Uniaxial size broadening"):
            RecipeModel.model_validate(recipe)

    def test_recipe_with_generalized_strain_raises_error(self):
        """Verify recipe with generalized strain model raises NotImplementedError."""
        recipe = {
            "schema_name": "GSASII_Rietveld",
            "schema_version": "0.26.0",
            "payload": {
                "xrd_data": {"tth": [10], "Itth": [100], "Itth_weights": [1]},
                "instrument": {
                    "description": "test",
                    "initialization": [{"Type": ["PXC", "PXC", False]}, {}]
                },
                "phases": {
                    "TestPhase": {
                        "structure": {
                            "phase_name": "TestPhase",
                            "space_group": "Pm-3m",
                            "unit_cell": {
                                "a": 4.1569, "b": 4.1569, "c": 4.1569,
                                "alpha": 90.0, "beta": 90.0, "gamma": 90.0
                            },
                            "atoms": {}
                        },
                        "parameterization": {
                            "peak_broadening": {
                                "strain_broadening": {
                                    "model": "generalized",
                                    "stephens_parameters": {"S400": [0.001, True, None, None]}
                                }
                            }
                        }
                    }
                },
                "refinement_controls": {
                    "refinement_cycles": 1
                }
            }
        }

        with pytest.raises(NotImplementedError, match="Generalized.*Stephens"):
            RecipeModel.model_validate(recipe)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
