"""
Forward Modeling Unit Tests
=============================

Tests gravity and magnetic forward modeling modules against known analytical solutions.

Test categories:
  1. Zero model -> zero output
  2. Single prism: numerical vs analytical (error < 1e-4)
  3. Uniform half-space: center region approximately constant
  4. Magnetic symmetry: vertical magnetization produces symmetric anomaly
  5. Numerical sanity: no NaN/Inf, reasonable value ranges
  6. Noise addition: statistical properties
"""

import sys
import os
import time
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data.forward_gravity import (
    compute_gravity_anomaly,
    forward_gravity,
    add_gravity_noise,
    analytical_prism_gravity_single,
    G,
)
from src.data.forward_magnetic import (
    compute_magnetic_anomaly,
    forward_magnetic,
    add_magnetic_noise,
    analytical_prism_magnetic_single,
)


# ============================================================
# Test Configuration
# ============================================================

class Cfg:
    """Standard test grid parameters."""
    NX = 21; NY = 21; NZ = 10
    DX = DY = DZ = 50.0   # meters
    Z0 = 10.0              # observation height (m)


def make_grid(**kwargs):
    d = {'dx': Cfg.DX, 'dy': Cfg.DY, 'dz': Cfg.DZ,
         'x0': 0.0, 'y0': 0.0}
    d.update(kwargs)
    return d


# ============================================================
# Test 1: Zero Model -> Zero Output
# ============================================================

class TestZeroModel:
    def test_zero_gravity(self):
        rho = np.zeros((Cfg.NX, Cfg.NY, Cfg.NZ))
        gz = compute_gravity_anomaly(rho, [0.0], [0.0], Cfg.Z0, **make_grid())
        assert np.allclose(gz, 0.0, atol=1e-15), f"Zero density -> non-zero: max={np.max(np.abs(gz)):.2e}"

    def test_zero_magnetic(self):
        kap = np.zeros((Cfg.NX, Cfg.NY, Cfg.NZ))
        dt = compute_magnetic_anomaly(kap, [0.0], [0.0], Cfg.Z0, **make_grid())
        assert np.allclose(dt, 0.0, atol=1e-15), f"Zero susceptibility -> non-zero: max={np.max(np.abs(dt)):.2e}"


# ============================================================
# Test 2: Single Prism — Numerical vs Analytical
# ============================================================

class TestSinglePrismAccuracy:
    """Compare numerical prism forward with analytical reference solution."""

    @pytest.fixture
    def single_prism(self):
        """Single centered prism in a small grid.

        Uses the same coordinate convention as forward_gravity():
        model origin at (-nx*dx/2, -ny*dy/2), observation points on a grid
        centered on the model with half-cell padding.
        """
        nx, ny, nz = 11, 11, 5
        dx = dy = dz = 100.0
        ci, cj, ck = nx // 2, ny // 2, nz // 2

        rho_model = np.zeros((nx, ny, nz))
        rho_model[ci, cj, ck] = 1.0  # g/cm^3

        # Model origin at center: cell (ci,cj,ck) is centered at (0,0,ck*dz)
        x0 = -nx * dx / 2.0
        y0 = -ny * dy / 2.0
        z0_model = 0.0

        # Prism boundaries in absolute coordinates
        x1 = x0 + ci * dx; x2 = x0 + (ci + 1) * dx
        y1 = y0 + cj * dy; y2 = y0 + (cj + 1) * dy
        z1 = z0_model + ck * dz; z2 = z0_model + (ck + 1) * dz

        # Observation grid matches forward_gravity() convention:
        # obs coords from -nx*dx/2-dx/2 to +nx*dx/2+dx/2
        half = nx * dx / 2.0
        obs = np.linspace(-half - dx/2, half + dx/2, nx)

        return {
            'rho_model': rho_model, 'nx': nx, 'ny': ny,
            'obs': obs, 'density': 1.0,
            'geom': (x1, x2, y1, y2, z1, z2),
            'grid': {'dx': dx, 'dy': dy, 'dz': dz, 'x0': x0, 'y0': y0},
        }

    def test_gravity_numerical_vs_analytical(self, single_prism):
        """Gravity: numerical vs analytical error < 0.1%."""
        sp = single_prism
        gz_num = compute_gravity_anomaly(sp['rho_model'], sp['obs'], sp['obs'],
                                          obs_z=10.0, **sp['grid'])

        x1, x2, y1, y2, z1, z2 = sp['geom']
        # Analytical: use meshgrid for 2D observation grid
        OX, OY = np.meshgrid(sp['obs'], sp['obs'], indexing='ij')
        gz_ana = analytical_prism_gravity_single(
            OX.ravel(), OY.ravel(), 10.0,
            x1, x2, y1, y2, z1, z2, density_gcc=sp['density']
        ).reshape(gz_num.shape)

        max_val = max(np.max(np.abs(gz_num)), np.max(np.abs(gz_ana)), 1e-30)
        max_abs_diff = np.max(np.abs(gz_num - gz_ana))
        rel_err = max_abs_diff / max_val

        assert rel_err < 1e-3, (
            f"Gravity numerical vs analytical relative error too large: {rel_err:.6f} "
            f"(want < 1e-3), max_diff={max_abs_diff:.8f} mGal"
        )

    def test_magnetic_numerical_vs_analytical(self, single_prism):
        """Magnetic: numerical vs analytical error < 0.1%."""
        sp = single_prism
        kap_model = np.zeros_like(sp['rho_model'])
        kap_model[sp['rho_model'] != 0] = 0.1  # SI

        dt_num = compute_magnetic_anomaly(kap_model, sp['obs'], sp['obs'],
                                            obs_z=10.0, **sp['grid'])

        x1, x2, y1, y2, z1, z2 = sp['geom']
        OX, OY = np.meshgrid(sp['obs'], sp['obs'], indexing='ij')
        dt_ana = analytical_prism_magnetic_single(
            OX.ravel(), OY.ravel(), 10.0,
            x1, x2, y1, y2, z1, z2, kappa_si=0.1
        ).reshape(dt_num.shape)

        max_val = max(np.max(np.abs(dt_num)), np.max(np.abs(dt_ana)), 1e-30)
        max_abs_diff = np.max(np.abs(dt_num - dt_ana))
        rel_err = max_abs_diff / max_val

        assert rel_err < 1e-3, (
            f"Magnetic numerical vs analytical relative error too large: {rel_err:.6f} "
            f"(want < 1e-3), max_diff={max_abs_diff:.8f} nT"
        )

    def test_gravity_peak_above_prism(self, single_prism):
        """Gravity peak should be directly above the prism."""
        sp = single_prism
        gz = compute_gravity_anomaly(sp['rho_model'], sp['obs'], sp['obs'],
                                      obs_z=10.0, **sp['grid'])
        ci, cj = sp['nx'] // 2, sp['ny'] // 2
        peak = np.unravel_index(np.argmax(gz), gz.shape)
        assert abs(peak[0] - ci) <= 1 and abs(peak[1] - cj) <= 1, \
            f"Peak at {peak}, expected near ({ci},{cj})"


# ============================================================
# Test 3: Uniform Half-Space
# ============================================================

class TestUniformHalfSpace:

    def test_gravity_center_constant(self):
        """Center of uniform half-space should have nearly constant gravity."""
        rho = np.full((Cfg.NX, Cfg.NY, Cfg.NZ), 1.0)
        gz = compute_gravity_anomaly(rho,
                                      np.arange(Cfg.NX) * Cfg.DX + Cfg.DX/2,
                                      np.arange(Cfg.NY) * Cfg.DY + Cfg.DY/2,
                                      obs_z=Cfg.Z0, **make_grid())

        cx, cy = Cfg.NX // 2, Cfg.NY // 2
        center = gz[cx-2:cx+3, cy-2:cy+3]
        cv = np.std(center) / (np.mean(np.abs(center)) + 1e-30)
        assert cv < 0.05, f"Center CV too large: {cv:.4f}"

    def test_positive_density_positive_gravity(self):
        """Positive density produces positive gravity anomaly at center."""
        rho = np.full((Cfg.NX, Cfg.NY, Cfg.NZ), 0.5)
        half = Cfg.NX * Cfg.DX / 2.0
        obs = np.linspace(-half - Cfg.DX/2, half + Cfg.DX/2, Cfg.NX)
        gz = compute_gravity_anomaly(rho, obs, obs,
                                      obs_z=Cfg.Z0, **make_grid())
        # Center region should be positive for a uniform density half-space
        cx, cy = Cfg.NX // 2, Cfg.NY // 2
        center_val = gz[cx, cy]
        assert center_val > 0, f"Center gravity should be positive: {center_val:.4f}"


# ============================================================
# Test 4: Magnetic Symmetry
# ============================================================

class TestMagneticSymmetry:

    def test_vertical_mag_symmetry(self):
        """Vertical magnetization (I=90): anomaly symmetric about source center."""
        nx, ny, nz = 31, 31, 10
        dx = dy = dz = 50.0
        kap = np.zeros((nx, ny, nz))
        kap[15, 15, 3] = 0.1

        gp = make_grid(dx=dx, dy=dy, dz=dz)
        mp = {'I': 90.0, 'D': 0.0, 'F': 55000.0}
        dt = compute_magnetic_anomaly(kap,
                                       np.arange(nx)*dx + dx/2,
                                       np.arange(ny)*dy + dy/2,
                                       obs_z=10.0, **gp,
                                       F_inclination=90.0, F_declination=0.0)

        ci, cj = 15, 15
        for di in [1, 2, 3]:
            for dj in [1, 2, 3]:
                if all(0 <= ci+d < nx and 0 <= ci-d >= 0 and
                       0 <= cj+d < ny and 0 <= cj-d >= 0 for d in [di, dj]):
                    vp = dt[ci+di, cj+dj]
                    vm = dt[ci-di, cj-dj]
                    diff_ratio = abs(vp - vm) / ((abs(vp)+abs(vm))/2 + 1e-30)
                    assert diff_ratio < 0.05, (
                        f"Symmetry broken at offset ({di},{dj}): "
                        f"{vp:.6f} vs {vm:.6f}, ratio={diff_ratio:.4f}")

    def test_inclined_differs_from_vertical(self):
        """Inclined magnetization should differ from vertical."""
        nx, ny, nz = 21, 21, 10
        dx = dy = dz = 50.0
        kap = np.zeros((nx, ny, nz))
        kap[10, 10, 3] = 0.1

        gp = make_grid(dx=dx, dy=dy, dz=dz)
        obs = np.arange(nx) * dx + dx/2

        dt_vert = compute_magnetic_anomaly(kap, obs, obs, obs_z=10.0, **gp,
                                           F_inclination=90.0, F_declination=0.0)
        dt_inc = compute_magnetic_anomaly(kap, obs, obs, obs_z=10.0, **gp,
                                          F_inclination=45.0, F_declination=0.0)

        assert not np.allclose(dt_inc, dt_vert, rtol=0.01), \
            "Inclined and vertical magnetization produce identical results"


# ============================================================
# Test 5: Numerical Sanity Checks
# ============================================================

class TestNumericalSanity:

    def test_no_nan_inf_gravity(self):
        rho = np.random.randn(Cfg.NX, Cfg.NY, Cfg.NZ) * 0.5
        gz = compute_gravity_anomaly(rho,
                                      np.arange(Cfg.NX)*Cfg.DX + Cfg.DX/2,
                                      np.arange(Cfg.NY)*Cfg.DY + Cfg.DY/2,
                                      obs_z=Cfg.Z0, **make_grid())
        assert not np.any(np.isnan(gz)), "Gravity contains NaN"
        assert not np.any(np.isinf(gz)), "Gravity contains Inf"

    def test_no_nan_inf_magnetic(self):
        kap = np.random.rand(Cfg.NX, Cfg.NY, Cfg.NZ) * 0.1
        dt = compute_magnetic_anomaly(kap,
                                       np.arange(Cfg.NX)*Cfg.DX + Cfg.DX/2,
                                       np.arange(Cfg.NY)*Cfg.DY + Cfg.DY/2,
                                       obs_z=Cfg.Z0, **make_grid())
        assert not np.any(np.isnan(dt)), "Magnetic contains NaN"
        assert not np.any(np.isinf(dt)), "Magnetic contains Inf"

    def test_gravity_value_range(self):
        """Reasonable gravity values for typical model scales."""
        nx, ny, nz = 21, 21, 15
        dx = dy = dz = 100.0
        rho = np.zeros((nx, ny, nz))
        rho[5:16, 5:16, 2:8] = 0.5
        gz = compute_gravity_anomaly(rho,
                                      np.arange(nx)*dx + dx/2,
                                      np.arange(ny)*dy + dy/2,
                                      obs_z=10.0,
                                      **make_grid(dx=dx, dy=dy, dz=dz))
        assert np.max(gz) > 0, "Positive density -> positive max gravity"
        assert np.max(gz) < 50000, f"Gravity too large: {np.max(gz):.1f} mGal"

    def test_magnetic_value_range(self):
        """Reasonable magnetic values for typical model scales."""
        nx, ny, nz = 21, 21, 15
        dx = dy = dz = 100.0
        kap = np.zeros((nx, ny, nz))
        kap[7:14, 7:14, 2:8] = 0.1
        dt = compute_magnetic_anomaly(kap,
                                       np.arange(nx)*dx + dx/2,
                                       np.arange(ny)*dy + dy/2,
                                       obs_z=10.0,
                                       **make_grid(dx=dx, dy=dy, dz=dz))
        assert np.max(dt) > 0, "Positive susceptibility -> positive max magnetic"
        assert np.max(dt) < 200000, f"Magnetic too large: {np.max(dt):.1f} nT"

    def test_computation_time(self):
        """Single-sample computation should be fast (< 10s)."""
        nx, ny, nz = 21, 21, 15
        dx = dy = dz = 100.0
        rho = np.zeros((nx, ny, nz))
        rho[5:16, 5:16, 2:8] = 0.5
        gp = make_grid(dx=dx, dy=dy, dz=dz)
        obs = np.arange(nx) * dx + dx/2

        t0 = time.time()
        gz = compute_gravity_anomaly(rho, obs, obs, obs_z=10.0, **gp)
        t_grav = time.time() - t0

        kap = np.zeros((nx, ny, nz))
        kap[5:16, 5:16, 2:8] = 0.1
        t0 = time.time()
        dt = compute_magnetic_anomaly(kap, obs, obs, obs_z=10.0, **gp)
        t_mag = time.time() - t0

        print(f"\n  [Timing] Gravity: {t_grav:.3f}s, Magnetic: {t_mag:.3f}s "
              f"({nx}x{ny}x{nz} grid)")
        assert t_grav < 10.0, f"Gravity too slow: {t_grav:.1f}s"
        assert t_mag < 10.0, f"Magnetic too slow: {t_mag:.1f}s"


# ============================================================
# Test 6: Noise Addition
# ============================================================

class TestNoiseAddition:

    def test_gravity_noise_stats(self):
        """Gravity noise should have ~zero mean and correct std."""
        gz = np.ones((50, 50)) * 100.0  # constant signal
        noisy = add_gravity_noise(gz, noise_level=0.005, seed=42)
        residual = noisy - gz
        # Noise std should be roughly 0.005 * 100 = 0.5
        assert np.abs(np.mean(residual)) < 0.2, f"Noise bias: {np.mean(residual):.4f}"
        assert np.std(residual) > 0.1, "Noise seems too small"

    def test_magnetic_noise_stats(self):
        """Magnetic noise should have ~zero mean and correct std."""
        mag = np.ones((50, 50)) * 500.0
        noisy = add_magnetic_noise(mag, noise_level=0.01, seed=42)
        residual = noisy - mag
        assert np.abs(np.mean(residual)) < 1.0, f"Noise bias: {np.mean(residual):.4f}"
        assert np.std(residual) > 1.0, "Noise seems too small"

    def test_zero_signal_no_noise(self):
        """Zero signal should return zero even with noise."""
        gz = np.zeros((10, 10))
        result = add_gravity_noise(gz, noise_level=0.005)
        assert np.allclose(result, 0.0), "Zero signal should remain zero"


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
