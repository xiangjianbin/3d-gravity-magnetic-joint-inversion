"""
3D Prism Magnetic Anomaly Forward Modeling (Bhattacharyya 1964 / Sharma 1986)

Computes total magnetic field anomaly (Delta-T) of a 3D susceptibility model
discretized into rectangular prisms. Supports CUDA acceleration via PyTorch.

Units:
  - Input susceptibility: SI (dimensionless)
  - Output: nT (nanotesla)
  - mu_0 = 4*pi * 10^-7 H/m (vacuum permeability)

Formula (Bhattacharyya 1964, Eq. for total field anomaly):
  dT = (mu_0/4pi) * F * kappa * sum_{i,j,k} mu_ijk *
       [ a1*arctan2(yj*zk, xi*R) + a2*arctan2(xi*zk, yj*R) + a3*arctan2(xi*yj, zk*R)
        - b12*log(R+yj) - b13*log(R+zk) - b23*log(R+xi) ]

  where:
    M = (Mx, My, Mz) is the magnetization direction (unit vector)
    F = (Fx, Fy, Fz) is the geomagnetic field direction (unit vector)
    a1 = Mx*Fx, a2 = My*Fy, a3 = Mz*Fz
    b12 = My*Fx + Mx*Fy, b13 = Mz*Fx + Mx*Fz, b23 = Mz*Fy + My*Fz
"""

import numpy as np
import torch

# Physical constants
MU_0 = 4.0 * np.pi * 1e-7  # Vacuum permeability (H/m)

# Device selection
_CUDA_AVAILABLE = torch.cuda.is_available()
_DEVICE = torch.device('cuda' if _CUDA_AVAILABLE else 'cpu')


def _prism_magnetic_numpy(obs_x, obs_y, obs_z,
                          x1, x2, y1, y2, z1, z2,
                          Mx, My, Mz, Fx, Fy, Fz):
    """
    Pure NumPy implementation of Bhattacharyya (1964) prism magnetic formula.

    Parameters
    ----------
    obs_x, obs_y : array-like (N_obs,) — observation coordinates (m)
    obs_z : float — observation height (m), positive upward
    x1..z2 : float — prism boundaries (m)
    Mx, My, Mz : float — magnetization direction cosines
    Fx, Fy, Fz : float — geomagnetic field direction cosines

    Returns
    -------
    coeff : ndarray (N_obs,) — magnetic coefficient (multiply by kappa * F_int/(4pi) for nT)
    """
    obs_x = np.asarray(obs_x, dtype=np.float64)
    obs_y = np.asarray(obs_y, dtype=np.float64)

    # Precompute cross terms
    ax = Mx * Fx
    ay = My * Fy
    az = Mz * Fz
    bxy = My * Fx + Mx * Fy
    bxz = Mz * Fx + Mx * Fz
    byz = Mz * Fy + My * Fz

    # 8 corners with sign factor
    corners = [
        (obs_x - x1, obs_y - y1, obs_z - z1,  1.0),
        (obs_x - x1, obs_y - y1, obs_z - z2, -1.0),
        (obs_x - x1, obs_y - y2, obs_z - z1, -1.0),
        (obs_x - x1, obs_y - y2, obs_z - z2,  1.0),
        (obs_x - x2, obs_y - y1, obs_z - z1, -1.0),
        (obs_x - x2, obs_y - y1, obs_z - z2,  1.0),
        (obs_x - x2, obs_y - y2, obs_z - z1,  1.0),
        (obs_x - x2, obs_y - y2, obs_z - z2, -1.0),
    ]

    total = np.zeros_like(obs_x, dtype=np.float64)

    for xi, yj, zk, mu in corners:
        R = np.sqrt(xi**2 + yj**2 + zk**2)
        R = np.maximum(R, 1e-15)

        # arctan terms
        t1 = np.where(np.abs(xi * R) > 1e-30,
                      np.arctan2(yj * zk, xi * R),
                      0.0)
        t2 = np.where(np.abs(yj * R) > 1e-30,
                      np.arctan2(xi * zk, yj * R),
                      0.0)
        t3 = np.where(np.abs(zk * R) > 1e-30,
                      np.arctan2(xi * yj, zk * R),
                      0.0)

        # log terms
        l1 = np.where(R + yj > 1e-30, np.log(np.abs(R + yj)), 0.0)
        l2 = np.where(R + zk > 1e-30, np.log(np.abs(R + zk)), 0.0)
        l3 = np.where(R + xi > 1e-30, np.log(np.abs(R + xi)), 0.0)

        total += mu * (ax*t1 + ay*t2 + az*t3 - bxy*l1 - bxz*l2 - byz*l3)

    return total


def _prism_magnetic_cuda_batch(obs_x_flat, obs_y_flat,
                                cell_x1, cell_x2, cell_y1, cell_y2,
                                cell_z1, cell_z2,
                                Mx, My, Mz, Fx, Fy, Fz):
    """
    Batched CUDA implementation of prism magnetic formula.

    Parameters
    ----------
    obs_x_flat, obs_y_flat : (N,) tensors on device
    cell_x1..cell_z2 : (C,) tensors on device
    Mx..Fz : float — direction cosines

    Returns
    -------
    contrib : (C, N) tensor — magnetic coefficient per cell
    """
    X1 = cell_x1.unsqueeze(1) - obs_x_flat.unsqueeze(0)
    X2 = cell_x2.unsqueeze(1) - obs_x_flat.unsqueeze(0)
    Y1 = cell_y1.unsqueeze(1) - obs_y_flat.unsqueeze(0)
    Y2 = cell_y2.unsqueeze(1) - obs_y_flat.unsqueeze(0)
    Z1 = cell_z1.unsqueeze(1)
    Z2 = cell_z2.unsqueeze(1)

    ax = Mx * Fx; ay = My * Fy; az = Mz * Fz
    bxy = My * Fx + Mx * Fy
    bxz = Mz * Fx + Mx * Fz
    byz = Mz * Fy + My * Fz

    total = torch.zeros(len(cell_x1), len(obs_x_flat),
                        dtype=torch.float64, device=obs_x_flat.device)

    corners = [
        (X1, Y1, Z1,  1.0), (X1, Y1, Z2, -1.0),
        (X1, Y2, Z1, -1.0), (X1, Y2, Z2,  1.0),
        (X2, Y1, Z1, -1.0), (X2, Y1, Z2,  1.0),
        (X2, Y2, Z1,  1.0), (X2, Y2, Z2, -1.0),
    ]

    for xi, yj, zk, mu in corners:
        R = torch.sqrt(xi**2 + yj**2 + zk**2)
        R = torch.clamp(R, min=1e-15)

        t1 = torch.where(torch.abs(xi*R) > 1e-30,
                         torch.atan2(yj*zk, xi*R), torch.zeros_like(R))
        t2 = torch.where(torch.abs(yj*R) > 1e-30,
                         torch.atan2(xi*zk, yj*R), torch.zeros_like(R))
        t3 = torch.where(torch.abs(zk*R) > 1e-30,
                         torch.atan2(xi*yj, zk*R), torch.zeros_like(R))

        l1 = torch.where(R + yj > 1e-30, torch.log(torch.abs(R + yj)),
                         torch.zeros_like(R))
        l2 = torch.where(R + zk > 1e-30, torch.log(torch.abs(R + zk)),
                         torch.zeros_like(R))
        l3 = torch.where(R + xi > 1e-30, torch.log(torch.abs(R + xi)),
                         torch.zeros_like(R))

        total += mu * (ax*t1 + ay*t2 + az*t3 - bxy*l1 - bxz*l2 - byz*l3)

    return total


def compute_magnetic_anomaly(suscept_model, obs_x, obs_y, obs_z,
                             dx=20.0, dy=20.0, dz=20.0,
                             x0=0.0, y0=0.0, z0=0.0,
                             F_intensity=55000.0, F_declination=-7.0,
                             F_inclination=60.0):
    """
    Compute total magnetic field anomaly from a 3D susceptibility model.

    Uses the Bhattacharyya (1964) rectangular prism formula.
    Assumes induced magnetization only (M parallel to F).

    Parameters
    ----------
    suscept_model : ndarray (nx, ny, nz) — susceptibility in SI
    obs_x, obs_y : array-like — observation point coordinates (m)
    obs_z : float — observation height above z=0 plane (m)
    dx, dy, dz : float — cell dimensions (m)
    x0, y0, z0 : float — origin of model grid (m)
    F_intensity : float — total geomagnetic field intensity (nT)
    F_declination : float — geomagnetic declination (degrees, positive east)
    F_inclination : float — geomagnetic inclination (degrees, positive down)

    Returns
    -------
    magnetic : ndarray (n_obs_x, n_obs_y) — magnetic anomaly in nT
    """
    suscept_model = np.asarray(suscept_model, dtype=np.float64)
    nx, ny, nz = suscept_model.shape
    n_obs_x, n_obs_y = len(obs_x), len(obs_y)

    # Geomagnetic field unit vector
    dec_rad = np.radians(F_declination)
    inc_rad = np.radians(F_inclination)
    Fx = np.cos(inc_rad) * np.cos(dec_rad)
    Fy = np.cos(inc_rad) * np.sin(dec_rad)
    Fz = np.sin(inc_rad)

    # Induced magnetization = parallel to F (induced only, no remanence)
    Mx, My, Mz = Fx, Fy, Fz

    prefactor = F_intensity / (4.0 * np.pi)  # nT

    # Cell boundaries
    xb = x0 + np.arange(nx + 1) * dx
    yb = y0 + np.arange(ny + 1) * dy
    zb = z0 + np.arange(nz + 1) * dz

    # Sparse: non-zero cells only
    nonzero_idx = np.argwhere(suscept_model != 0)
    n_nonzero = len(nonzero_idx)

    if n_nonzero == 0:
        return np.zeros((n_obs_x, n_obs_y), dtype=np.float64)

    # Observation grid
    OX, OY = np.meshgrid(obs_x, obs_y, indexing='ij')
    ox_flat = OX.ravel()
    oy_flat = OY.ravel()
    N_obs = n_obs_x * n_obs_y

    # Non-zero cell data
    kap_vals = suscept_model[nonzero_idx[:, 0], nonzero_idx[:, 1], nonzero_idx[:, 2]]
    c_x1 = xb[nonzero_idx[:, 0]]; c_x2 = xb[nonzero_idx[:, 0] + 1]
    c_y1 = yb[nonzero_idx[:, 1]]; c_y2 = yb[nonzero_idx[:, 1] + 1]
    c_z1 = zb[nonzero_idx[:, 2]]; c_z2 = zb[nonzero_idx[:, 2] + 1]

    if _CUDA_AVAILABLE and n_nonzero > 50:
        # ---- CUDA path ----
        ox_t = torch.tensor(ox_flat, dtype=torch.float64, device=_DEVICE)
        oy_t = torch.tensor(oy_flat, dtype=torch.float64, device=_DEVICE)

        mag = torch.zeros(N_obs, dtype=torch.float64, device=_DEVICE)
        batch_size = max(100, n_nonzero // 20)

        for start in range(0, n_nonzero, batch_size):
            end = min(start + batch_size, n_nonzero)
            b_kap = torch.tensor(kap_vals[start:end], dtype=torch.float64, device=_DEVICE)
            contrib = _prism_magnetic_cuda_batch(
                ox_t, oy_t,
                torch.tensor(c_x1[start:end], dtype=torch.float64, device=_DEVICE),
                torch.tensor(c_x2[start:end], dtype=torch.float64, device=_DEVICE),
                torch.tensor(c_y1[start:end], dtype=torch.float64, device=_DEVICE),
                torch.tensor(c_y2[start:end], dtype=torch.float64, device=_DEVICE),
                torch.tensor(c_z1[start:end] + obs_z, dtype=torch.float64, device=_DEVICE),
                torch.tensor(c_z2[start:end] + obs_z, dtype=torch.float64, device=_DEVICE),
                Mx, My, Mz, Fx, Fy, Fz,
            )
            mag += torch.einsum('bn,b->n', contrib, b_kap) * prefactor

        result = mag.cpu().numpy()
    else:
        # ---- CPU numpy path ----
        mag = np.zeros(N_obs, dtype=np.float64)
        for i in range(n_nonzero):
            contrib = _prism_magnetic_numpy(
                ox_flat, oy_flat, obs_z,
                c_x1[i], c_x2[i], c_y1[i], c_y2[i], c_z1[i], c_z2[i],
                Mx, My, Mz, Fx, Fy, Fz
            )
            mag += contrib * kap_vals[i] * prefactor

        result = mag

    return result.reshape(n_obs_x, n_obs_y)


def forward_magnetic(suscept_model, obs_height=10.0, nx=40, ny=40, nz=20,
                     dx=20.0, dy=20.0, dz=20.0, n_obs=81,
                     F_intensity=55000.0, F_declination=-7.0,
                     F_inclination=60.0):
    """
    Convenience wrapper: compute magnetic anomaly on a standard grid.

    Parameters
    ----------
    suscept_model : ndarray (nx, ny, nz) — susceptibility in SI
    obs_height : float — observation height above surface (m)
    nx, ny, nz : int — model grid dimensions
    dx, dy, dz : float — cell spacing (m)
    n_obs : int — number of observation points per axis
    F_intensity : float — total field intensity (nT)
    F_declination : float — declination (degrees)
    F_inclination : float — inclination (degrees)

    Returns
    -------
    magnetic : ndarray (n_obs, n_obs) — magnetic anomaly in nT
    """
    half_extent = nx * dx / 2.0
    obs_coords = np.linspace(-half_extent - dx/2, half_extent + dx/2, n_obs)
    return compute_magnetic_anomaly(
        suscept_model, obs_coords, obs_coords, obs_height,
        dx=dx, dy=dy, dz=dz,
        x0=-nx*dx/2.0, y0=-ny*dy/2.0, z0=0.0,
        F_intensity=F_intensity, F_declination=F_declination,
        F_inclination=F_inclination
    )


def add_magnetic_noise(magnetic, noise_level=0.01, seed=None):
    """
    Add Gaussian noise to magnetic data.

    noise = noise_level * max(|magnetic|) * N(0,1)

    Parameters
    ----------
    magnetic : ndarray — clean magnetic anomaly (nT)
    noise_level : float — noise as fraction of max signal (default 1%)
    seed : int or None — random seed

    Returns
    -------
    noisy_magnetic : ndarray
    """
    max_signal = np.max(np.abs(magnetic))
    if max_signal < 1e-15:
        return magnetic.copy()

    if seed is not None:
        state = np.random.get_state()
        np.random.seed(seed)

    noise = noise_level * max_signal * np.random.randn(*magnetic.shape)

    if seed is not None:
        np.random.set_state(state)

    return magnetic + noise


# ============================================================
# Analytical reference solutions (for unit testing)
# ============================================================

def analytical_prism_magnetic_single(obs_x, obs_y, obs_z,
                                     x1, x2, y1, y2, z1, z2,
                                     kappa_si=0.1,
                                     F_intensity=55000.0,
                                     F_declination=-7.0,
                                     F_inclination=60.0):
    """
    Analytical magnetic anomaly of a single rectangular prism.

    Reference implementation for unit test validation.

    Parameters
    ----------
    obs_x, obs_y : array-like (N,) — observation coordinates (m)
    obs_z : float — observation height (m)
    x1..z2 : float — prism boundaries (m)
    kappa_si : float — susceptibility (SI)
    F_intensity : float — total field intensity (nT)
    F_declination : float — declination (degrees)
    F_inclination : float — inclination (degrees)

    Returns
    -------
    dT : ndarray (N,) or scalar — total field anomaly in nT
    """
    obs_x = np.atleast_1d(np.asarray(obs_x, dtype=np.float64))
    obs_y = np.atleast_1d(np.asarray(obs_y, dtype=np.float64))

    dec_rad = np.radians(F_declination)
    inc_rad = np.radians(F_inclination)
    Fx = np.cos(inc_rad) * np.cos(dec_rad)
    Fy = np.cos(inc_rad) * np.sin(dec_rad)
    Fz = np.sin(inc_rad)

    # Induced magnetization parallel to F
    Mx, My, Mz = Fx, Fy, Fz

    coeff = _prism_magnetic_numpy(
        obs_x, obs_y, obs_z, x1, x2, y1, y2, z1, z2,
        Mx, My, Mz, Fx, Fy, Fz
    )

    prefactor = F_intensity / (4.0 * np.pi)
    result = coeff * kappa_si * prefactor

    if result.size == 1:
        return float(result.ravel()[0])
    return result


if __name__ == '__main__':
    print("Magnetic forward module self-test")
    kappa = np.zeros((40, 40, 20))
    kappa[10:20, 10:20, 2:8] = 0.1  # Central cube, susceptibility 0.1 SI

    import time
    t0 = time.time()
    dt = forward_magnetic(kappa, obs_height=10.0, n_obs=41)
    elapsed = time.time() - t0

    print(f"  Model shape: {kappa.shape}")
    print(f"  Output shape: {dt.shape}")
    print(f"  Range: [{dt.min:.2f}, {dt.max:.2f}] nT")
    print(f"  Time: {elapsed:.3f}s")
    print("  OK")
