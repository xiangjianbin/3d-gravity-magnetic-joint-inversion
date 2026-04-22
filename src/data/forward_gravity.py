"""
3D Prism Gravity Forward Modeling (Nagy et al., 2000 / Talwani et al., 1959)

Computes gravity anomaly of a 3D density model discretized into rectangular prisms.
Supports CUDA acceleration via PyTorch for batched computation.

Units:
  - Input density: g/cm^3 (converted internally to kg/m^3)
  - Output: mGal (1 mGal = 10^-5 m/s^2)
  - G = 6.67430e-11 m^3 kg^-1 s^-2

Formula (Nagy 2000, Eq. 4):
  dg = G * rho * sum_{i=1}^{2} sum_{j=1}^{2} sum_{k=1}^{2}
        mu_ijk * [ z_k * arctan(x_i*y_j / (z_k*R))
                 - x_i * arctan(y_j*z_k / (x_i*R))
                 - y_j * arctan(x_i*z_k / (y_j*R)) ]

  where mu_ijk = (-1)^(i+j+k), R = sqrt(x_i^2 + y_j^2 + z_k^2)
  and (x_i, y_j, z_k) are coordinates of observation point relative to prism corners.
"""

import numpy as np
import torch

# Physical constants
G = 6.67430e-11  # Gravitational constant (m^3 kg^-1 s^-2)

# Device selection
_CUDA_AVAILABLE = torch.cuda.is_available()
_DEVICE = torch.device('cuda' if _CUDA_AVAILABLE else 'cpu')


def _prism_gravity_numpy(obs_x, obs_y, obs_z, x1, x2, y1, y2, z1, z2):
    """
    Pure NumPy implementation of Nagy (2000) prism gravity formula.

    Parameters
    ----------
    obs_x, obs_y : array-like (N_obs,) — observation point coordinates (m)
    obs_z : float — observation height above surface (m), positive upward
    x1, x2, y1, y2, z1, z2 : float — prism corner coordinates (m)

    Returns
    -------
    dg : ndarray (N_obs,) — gravity contribution in m/s^2 (before G*rho scaling)
    """
    obs_x = np.asarray(obs_x, dtype=np.float64)
    obs_y = np.asarray(obs_y, dtype=np.float64)

    # Relative coordinates: obs - corner
    # Corner ordering: (i,j,k) where i,j,k in {0,1}
    # xi = x_obs - x_corner, sign flips with corner index
    corners = [
        (obs_x - x1, obs_y - y1, obs_z - z1,  1),   # (0,0,0) mu=+1
        (obs_x - x1, obs_y - y1, obs_z - z2, -1),   # (0,0,1) mu=-1
        (obs_x - x1, obs_y - y2, obs_z - z1, -1),   # (0,1,0) mu=-1
        (obs_x - x1, obs_y - y2, obs_z - z2,  1),   # (0,1,1) mu=+1
        (obs_x - x2, obs_y - y1, obs_z - z1, -1),   # (1,0,0) mu=-1
        (obs_x - x2, obs_y - y1, obs_z - z2,  1),   # (1,0,1) mu=+1
        (obs_x - x2, obs_y - y2, obs_z - z1,  1),   # (1,1,0) mu=+1
        (obs_x - x2, obs_y - y2, obs_z - z2, -1),   # (1,1,1) mu=-1
    ]

    total = np.zeros_like(obs_x, dtype=np.float64)

    for xi, yj, zk, mu in corners:
        R = np.sqrt(xi**2 + yj**2 + zk**2)
        R = np.maximum(R, 1e-15)  # Avoid division by zero

        # Safe arctan2: handle near-zero denominators
        t1 = np.where(np.abs(zk * R) > 1e-30,
                      zk * np.arctan2(xi * yj, zk * R),
                      0.0)
        t2 = np.where(np.abs(xi * R) > 1e-30,
                      xi * np.arctan2(yj * zk, xi * R),
                      0.0)
        t3 = np.where(np.abs(yj * R) > 1e-30,
                      yj * np.arctan2(xi * zk, yj * R),
                      0.0)

        total += mu * (t1 - t2 - t3)

    return -total  # Negate for z-down convention


def _prism_gravity_cuda_batch(obs_x_flat, obs_y_flat, cell_x1, cell_x2,
                               cell_y1, cell_y2, cell_z1, cell_z2):
    """
    Batched CUDA implementation: compute gravity from C cells at N observation points.

    Parameters
    ----------
    obs_x_flat, obs_y_flat : (N,) tensors on device
    cell_x1..cell_z2 : (C,) tensors on device (prism boundaries in meters)

    Returns
    -------
    contrib : (C, N) tensor — gravity coefficient per cell (multiply by G*rho for m/s^2)
    """
    # Broadcast to (C, N)
    X1 = cell_x1.unsqueeze(1) - obs_x_flat.unsqueeze(0)
    X2 = cell_x2.unsqueeze(1) - obs_x_flat.unsqueeze(0)
    Y1 = cell_y1.unsqueeze(1) - obs_y_flat.unsqueeze(0)
    Y2 = cell_y2.unsqueeze(1) - obs_y_flat.unsqueeze(0)
    Z1 = cell_z1.unsqueeze(1)
    Z2 = cell_z2.unsqueeze(1)

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

        t1 = torch.where(torch.abs(zk * R) > 1e-30,
                         zk * torch.atan2(xi * yj, zk * R),
                         torch.zeros_like(R))
        t2 = torch.where(torch.abs(xi * R) > 1e-30,
                         xi * torch.atan2(yj * zk, xi * R),
                         torch.zeros_like(R))
        t3 = torch.where(torch.abs(yj * R) > 1e-30,
                         yj * torch.atan2(xi * zk, yj * R),
                         torch.zeros_like(R))

        total += mu * (t1 - t2 - t3)

    return -total


def compute_gravity_anomaly(density_model, obs_x, obs_y, obs_z,
                            dx=20.0, dy=20.0, dz=20.0,
                            x0=0.0, y0=0.0, z0=0.0):
    """
    Compute gravity anomaly field from a 3D density model.

    Uses the Nagy et al. (2000) rectangular prism formula.
    Automatically selects CUDA (batched) or CPU (numpy) backend.

    Parameters
    ----------
    density_model : ndarray (nx, ny, nz) — density in g/cm^3
    obs_x, obs_y : array-like — observation point coordinates (m)
    obs_z : float — observation height above z=0 plane (m)
    dx, dy, dz : float — cell dimensions (m)
    x0, y0, z0 : float — origin of the model grid (m)

    Returns
    -------
    gravity : ndarray (n_obs_x, n_obs_y) — gravity anomaly in mGal
    """
    density_model = np.asarray(density_model, dtype=np.float64)
    nx, ny, nz = density_model.shape
    n_obs_x, n_obs_y = len(obs_x), len(obs_y)

    # Convert density from g/cm^3 to kg/m^3
    density_kgm3 = density_model * 1000.0

    # Cell boundary coordinates
    xb = x0 + np.arange(nx + 1) * dx
    yb = y0 + np.arange(ny + 1) * dy
    zb = z0 + np.arange(nz + 1) * dz

    # Find non-zero cells only (sparse optimization)
    nonzero_idx = np.argwhere(density_model != 0)
    n_nonzero = len(nonzero_idx)

    if n_nonzero == 0:
        return np.zeros((n_obs_x, n_obs_y), dtype=np.float64)

    # Observation grid flattened
    OX, OY = np.meshgrid(obs_x, obs_y, indexing='ij')
    ox_flat = OX.ravel()
    oy_flat = OY.ravel()
    N_obs = n_obs_x * n_obs_y

    # Extract non-zero cell data
    rho_vals = density_kgm3[nonzero_idx[:, 0], nonzero_idx[:, 1], nonzero_idx[:, 2]]
    c_x1 = xb[nonzero_idx[:, 0]]
    c_x2 = xb[nonzero_idx[:, 0] + 1]
    c_y1 = yb[nonzero_idx[:, 1]]
    c_y2 = yb[nonzero_idx[:, 1] + 1]
    c_z1 = zb[nonzero_idx[:, 2]]
    c_z2 = zb[nonzero_idx[:, 2] + 1]

    if _CUDA_AVAILABLE and n_nonzero > 50:
        # ---- CUDA batched path ----
        ox_t = torch.tensor(ox_flat, dtype=torch.float64, device=_DEVICE)
        oy_t = torch.tensor(oy_flat, dtype=torch.float64, device=_DEVICE)

        grav = torch.zeros(N_obs, dtype=torch.float64, device=_DEVICE)
        batch_size = max(100, n_nonzero // 20)

        for start in range(0, n_nonzero, batch_size):
            end = min(start + batch_size, n_nonzero)
            b_rho = torch.tensor(rho_vals[start:end], dtype=torch.float64, device=_DEVICE)
            contrib = _prism_gravity_cuda_batch(
                ox_t, oy_t,
                torch.tensor(c_x1[start:end], dtype=torch.float64, device=_DEVICE),
                torch.tensor(c_x2[start:end], dtype=torch.float64, device=_DEVICE),
                torch.tensor(c_y1[start:end], dtype=torch.float64, device=_DEVICE),
                torch.tensor(c_y2[start:end], dtype=torch.float64, device=_DEVICE),
                torch.tensor(c_z1[start:end] + obs_z, dtype=torch.float64, device=_DEVICE),
                torch.tensor(c_z2[start:end] + obs_z, dtype=torch.float64, device=_DEVICE),
            )
            grav += torch.einsum('bn,b->n', contrib, b_rho)

        result = (-grav * G * 1e5).cpu().numpy()  # m/s^2 -> mGal
    else:
        # ---- CPU numpy path ----
        grav = np.zeros(N_obs, dtype=np.float64)
        for i in range(n_nonzero):
            contrib = _prism_gravity_numpy(
                ox_flat, oy_flat, obs_z,
                c_x1[i], c_x2[i], c_y1[i], c_y2[i], c_z1[i], c_z2[i]
            )
            grav += contrib * rho_vals[i]

        result = (-grav * G * 1e5)  # m/s^2 -> mGal

    return result.reshape(n_obs_x, n_obs_y)


def forward_gravity(density_model, obs_height=10.0, nx=40, ny=40, nz=20,
                    dx=20.0, dy=20.0, dz=20.0, n_obs=81):
    """
    Convenience wrapper: compute gravity anomaly on a standard grid.

    Grid covers [-nx*dx/2, nx*dx/2] in x and y.
    Observation points form an n_obs x n_obs grid centered on the model.

    Parameters
    ----------
    density_model : ndarray (nx, ny, nz) — density in g/cm^3
    obs_height : float — observation height above surface (m)
    nx, ny, nz : int — model grid dimensions
    dx, dy, dz : float — cell spacing (m)
    n_obs : int — number of observation points per axis

    Returns
    -------
    gravity : ndarray (n_obs, n_obs) — gravity anomaly in mGal
    """
    half_extent = nx * dx / 2.0
    obs_coords = np.linspace(-half_extent - dx/2, half_extent + dx/2, n_obs)
    return compute_gravity_anomaly(
        density_model, obs_coords, obs_coords, obs_height,
        dx=dx, dy=dy, dz=dz,
        x0=-nx*dx/2.0, y0=-ny*dy/2.0, z0=0.0
    )


def add_gravity_noise(gravity, noise_level=0.005, seed=None):
    """
    Add Gaussian noise to gravity data.

    noise = noise_level * max(|gravity|) * N(0,1)

    Parameters
    ----------
    gravity : ndarray — clean gravity anomaly (mGal)
    noise_level : float — noise as fraction of max signal (default 0.5%)
    seed : int or None — random seed for reproducibility

    Returns
    -------
    noisy_gravity : ndarray — same shape as input
    """
    max_signal = np.max(np.abs(gravity))
    if max_signal < 1e-15:
        return gravity.copy()

    if seed is not None:
        state = np.random.get_state()
        np.random.seed(seed)

    noise = noise_level * max_signal * np.random.randn(*gravity.shape)

    if seed is not None:
        np.random.set_state(state)

    return gravity + noise


# ============================================================
# Analytical reference solutions (for unit testing)
# ============================================================

def analytical_prism_gravity_single(obs_x, obs_y, obs_z,
                                    x1, x2, y1, y2, z1, z2,
                                    density_gcc=1.0):
    """
    Analytical gravity of a single rectangular prism at observation points.

    This is the reference implementation used for unit test validation.
    Uses the exact Nagy (2000) formula in pure Python/numpy.

    Parameters
    ----------
    obs_x, obs_y : array-like (N,) — observation coordinates (m)
    obs_z : float — observation height (m)
    x1..z2 : float — prism boundaries (m)
    density_gcc : float — density in g/cm^3

    Returns
    -------
    dg : ndarray (N,) or scalar — gravity anomaly in mGal
    """
    obs_x = np.atleast_1d(np.asarray(obs_x, dtype=np.float64))
    obs_y = np.atleast_1d(np.asarray(obs_y, dtype=np.float64))

    contrib = _prism_gravity_numpy(obs_x, obs_y, obs_z, x1, x2, y1, y2, z1, z2)
    density_kgm3 = density_gcc * 1000.0
    result = -contrib * G * density_kgm3 * 1e5  # mGal

    if result.size == 1:
        return float(result.ravel()[0])
    return result


if __name__ == '__main__':
    # Quick smoke test
    print("Gravity forward module self-test")
    rho = np.zeros((40, 40, 20))
    rho[10:20, 10:20, 2:8] = 1.0  # Central cube, density 1 g/cc

    import time
    t0 = time.time()
    gz = forward_gravity(rho, obs_height=10.0, n_obs=41)
    elapsed = time.time() - t0

    print(f"  Model shape: {rho.shape}")
    print(f"  Output shape: {gz.shape}")
    print(f"  Range: [{gz.min():.4f}, {gz.max():.4f}] mGal")
    print(f"  Time: {elapsed:.3f}s")
    print("  OK")
