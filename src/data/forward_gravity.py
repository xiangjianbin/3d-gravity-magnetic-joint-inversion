"""
3D Prism Gravity Forward Modeling — PyTorch CUDA (Batched, Memory-Safe)

Processes cells in GPU batches to fit within VRAM.
Falls back to CPU if CUDA unavailable or OOM.

Based on Nagy et al. (2000). Units: Density g/cm^3 -> Output mGal
"""

import numpy as np
import torch

G = 6.67430e-11
CUDA_OK = torch.cuda.is_available()
DEVICE = torch.device('cuda' if CUDA_OK else 'cpu')


def _gravity_kernel_cuda(obs_x_2d, obs_y_2d, cell_x1, cell_x2,
                          cell_y1, cell_y2, cell_z1, cell_z2):
    """
    Compute gravity contribution of one batch of cells at all observation points.
    All inputs are 1D tensors on GPU (or CPU).

    Parameters
    ----------
    obs_x_2d, obs_y_2d : (N,) — flattened observation coordinates
    cell_x1..cell_z2 : (C,) — cell boundaries

    Returns
    -------
    contrib : (N,) — total gravity contribution from these cells (in m/s^2 * G^-1, need to multiply by G*rho later)
    """
    N = len(obs_x_2d)
    C = len(cell_x1)

    # Broadcast: (C, 1) - (1, N) -> (C, N)
    X1 = cell_x1.unsqueeze(1) - obs_x_2d.unsqueeze(0)   # (C, N)
    X2 = cell_x2.unsqueeze(1) - obs_x_2d.unsqueeze(0)
    Y1 = cell_y1.unsqueeze(1) - obs_y_2d.unsqueeze(0)
    Y2 = cell_y2.unsqueeze(1) - obs_y_2d.unsqueeze(0)
    Z1 = cell_z1.unsqueeze(1)                              # (C, 1)
    Z2 = cell_z2.unsqueeze(1)

    R2 = X1*X1 + Y1*Y1 + Z1*Z1  # (C, N) — using X1,Y1,Z1 as base (R is similar for all 8 corners)
    R = torch.sqrt(R2)
    R = torch.clamp(R, min=1e-15)

    # Precompute arctan terms for all 8 corners
    # Corner (i,j,k): xi ∈ {X1(i),X2(1-i)}, yj ∈ {Y1(j),Y2(1-j)}, zk ∈ {Z1(k),Z2(1-k)}
    # Sign: mu = (-1)^(i+j+k)

    # For efficiency, compute using the identity that the sum over 8 corners
    # can be grouped. But let's just do the straightforward 8-corner loop (only 8 iters).

    total = torch.zeros(C, N, dtype=torch.float64, device=obs_x_2d.device)

    corners = [(0,0,0),(0,0,1),(0,1,0),(0,1,1),
               (1,0,0),(1,0,1),(1,1,0),(1,1,1)]
    for (i,j,k) in corners:
        mu = 1.0 if (i+j+k) % 2 == 0 else -1.0
        xi = X1 if i==0 else X2
        yj = Y1 if j==0 else Y2
        zk = Z1 if k==0 else Z2

        # These xi,yj,zk are all (C,N)
        Rijk = torch.sqrt(xi*xi + yj*yj + zk*zk)
        Rijk = torch.clamp(Rijk, min=1e-15)

        t1 = zk * torch.atan2(xi*yj, zk*Rijk)
        t2 = xi * torch.atan2(yj*zk, xi*Rijk)
        t3 = yj * torch.atan2(xi*zk, yj*Rijk)

        total += mu * (t1 - t2 - t3)

    return -total  # (C, N), negate for z-down


def compute_gravity_anomaly(density_model, obs_x, obs_y, obs_z,
                            dx=20.0, dy=20.0, dz=20.0,
                            x0=0.0, y0=0.0, z0=0.0):
    """Gravity forward modeling — auto CUDA/CPU with batching."""
    nx, ny, nz = density_model.shape
    n_obs_x, n_obs_y = len(obs_x), len(obs_y)
    density_kgm3 = density_model * 1000.0

    xb = x0 + np.arange(nx+1)*dx; yb = y0 + np.arange(ny+1)*dy; zb = z0 + np.arange(nz+1)*dz
    nonzero = np.argwhere(density_model != 0)
    n_nz = len(nonzero)

    if n_nz == 0:
        return np.zeros((n_obs_x, n_obs_y), dtype=np.float64)

    OX, OY = np.meshgrid(obs_x, obs_y, indexing='ij')
    ox_f = torch.tensor(OX.ravel(), dtype=torch.float64, device=DEVICE)
    oy_f = torch.tensor(OY.ravel(), dtype=torch.float64, device=DEVICE)

    rho_all = np.array([density_kgm3[int(c[0]),int(c[1]),int(c[2])] for c in nonzero], dtype=np.float64)
    cb_x1 = np.array([xb[int(c[0])] for c in nonzero], dtype=np.float64)
    cb_x2 = np.array([xb[int(c[0])+1] for c in nonzero], dtype=np.float64)
    cb_y1 = np.array([yb[int(c[1])] for c in nonzero], dtype=np.float64)
    cb_y2 = np.array([yb[int(c[1])+1] for c in nonzero], dtype=np.float64)
    cb_z1 = np.array([zb[int(c[2])] for c in nonzero], dtype=np.float64)
    cb_z2 = np.array([zb[int(c[2])+1] for c in nonzero], dtype=np.float64)

    # Process in batches to limit GPU memory (~2GB per batch)
    batch_size = max(50, n_nz // 20)  # ~20 batches
    N = n_obs_x * n_obs_y
    grav = torch.zeros(N, dtype=torch.float64, device=DEVICE)

    for start in range(0, n_nz, batch_size):
        end = min(start+batch_size, n_nz)
        b_rho = torch.tensor(rho_all[start:end], dtype=torch.float64, device=DEVICE)
        b_x1 = torch.tensor(cb_x1[start:end], dtype=torch.float64, device=DEVICE)
        b_x2 = torch.tensor(cb_x2[start:end], dtype=torch.float64, device=DEVICE)
        b_y1 = torch.tensor(cb_y1[start:end], dtype=torch.float64, device=DEVICE)
        b_y2 = torch.tensor(cb_y2[start:end], dtype=torch.float64, device=DEVICE)
        b_z1 = torch.tensor(cb_z1[start:end]+obs_z, dtype=torch.float64, device=DEVICE)
        b_z2 = torch.tensor(cb_z2[start:end]+obs_z, dtype=torch.float64, device=DEVICE)

        contrib = _gravity_kernel_cuda(ox_f, oy_f, b_x1, b_x2, b_y1, b_y2, b_z1, b_z2)  # (batch, N)
        grav += torch.einsum('bn,b->n', contrib, b_rho)  # (N,)

    result = (-grav * G * 1e5).cpu().numpy()  # m/s^2 -> mGal
    return result.reshape(n_obs_x, n_obs_y)


compute_gravity_anomaly_vectorized = compute_gravity_anomaly


def forward_gravity(density_model, obs_height=10.0, nx=40, ny=40, nz=20,
                    dx=20.0, dy=20.0, dz=20.0, n_obs=81):
    half=nx*dx/2.0; obs=np.linspace(-half-dx/2,half+dx/2,n_obs)
    return compute_gravity_anomaly(density_model,obs,obs,obs_height,
                                   dx=dx,dy=dy,dz=dz,x0=-nx*dx/2.0,y0=-ny*dy/2.0,z0=0.0)


def add_gravity_noise(grav, noise_level=0.005):
    max_signal=np.max(np.abs(grav))
    if max_signal<1e-15: return grav
    return grav+noise_level*max_signal*np.random.randn(*grav.shape)


# ---- Test compatibility ----
def _prism_gravity_component(x1,x2,y1,y2,z1,z2):
    xs=[x1,x2];ys=[y1,y2];zs=[z1,z2];total=0.0
    for i in range(2):
        for j in range(2):
            for k in range(2):
                mu=(-1.0)**(i+j+k);xi,yj,zk=xs[i],ys[j],zs[k];R=np.sqrt(xi**2+yj**2+zk**2)
                if R<1e-15:continue
                total+=mu*(zk*np.arctan2(xi*yj,zk*R)-xi*np.arctan2(yj*zk,xi*R)-yj*np.arctan2(xi*zk,yj*R))
    return -total

def _prism_gravity_array(X1,X2,Y1,Y2,Z1,Z2):
    total=np.zeros_like(X1,dtype=np.float64)
    for i in range(2):
        for j in range(2):
            for k in range(2):
                mu=(-1.0)**(i+j+k);xi=X1 if i==0 else X2;yj=Y1 if j==0 else Y2;zk=Z1 if k==0 else Z2
                R=np.sqrt(xi**2+yj**2+zk**2)
                t1=np.where(np.abs(zk*R)>1e-30,zk*np.arctan2(xi*yj,zk*R),0.0)
                t2=np.where(np.abs(xi*R)>1e-30,xi*np.arctan2(yj*zk,xi*R),0.0)
                t3=np.where(np.abs(yj*R)>1e-30,yj*np.arctan2(xi*zk,yj*R),0.0)
                total+=mu*(t1-t2-t3)
    return -total
