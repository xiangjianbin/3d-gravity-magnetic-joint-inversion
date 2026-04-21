"""
3D Prism Magnetic Anomaly Forward Modeling — PyTorch CUDA (Batched)

Based on Bhattacharyya (1964). Units: Susceptibility SI, Output nT
"""

import numpy as np
import torch

MU_0 = 4.0 * np.pi * 1e-7
CUDA_OK = torch.cuda.is_available()
DEVICE = torch.device('cuda' if CUDA_OK else 'cpu')


def _magnetic_kernel_cuda(obs_x_2d, obs_y_2d, cell_x1, cell_x2,
                            cell_y1, cell_y2, cell_z1, cell_z2,
                            Mx, My, Mz, Fx, Fy, Fz):
    """
    Compute magnetic contribution of one batch of cells at all observation points.
    Returns (C, N) coefficient array (multiply by kappa * prefactor for nT).
    """
    C = len(cell_x1)

    X1 = cell_x1.unsqueeze(1) - obs_x_2d.unsqueeze(0)
    X2 = cell_x2.unsqueeze(1) - obs_x_2d.unsqueeze(0)
    Y1 = cell_y1.unsqueeze(1) - obs_y_2d.unsqueeze(0)
    Y2 = cell_y2.unsqueeze(1) - obs_y_2d.unsqueeze(0)
    Z1 = cell_z1.unsqueeze(1)
    Z2 = cell_z2.unsqueeze(1)

    ax=Mx*Fx; ay=My*Fy; az=Mz*Fz
    bxy=My*Fx+Mx*Fy; bxz=Mz*Fx+Mx*Fz; byz=Mz*Fy+My*Fz

    total = torch.zeros(C, len(obs_x_2d), dtype=torch.float64, device=obs_x_2d.device)

    corners = [(0,0,0),(0,0,1),(0,1,0),(0,1,1),
               (1,0,0),(1,0,1),(1,1,0),(1,1,1)]
    for (i,j,k) in corners:
        mu = 1.0 if (i+j+k)%2==0 else -1.0
        xi = X1 if i==0 else X2
        yj = Y1 if j==0 else Y2
        zk = Z1 if k==0 else Z2

        R = torch.sqrt(xi*xi + yj*yj + zk*zk); R = torch.clamp(R, min=1e-15)

        a1 = torch.where(torch.abs(xi*R)>1e-30, torch.atan2(yj*zk, xi*R), torch.zeros_like(R))
        a2 = torch.where(torch.abs(yj*R)>1e-30, torch.atan2(xi*zk, yj*R), torch.zeros_like(R))
        a3 = torch.where(torch.abs(zk*R)>1e-30, torch.atan2(xi*yj, zk*R), torch.zeros_like(R))

        l1 = torch.where(R+yj>1e-30, torch.log(torch.abs(R+yj)), torch.zeros_like(R))
        l2 = torch.where(R+zk>1e-30, torch.log(torch.abs(R+zk)), torch.zeros_like(R))
        l3 = torch.where(R+xi>1e-30, torch.log(torch.abs(R+xi)), torch.zeros_like(R))

        total += mu * (ax*a1 + ay*a2 + az*a3 - bxy*l1 - bxz*l2 - byz*l3)

    return total  # (C, N)


def compute_magnetic_anomaly(suscept_model, obs_x, obs_y, obs_z,
                             dx=20.0, dy=20.0, dz=20.0,
                             x0=0.0, y0=0.0, z0=0.0,
                             F_intensity=55000.0, F_declination=-7.0,
                             F_inclination=60.0):
    """Magnetic forward — auto CUDA/CPU with batching."""
    nx,ny,nz=suscept_model.shape; n_obs_x,n_obs_y=len(obs_x),len(obs_y)

    dec_rad=np.radians(F_declination); inc_rad=np.radians(F_inclination)
    Fx=np.cos(inc_rad)*np.cos(dec_rad); Fy=np.cos(inc_rad)*np.sin(dec_rad); Fz=np.sin(inc_rad)
    Mx,My,Mz=Fx,Fy,Fz
    prefactor=F_intensity/(4.0*np.pi)

    xb=x0+np.arange(nx+1)*dx; yb=y0+np.arange(ny+1)*dy; zb=z0+np.arange(nz+1)*dz
    nonzero=np.argwhere(suscept_model!=0); n_nz=len(nonzero)
    if n_nz==0: return np.zeros((n_obs_x,n_obs_y),dtype=np.float64)

    OX,OY=np.meshgrid(obs_x,obs_y,indexing='ij')
    ox_f=torch.tensor(OX.ravel(),dtype=torch.float64,device=DEVICE)
    oy_f=torch.tensor(OY.ravel(),dtype=torch.float64,device=DEVICE)

    kap_all=np.array([suscept_model[int(c[0]),int(c[1]),int(c[2])] for c in nonzero],dtype=np.float64)
    cb_x1=np.array([xb[int(c[0])] for c in nonzero],dtype=np.float64)
    cb_x2=np.array([xb[int(c[0])+1] for c in nonzero],dtype=np.float64)
    cb_y1=np.array([yb[int(c[1])] for c in nonzero],dtype=np.float64)
    cb_y2=np.array([yb[int(c[1])+1] for c in nonzero],dtype=np.float64)
    cb_z1=np.array([zb[int(c[2])] for c in nonzero],dtype=np.float64)
    cb_z2=np.array([zb[int(c[2])+1] for c in nonzero],dtype=np.float64)

    batch_size=max(50, n_nz//20); N=n_obs_x*n_obs_y
    mag=torch.zeros(N,dtype=torch.float64,device=DEVICE)

    for start in range(0, n_nz, batch_size):
        end=min(start+batch_size, n_nz)
        b_kap=torch.tensor(kap_all[start:end],dtype=torch.float64,device=DEVICE)
        b_x1=torch.tensor(cb_x1[start:end],dtype=torch.float64,device=DEVICE)
        b_x2=torch.tensor(cb_x2[start:end],dtype=torch.float64,device=DEVICE)
        b_y1=torch.tensor(cb_y1[start:end],dtype=torch.float64,device=DEVICE)
        b_y2=torch.tensor(cb_y2[start:end],dtype=torch.float64,device=DEVICE)
        b_z1=torch.tensor(cb_z1[start:end]+obs_z,dtype=torch.float64,device=DEVICE)
        b_z2=torch.tensor(cb_z2[start:end]+obs_z,dtype=torch.float64,device=DEVICE)
        contrib=_magnetic_kernel_cuda(ox_f,oy_f,b_x1,b_x2,b_y1,b_y2,b_z1,b_z2,
                                       Mx,My,Mz,Fx,Fy,Fz)
        mag+=torch.einsum('bn,b->n',contrib,b_kap)*prefactor

    return mag.cpu().numpy().reshape(n_obs_x,n_obs_y)


compute_magnetic_anomaly_vectorized=compute_magnetic_anomaly


def forward_magnetic(suscept_model, obs_height=10.0, nx=40, ny=40, nz=20,
                     dx=20.0, dy=20.0, dz=20.0, n_obs=81,
                     F_intensity=55000.0, F_declination=-7.0,
                     F_inclination=60.0):
    half=nx*dx/2.0; obs=np.linspace(-half-dx/2,half+dx/2,n_obs)
    return compute_magnetic_anomaly(suscept_model,obs,obs,obs_height,
                                    dx=dx,dy=dy,dz=dz,x0=-nx*dx/2.0,y0=-ny*dy/2.0,z0=0.0,
                                    F_intensity=F_intensity,F_declination=F_declination,
                                    F_inclination=F_inclination)


def add_magnetic_noise(mag, noise_level=0.01):
    max_signal=np.max(np.abs(mag))
    if max_signal<1e-15: return mag
    return mag+noise_level*max_signal*np.random.randn(*mag.shape)


# ---- Test compatibility ----
def _magnetic_field_array(X1,X2,Y1,Y2,Z1,Z2,Mx,My,Mz,Fx,Fy,Fz):
    ax,ay,az=Mx*Fx,My*Fy,Mz*Fz;bxy=My*Fx+Mx*Fy;bxz=Mz*Fx+Mx*Fz;byz=Mz*Fy+My*Fz
    total=np.zeros_like(X1,dtype=np.float64)
    for i in range(2):
        for j in range(2):
            for k in range(2):
                mu=(-1.0)**(i+j+k);xi=X1 if i==0 else X2;yj=Y1 if j==0 else Y2;zk=Z1 if k==0 else Z2
                R=np.sqrt(xi**2+yj**2+zk**2)
                t1=np.where(np.abs(xi*R)>1e-30,np.arctan2(yj*zk,xi*R),0.0)
                t2=np.where(np.abs(yj*R)>1e-30,np.arctan2(xi*zk,yj*R),0.0)
                t3=np.where(np.abs(zk*R)>1e-30,np.arctan2(xi*yj,zk*R),0.0)
                lz=np.where(R+yj>1e-30,np.log(np.abs(R+yj)),0.0)
                ly=np.where(R+zk>1e-30,np.log(np.abs(R+zk)),0.0)
                lx=np.where(R+xi>1e-30,np.log(np.abs(R+xi)),0.0)
                total+=mu*(ax*t1+ay*t2+az*t3-bxy*lz-bxz*ly-byz*lx)
    return total

def _magnetic_field_single_prism(x1,x2,y1,y2,z1,z2,Mx,My,Mz,Fx,Fy,Fz):
    xs,ys,zs=[x1,x2],[y1,y2],[z1,z2];ax,ay,az=Mx*Fx,My*Fy,Mz*Fz
    bxy=My*Fx+Mx*Fy;bxz=Mz*Fx+Mx*Fz;byz=Mz*Fy+My*Fz;total=0.0
    for i in range(2):
        for j in range(2):
            for k in range(2):
                mu=(-1.0)**(i+j+k);xi,yj,zk=xs[i],ys[j],zs[k];R=np.sqrt(xi**2+yj**2+zk**2)
                if R<1e-15: continue
                def sl(v): return 0.0 if abs(v)<1e-30 else np.log(abs(v))
                total+=mu*(ax*np.arctan2(yj*zk,xi*R)+ay*np.arctan2(xi*zk,yj*R)+az*np.arctan2(xi*yj,zk*R)-bxy*sl(R+yj)-bxz*sl(R+zk)-byz*sl(R+xi))
    return total

_magnetic_field_array_vectorized=_magnetic_field_array
_magnetic_field_array=_magnetic_field_array
