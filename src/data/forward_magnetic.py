"""
3D 磁法正演计算模块
===================

基于总场异常 (Total Field Anomaly, ΔT) 公式，实现 3D 磁化率模型的磁法正演。

物理原理:
    在感应磁化假设下，磁性体被地磁场磁化产生的异常磁场。
    总场异常 ΔT 是异常磁场在地磁场方向上的投影。

核心公式（感应磁化模型）:
    ΔT = κ * F * K_m

    其中 K_m 为磁法核函数（重力位二阶导数张量的方向缩并），
    考虑了地磁场方向 (I, D) 和几何衰减。

单位制:
    - 输入: 磁化率 SI, 地磁场 F(nT)
    - 输出: 总场异常 ΔT(nT)

参考文献:
    1. Blakely, R.J., 1995. Potential Theory in Gravity and Magnetic Applications.
    2. Li & Oldenburg, 1996. 3-D inversion of magnetic data.
"""

import numpy as np


def _magnetic_kernel_on_points(x0, y0, z0, x1, x2, y1, y2, z1, z2,
                               alpha1, alpha2, alpha3):
    """
    计算单个棱柱体在一组观测点上的磁法核函数值。

    磁法核函数 = 重力位二阶导数张量的方向缩并:
        K_m = l²·V_xx + m²·V_yy + n²·V_zz +
              2lm·V_xy + 2ln·V_xz + 2mn·V_yz

    Parameters
    ----------
    x0, y0 : ndarray, shape (N,)
        观测点 x, y 坐标（1D）。
    z0 : float
        观测面高度（标量）。
    x1..z2 : float
        棱柱体范围 (m)。
    alpha1, alpha2, alpha3 : float
        地磁场方向余弦 (l, m, n)。

    Returns
    -------
    kernel : ndarray, shape (N,)
        磁法核函数值。
    """
    # 确保 1D
    x0 = np.asarray(x0, dtype=np.float64).ravel()
    y0 = np.asarray(y0, dtype=np.float64).ravel()

    # 8 个角点坐标差值 — 形状均为 (2, 2, 2, N)
    xi = np.array([x1, x2], dtype=np.float64)
    yj = np.array([y1, y2], dtype=np.float64)
    zk = np.array([z1, z2], dtype=np.float64)

    dx = xi[:, np.newaxis, np.newaxis, np.newaxis] - x0   # (2, 1, 1, N)
    dy = yj[np.newaxis, :, np.newaxis, np.newaxis] - y0   # (1, 2, 1, N)
    dz = zk[np.newaxis, np.newaxis, :, np.newaxis] - z0   # (1, 1, 2, 1)

    r = np.sqrt(dx**2 + dy**2 + dz**2 + 1e-30)

    # 符号因子
    signs = np.array([
        [[+1, -1],
         [-1, +1]],
        [[-1, +1],
         [+1, -1]]
    ], dtype=np.float64)
    signs_expanded = signs[..., np.newaxis]  # (2, 2, 2, 1)

    # === 6 个独立张量分量（Blakely 1995, Eq. 12.10a-f）===
    with np.errstate(divide='ignore', invalid='ignore'):
        Vxx_raw = -np.arctan2(dy * dz, dx * r + 1e-30)
        Vyy_raw = -np.arctan2(dx * dz, dy * r + 1e-30)
        Vzz_raw = -np.arctan2(dx * dy, dz * r + 1e-30)

    Vxy_raw = np.log(np.abs(r - dz) + 1e-30)
    Vxz_raw = np.log(np.abs(dy) + r)
    Vyz_raw = np.log(np.abs(dx) + r)

    # 对 8 角点求和 -> (N,)
    Vxx = np.sum(signs_expanded * Vxx_raw, axis=(0, 1, 2))
    Vyy = np.sum(signs_expanded * Vyy_raw, axis=(0, 1, 2))
    Vzz = np.sum(signs_expanded * Vzz_raw, axis=(0, 1, 2))
    Vxy = np.sum(signs_expanded * Vxy_raw, axis=(0, 1, 2))
    Vxz = np.sum(signs_expanded * Vxz_raw, axis=(0, 1, 2))
    Vyz = np.sum(signs_expanded * Vyz_raw, axis=(0, 1, 2))

    # 缩并: K_m = a^T · V · a
    l, m, n_val = alpha1, alpha2, alpha3
    kernel = (l**2 * Vxx + m**2 * Vyy + n_val**2 * Vzz +
              2*l*m * Vxy + 2*l*n_val * Vxz + 2*m*n_val * Vyz)

    return kernel


def forward_magnetic(kappa_model, grid_params, mag_params=None):
    """
    3D 磁法正演：从磁化率模型计算地表总场异常 ΔT。

    采用感应磁化假设：磁性体的磁化强度由地磁场感应产生，
    磁化方向与地磁场方向相同。

    Parameters
    ----------
    kappa_model : ndarray, shape (nx, ny, nz)
        3D 磁化率模型，SI 单位。
    grid_params : dict
        网格参数:
        - dx, dy, dz : float — 网格间距 (m)
        - x0, y0 : float — 坐标原点 (m)，默认 0
        - z0 : float — 观测面高度 (m)，默认 0
    mag_params : dict or None
        地磁参数:
        - I : float — 磁倾角 (度)，默认 45°
        - D : float — 磁偏角 (度)，默认 0°
        - F : float — 地磁场总强度 (nT)，默认 50000

    Returns
    -------
    magnetic_anomaly : ndarray, shape (nx, ny)
        总场异常 ΔT (nT)。
    """
    # ===== 默认地磁参数（Gap 7: 中国中纬度典型值）=====
    default_mag = {'I': 45.0, 'D': 0.0, 'F': 50000.0}
    if mag_params is None:
        mag_params = default_mag
    else:
        for k, v in default_mag.items():
            mag_params.setdefault(k, v)

    I_deg = mag_params['I']
    D_deg = mag_params['D']
    F = mag_params['F']

    # 方向余弦
    I_rad = np.deg2rad(I_deg)
    D_rad = np.deg2rad(D_deg)
    alpha1 = np.cos(I_rad) * np.cos(D_rad)   # Easting 分量
    alpha2 = np.cos(I_rad) * np.sin(D_rad)   # Northing 分量
    alpha3 = np.sin(I_rad)                   # 向下分量

    # ===== 网格参数 =====
    dx = grid_params['dx']
    dy = grid_params['dy']
    dz = grid_params['dz']
    z0_obs = grid_params.get('z0', 0.0)

    nx, ny, nz = kappa_model.shape
    x_origin = grid_params.get('x0', 0.0)
    y_origin = grid_params.get('y0', 0.0)

    # 观测面坐标（展平为 1D）
    obs_x = x_origin + (np.arange(nx) + 0.5) * dx
    obs_y = y_origin + (np.arange(ny) + 0.5) * dy
    OX_flat, OY_flat = np.meshgrid(obs_x, obs_y, indexing='ij')
    OX_flat = OX_flat.ravel()  # (nx*ny,)
    OY_flat = OY_flat.ravel()

    magnetic_anomaly = np.zeros((nx, ny), dtype=np.float64)

    # ===== 按深度层循环 =====
    for k in range(nz):
        z1 = k * dz
        z2 = (k + 1) * dz

        kappa_slice = kappa_model[:, :, k]

        nonzero_mask = np.abs(kappa_slice) > 1e-10
        if not np.any(nonzero_mask):
            continue

        ii, jj = np.where(nonzero_mask)
        kappa_vals = kappa_slice[ii, jj]

        # 对每个非零单元计算
        for idx in range(len(ii)):
            i_cell, j_cell = ii[idx], jj[idx]
            kappa_val = kappa_vals[idx]

            x1_c = x_origin + i_cell * dx
            x2_c = x1_c + dx
            y1_c = y_origin + j_cell * dy
            y2_c = y1_c + dy

            kernel = _magnetic_kernel_on_points(
                OX_flat, OY_flat, z0_obs,
                x1_c, x2_c, y1_c, y2_c, z1, z2,
                alpha1, alpha2, alpha3
            )  # (nx*ny,)

            magnetic_anomaly += (kappa_val * F * kernel).reshape(nx, ny)

    return magnetic_anomaly


def analytical_prism_magnetic(x0, y0, z0, x1, x2, y1, y2, z1, z2,
                              kappa, I_deg=45.0, D_deg=0.0, F=50000.0):
    """
    单个棱柱体磁法异常解析解（直接调用核函数）。

    Parameters
    ----------
    x0, y0 : array_like
        观测点坐标 (m)。
    z0 : float
        观测面高度 (m)。
    x1..z2 : float
        棱柱体范围 (m)。
    kappa : float
        磁化率 (SI)。
    I_deg, D_deg : float
        磁倾角、偏角 (度)。
    F : float
        地磁场强度 (nT)。

    Returns
    -------
    delta_T : ndarray
        总场异常 (nT)。
    """
    x0_arr = np.asarray(x0, dtype=np.float64)
    y0_arr = np.asarray(y0, dtype=np.float64)
    orig_shape = x0_arr.shape

    I_rad = np.deg2rad(I_deg)
    D_rad = np.deg2rad(D_deg)
    a1 = np.cos(I_rad) * np.cos(D_rad)
    a2 = np.cos(I_rad) * np.sin(D_rad)
    a3 = np.sin(I_rad)

    kernel = _magnetic_kernel_on_points(
        x0_arr.ravel(), y0_arr.ravel(), z0,
        x1, x2, y1, y2, z1, z2, a1, a2, a3
    )
    return (kappa * F * kernel).reshape(orig_shape)
