"""
3D 重力正演计算模块
===================

基于 Blakely (1995) 棱柱体重力公式，实现 3D 密度模型的垂直重力异常正演。

物理原理:
    对于地下每个棱柱体单元 (prism cell)，其在地表观测点产生的垂直重力异常由
    牛顿万有引力定律的积分形式给出。Blakely (1995) 给出了长方体棱柱体的解析解，
    即对 8 个角点求和的形式。

核心公式:
    g_z = G * rho * V * kernel

    其中核函数 kernel 为棱柱体 8 个角点的加权和:
    kernel = sum_{i,j,k in {1,2}} (-1)^(i+j+k) *
             [ x*ln(y+r) + y*ln(x+r) - z*atan(xy/(z*r)) ]
    r = sqrt(x^2 + y^2 + z^2)

单位制说明:
    - 输入密度: g/cm^3 (CGS)
    - 长度单位: m (SI)
    - G = 6.674e-3: CGS 单位制下的引力常数，使得输出直接为 mGal
      (1 mGal = 10^-5 m/s^2, CGS 中 G = 6.674e-8 cm^3/(g*s^2),
       转换后 G_eff = 6.674e-3 mGal * cm^3 / (g * m^2))

参考文献:
    Blakely, R.J., 1995. Potential Theory in Gravity and Magnetic Applications.
    Cambridge University Press.
"""

import numpy as np


def _gravity_kernel_on_points(x0, y0, z0, x1, x2, y1, y2, z1, z2):
    """
    计算单个棱柱体在一组观测点上的重力核函数值（无密度因子）。

    这是 Blakely (1995) 公式的纯几何部分。观测点坐标为 1D 数组，
    内部通过 numpy 广播对所有观测点同时计算。

    Parameters
    ----------
    x0, y0 : ndarray, shape (N,)
        观测点 x, y 坐标数组（1D）。
    z0 : float
        观测面高度（标量）。
    x1, x2, y1, y2, z1, z2 : float
        棱柱体范围 (m)。

    Returns
    -------
    kernel : ndarray, shape (N,)
        核函数值，量纲为 [长度]。乘以 G * rho 后得到 mGal。
    """
    # 确保是 1D float64 数组
    x0 = np.asarray(x0, dtype=np.float64).ravel()
    y0 = np.asarray(y0, dtype=np.float64).ravel()

    # 8 个角点的坐标差值
    # xi: (2, 1), yj: (1, 2), zk: (1, 1, 2) 与 x0(N,), y0(N,) 广播
    xi = np.array([x1, x2], dtype=np.float64)  # (2,)
    yj = np.array([y1, y2], dtype=np.float64)  # (2,)
    zk = np.array([z1, z2], dtype=np.float64)  # (2,)

    # 距离分量 — 形状均为 (2, 2, 2, N)
    dx = xi[:, np.newaxis, np.newaxis, np.newaxis] - x0   # (2, 1, 1) - (N,) -> (2, 1, 1, N)
    dy = yj[np.newaxis, :, np.newaxis, np.newaxis] - y0   # (1, 2, 1) - (N,) -> (1, 2, 1, N)
    dz = zk[np.newaxis, np.newaxis, :, np.newaxis] - z0   # (1, 1, 2) - () -> (1, 1, 2, 1)

    r = np.sqrt(dx**2 + dy**2 + dz**2 + 1e-30)

    # 符号因子: (-1)^(i+j+k)
    signs = np.array([
        [[+1, -1],
         [-1, +1]],
        [[-1, +1],
         [+1, -1]]
    ], dtype=np.float64)

    signs_expanded = signs[..., np.newaxis]  # (2, 2, 2, 1)

    # 核函数三项
    term1 = dx * np.log(np.abs(dy) + r)
    term2 = dy * np.log(np.abs(dx) + r)

    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = (dx * dy) / (dz * r + 1e-30)
    term3 = -dz * np.arctan(ratio)

    # 对 8 个角点求和 -> (N,)
    kernel = np.sum(signs_expanded * (term1 + term2 + term3), axis=(0, 1, 2))

    return kernel


def forward_gravity(rho_model, grid_params):
    """
    3D 重力正演：从密度模型计算地表垂直重力异常。

    基于 Blakely (1995) 棱柱体公式，将地下 3D 密度模型离散为均匀棱柱体网格，
    对每个棱柱体计算其在所有观测点产生的重力异常，最后求和。

    Parameters
    ----------
    rho_model : ndarray, shape (nx, ny, nz)
        3D 密度模型，单位 g/cm^3 (CGS)。
        nx, ny 为水平方向网格数，nz 为深度层数。
    grid_params : dict
        网格参数字典:
        - dx : float — x 方向网格间距 (m)
        - dy : float — y 方向网格间距 (m)
        - dz : float — z 方向(深度)网格间距 (m)
        - x0 : float or None — 观测面 x 坐标原点 (m)。默认 0
        - y0 : float or None — 观测面 y 坐标原点 (m)。默认 0
        - z0 : float — 观测面高度 (m)。正值表示观测面在模型顶部之上

    Returns
    -------
    gravity_anomaly : ndarray, shape (nx_obs, ny_obs)
        垂直重力异常，单位 mGal。
        默认观测点位于每个水平网格柱的正上方。

    Notes
    -----
    - 实现策略: 对每个非零单元调用一次核函数（内部对所有观测点向量化）。
    - 内存安全: 避免了 (nx*ny*n_cells) 的巨型中间数组。
    - 坐标系: x(easting), y(northing), z(深度向下为正)。
    """
    # ===== 参数提取 =====
    dx = grid_params['dx']
    dy = grid_params['dy']
    dz = grid_params['dz']
    z0_obs = grid_params.get('z0', 0.0)

    nx, ny, nz = rho_model.shape
    x_origin = grid_params.get('x0', 0.0)
    y_origin = grid_params.get('y0', 0.0)

    # 引力常数 (CGS -> mGal 转换)
    G = 6.674e-3

    # ===== 观测面坐标（1D 展平）=====
    obs_x = x_origin + (np.arange(nx) + 0.5) * dx  # (nx,)
    obs_y = y_origin + (np.arange(ny) + 0.5) * dy  # (ny,)

    # 构造完整的 2D 观测点网格（展平为 1D）
    OX_flat, OY_flat = np.meshgrid(obs_x, obs_y, indexing='ij')
    OX_flat = OX_flat.ravel()  # (nx*ny,)
    OY_flat = OY_flat.ravel()  # (nx*ny,)

    # ===== 初始化输出 =====
    gravity_anomaly = np.zeros((nx, ny), dtype=np.float64)

    # ===== 按深度层循环，每层内按非零单元循环 =====
    for k in range(nz):
        z1 = k * dz           # 棱柱体顶面深度
        z2 = (k + 1) * dz     # 棱柱体底面深度

        rho_slice = rho_model[:, :, k]

        # 非零密度单元优化
        nonzero_mask = np.abs(rho_slice) > 1e-10
        if not np.any(nonzero_mask):
            continue

        ii, jj = np.where(nonzero_mask)
        rho_vals = rho_slice[ii, jj]

        # 对每个非零单元: 核函数对所有观测点向量化计算
        for idx in range(len(ii)):
            i_cell, j_cell = ii[idx], jj[idx]
            rho_val = rho_vals[idx]

            x1_c = x_origin + i_cell * dx
            x2_c = x1_c + dx
            y1_c = y_origin + j_cell * dy
            y2_c = y1_c + dy

            # 一次调用计算该单元对所有观测点的贡献 (1D 向量)
            kernel = _gravity_kernel_on_points(
                OX_flat, OY_flat, z0_obs,
                x1_c, x2_c, y1_c, y2_c, z1, z2
            )  # shape (nx*ny,)

            # 累加到输出网格
            gravity_anomaly += (G * rho_val * kernel).reshape(nx, ny)

    return gravity_anomaly


def forward_gravity_dense(rho_model, grid_params):
    """
    3D 重力正演稠密版本 — 接口兼容，实现与 forward_gravity 一致。

    Parameters
    ----------
    rho_model : ndarray, shape (nx, ny, nz)
        3D 密度模型，单位 g/cm³。
    grid_params : dict
        同 forward_gravity。

    Returns
    -------
    gravity_anomaly : ndarray, shape (nx, ny)
        垂直重力异常，单位 mGal。
    """
    return forward_gravity(rho_model, grid_params)


# ===== 解析解验证函数 =====

def analytical_sphere_gravity(x0, y0, z0, sphere_center, sphere_radius, density):
    """
    均匀球体的重力异常解析解（质点近似）。

    Parameters
    ----------
    x0, y0 : array_like
        观测点坐标 (m)。
    z0 : float
        观测面高度 (m)。
    sphere_center : tuple (cx, cy, cz)
        球心坐标 (cz 为深度向下为正, m)。
    sphere_radius : float
        球体半径 (m)。
    density : float
        密度差 (g/cm³)。

    Returns
    -------
    gz : ndarray
        重力异常 (mGal)。
    """
    cx, cy, cz = sphere_center
    x0 = np.asarray(x0, dtype=np.float64)
    y0 = np.asarray(y0, dtype=np.float64)

    R_cm = sphere_radius * 100.0
    volume_cm3 = (4.0 / 3.0) * np.pi * R_cm**3
    mass_g = density * volume_cm3

    rx = (x0 - cx) * 100.0
    ry = (y0 - cy) * 100.0
    rz = (z0 + cz) * 100.0
    r = np.sqrt(rx**2 + ry**2 + rz**2)

    G_cgs = 6.674e-8
    gz_gal = G_cgs * mass_g * rz / (r**3 + 1e-30)
    return gz_gal * 1000.0


def analytical_prism_gravity(x0, y0, z0, x1, x2, y1, y2, z1, z2, density):
    """
    单个棱柱体重力异常解析解（直接调用核函数）。

    Parameters
    ----------
    x0, y0 : array_like
        观测点坐标 (m)，支持任意形状（将 ravel 后计算再 reshape 回原形状）。
    z0 : float
        观测面高度 (m)。
    x1..z2 : float
        棱柱体范围 (m)。
    density : float
        密度 (g/cm³)。

    Returns
    -------
    gz : ndarray
        重力异常 (mGal)，与 x0 同形状。
    """
    x0_arr = np.asarray(x0, dtype=np.float64)
    y0_arr = np.asarray(y0, dtype=np.float64)
    orig_shape = x0_arr.shape

    G = 6.674e-3
    kernel = _gravity_kernel_on_points(
        x0_arr.ravel(), y0_arr.ravel(), z0,
        x1, x2, y1, y2, z1, z2
    )
    return (G * density * kernel).reshape(orig_shape)
