"""
合成数据集生成器
================
生成论文 Table I 规定的全部 6 类数据集，用于 3D 重磁联合反演深度学习训练。

三种基础地质体模型:
  (a) 单长方体 (single_cuboid)
  (b) 单倾斜体 (single_tilting)
  (c) 单随机游走体 (random_walk_body)

内置简化正演: 将 3D 模型沿 Z 轴深度加权投影得到 2D 异常观测数据。
(如果后续有严格物理正演模块 forward_gravity.py / forward_magnetic.py，
 可替换 simple_forward_* 函数。)

作者: Agent-DataEngineering
日期: 2026-04-21
"""

import numpy as np
import os
import json
import time
from typing import List, Dict, Tuple, Optional

# ============================================================
# 全局常量（来自论文 + Gap 填充）
# ============================================================

# 地磁参数 (Gap 7: 中国地区典型中纬度值)
DEFAULT_INCLINATION = 45.0   # 磁倾角 I (度)
DEFAULT_DECLINATION = 0.0    # 磁偏角 D (度)
DEFAULT_FIELD_INTENSITY = 50000.0  # 地磁场总强度 F (nT)

# 默认网格尺寸
DEFAULT_NX = 40
DEFAULT_NY = 40
DEFAULT_NZ = 20

# 默认噪声水平 (论文 Table I, 归一化后)
DEFAULT_NOISE_GRAVITY = 0.005
DEFAULT_NOISE_MAGNETIC = 0.108

# ============================================================
# 数据集规格表 (Table I 精确数值)
# ============================================================
DATASET_SPECS = {
    1: {
        'global':      {'model_type': 'cuboid',       'n': 6000},
        'partial':     {'model_type': 'combined_cuboid', 'n': 3400},
        'inconsistent':{'model_type': 'combined_randomwalk', 'n': 3400},
        'total': 12800,
    },
    2: {
        'global':      {'model_type': 'tilting',       'n': 3200},
        'partial':     {'model_type': 'combined_tilting', 'n': 3400},
        'inconsistent':{'model_type': 'combined_randomwalk', 'n': 3400},
        'total': 10000,
    },
    3: {
        'global':      {'model_type': 'randomwalk',    'n': 1600},
        'partial':     {'model_type': 'combined_randomwalk', 'n': 1700},
        'inconsistent':{'model_type': 'combined_multi_rw',   'n': 1700},
        'total': 5000,
    },
    4: {
        'global':      {'model_type': 'cuboid',       'n': 800},
        'partial':     {'model_type': 'combined_cuboid', 'n': 850},
        'inconsistent':{'model_type': 'combined_randomwalk', 'n': 850},
        'total': 2500,
    },
    5: {
        'global':      {'model_type': 'randomwalk',    'n': 1600},
        'partial':     {'model_type': 'combined_randomwalk', 'n': 1700},
        'inconsistent':{'model_type': 'combined_multi_rw',   'n': 1700},
        'total': 5000,
    },
    6: {
        'global':      {'model_type': 'cuboid',       'n': 1500},
        'partial':     {'model_type': 'combined_cuboid', 'n': 1400},
        'inconsistent':{'model_type': 'combined_mixed',      'n': 2100},
        'total': 5000,
    },
}

TOTAL_SAMPLES = sum(spec['total'] for spec in DATASET_SPECS.values())  # 45300


# ============================================================
# 工具函数: 随机种子、归一化、加噪
# ============================================================

def set_seed(seed: int = 42):
    """固定随机种子保证可复现性"""
    np.random.seed(seed)


def normalize_data(data: np.ndarray) -> np.ndarray:
    """
    Min-Max 归一化到 [0,1] (Gap 9)

    参数:
        data: 输入数组 (任意形状)

    返回:
        归一化后的数组, shape 与输入相同
    """
    dmin = data.min()
    dmax = data.max()
    if dmax - dmin < 1e-10:
        return np.zeros_like(data)
    return (data - dmin) / (dmax - dmin)


def add_noise(data: np.ndarray, noise_level: float, seed: Optional[int] = None) -> np.ndarray:
    """
    添加高斯噪声。

    使用 noise_level/3 作为标准差，使得 ~99.7% 的噪声落在 [-noise_level, +noise_level] 范围内 (3-sigma)。
    最终 clip 到 [0, 1]。

    参数:
        data: 已归一化到 [0,1] 的数据
        noise_level: 最大噪声幅度 (如 0.005 或 0.108)
        seed: 噪声随机种子 (None 则使用全局状态)

    返回:
        加噪后的数据, shape 与输入相同
    """
    if seed is not None:
        rng_state = np.random.get_state()
        np.random.seed(seed)
    noise = np.random.normal(0, noise_level / 3.0, data.shape)
    if seed is not None:
        np.random.set_state(rng_state)
    return np.clip(data + noise, 0.0, 1.0)


# ============================================================
# 三种基础地质体模型生成函数
# ============================================================

def generate_cuboid(nx: int, ny: int, nz: int,
                    density_range: Tuple[float, float] = (0.5, 1.5),
                    susc_range: Tuple[float, float] = (0.05, 0.3),
                    rng: Optional[np.random.RandomState] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    (a) 单长方体 — 随机生成长方体地质体。

    在 3D 网格中放置一个随机位置、大小的长方体，赋予均匀密度和磁化率。

    参数:
        nx, ny, nz: 网格尺寸
        density_range: 密度取值范围 (g/cm^3)
        susc_range: 磁化率取值范围 (SI)
        rng: RandomState 实例 (None 则用全局)

    返回:
        rho_model: (nx, ny, nz) 密度模型
        kappa_model: (nx, ny, nz) 磁化率模型
    """
    if rng is None:
        rng = np.random.RandomState()

    rho_model = np.zeros((nx, ny, nz), dtype=np.float32)
    kappa_model = np.zeros((nx, ny, nz), dtype=np.float32)

    # 随机长方体参数
    # 大小: 占网格的 15%~50%
    size_x = rng.randint(max(3, nx // 6), max(4, nx // 2))
    size_y = rng.randint(max(3, ny // 6), max(4, ny // 2))
    size_z = rng.randint(max(2, nz // 4), max(3, nz // 2))

    # 位置: 保证完全在网格内
    x_start = rng.randint(0, nx - size_x)
    y_start = rng.randint(0, ny - size_y)
    z_start = rng.randint(0, nz - size_z)

    # 物性参数
    density = rng.uniform(*density_range)
    susceptibility = rng.uniform(*susc_range)

    # 填充长方体
    rho_model[x_start:x_start+size_x,
              y_start:y_start+size_y,
              z_start:z_start+size_z] = density
    kappa_model[x_start:x_start+size_x,
                y_start:y_start+size_y,
                z_start:z_start+size_z] = susceptibility

    return rho_model, kappa_model


def generate_tilting_body(nx: int, ny: int, nz: int,
                          density_range: Tuple[float, float] = (0.5, 1.5),
                          susc_range: Tuple[float, float] = (0.05, 0.3),
                          tilt_angle_range: Tuple[float, float] = (10.0, 45.0),
                          rng: Optional[np.random.RandomState] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    (b) 单倾斜体 — 通过坐标旋转实现倾斜效果。

    先在局部坐标系生成长方体，再绕 Y 轴旋转一定角度映射到网格坐标。
    使用超采样抗锯齿以减少旋转产生的锯齿边缘。

    参数:
        nx, ny, nz: 网格尺寸
        density_range: 密度取值范围
        susc_range: 磁化率取值范围
        tilt_angle_range: 倾斜角度范围 (度)，绕水平轴
        rng: RandomState 实例

    返回:
        rho_model, kappa_model
    """
    if rng is None:
        rng = np.random.RandomState()

    rho_model = np.zeros((nx, ny, nz), dtype=np.float32)
    kappa_model = np.zeros((nx, ny, nz), dtype=np.float32)

    # 局部长方体参数 (略小于网格以确保旋转后不越界)
    local_size_x = rng.randint(max(3, nx // 7), max(4, nx // 3))
    local_size_y = rng.randint(max(3, ny // 6), max(4, ny // 2))
    local_size_z = rng.randint(max(2, nz // 5), max(3, nz // 3))

    # 物性
    density = rng.uniform(*density_range)
    susceptibility = rng.uniform(*susc_range)

    # 倾斜角度 (绕 X-Z 平面内的轴旋转，模拟地质体倾斜)
    angle_deg = rng.uniform(*tilt_angle_range)
    angle_rad = np.radians(angle_deg)

    # 局部坐标中心 (放在网格中心附近)
    cx_local = local_size_x / 2.0
    cy_local = local_size_y / 2.0
    cz_local = local_size_z / 2.0

    # 全局中心偏移
    offset_x = rng.uniform(nx * 0.25, nx * 0.75)
    offset_y = rng.uniform(ny * 0.25, ny * 0.75)
    offset_z = rng.uniform(nz * 0.2, nz * 0.8)

    # 超采样因子 (抗锯齿)
    super_sample = 3

    # 构建局部坐标系中的长方体点云
    lx_pts = np.linspace(0, local_size_x, local_size_x * super_sample)
    ly_pts = np.linspace(0, local_size_y, local_size_y * super_sample)
    lz_pts = np.linspace(0, local_size_z, local_size_z * super_sample)

    # 生成网格点
    llx, lly, llz = np.meshgrid(lx_pts, ly_pts, lz_pts, indexing='ij')
    llx_flat = llx.ravel() - cx_local
    lly_flat = lly.ravel() - cy_local
    llz_flat = llz.ravel() - cz_local

    # 绕 Y 轴旋转 (X-Z 平面内倾斜)
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)
    rx_flat = llx_flat * cos_a - llz_flat * sin_a
    ry_flat = lly_flat.copy()
    rz_flat = llx_flat * sin_a + llz_flat * cos_a

    # 映射到全局坐标
    gx_flat = rx_flat + offset_x
    gy_flat = ry_flat + offset_y
    gz_flat = rz_flat + offset_z

    # 过滤出在网格范围内的点
    valid = (gx_flat >= 0) & (gx_flat < nx) & \
            (gy_flat >= 0) & (gy_flat < ny) & \
            (gz_flat >= 0) & (gz_flat < nz)

    gx_valid = gx_flat[valid].astype(int)
    gy_valid = gy_flat[valid].astype(int)
    gz_valid = gz_flat[valid].astype(int)

    # 使用累加实现抗锯齿 (超采样点贡献权重 = 1/super_sample^3)
    weight = 1.0 / (super_sample ** 3)
    np.add.at(rho_model, (gx_valid, gy_valid, gz_valid), density * weight)
    np.add.at(kappa_model, (gx_valid, gy_valid, gz_valid), susceptibility * weight)

    return rho_model, kappa_model


def generate_random_walk_body(nx: int, ny: int, nz: int,
                              density_range: Tuple[float, float] = (0.5, 1.5),
                              susc_range: Tuple[float, float] = (0.05, 0.3),
                              n_steps: int = 20,
                              rng: Optional[np.random.RandomState] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    (c) 单随机游走体 — 从种子点出发随机游走扩展形成不规则体。

    算法:
      1. 在网格中心区域选一个种子点
      2. 从种子出发进行 n_steps 步 3D 随机游走 (6邻域或26邻域)
      3. 游走经过的所有体素构成不规则地质体
      4. 对游走路径做形态学膨胀使体积更饱满

    参数:
        nx, ny, nz: 网格尺寸
        density_range: 密度取值范围
        susc_range: 磁化率取值范围
        n_steps: 随机游走步数 (控制体大小)
        rng: RandomState 实例

    返回:
        rho_model, kappa_model
    """
    if rng is None:
        rng = np.random.RandomState()

    rho_model = np.zeros((nx, ny, nz), dtype=np.float32)
    kappa_model = np.zeros((nx, ny, nz), dtype=np.float32)

    # 种子点 (避开边界)
    margin_x = nx // 5
    margin_y = ny // 5
    margin_z = nz // 5
    seed_x = rng.randint(margin_x, nx - margin_x)
    seed_y = rng.randint(margin_y, ny - margin_y)
    seed_z = rng.randint(margin_z, nz - margin_z)

    # 物性
    density = rng.uniform(*density_range)
    susceptibility = rng.uniform(*susc_range)

    # 26-邻域方向向量 (3D 中所有可能的单步移动)
    directions_26 = []
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            for dz in [-1, 0, 1]:
                if dx == 0 and dy == 0 and dz == 0:
                    continue
                directions_26.append((dx, dy, dz))
    directions_26 = np.array(directions_26, dtype=int)

    # 随机游走
    current = np.array([seed_x, seed_y, seed_z], dtype=int)
    visited = set()
    visited.add((current[0], current[1], current[2]))

    # 多次游走以增加体积 (每次从种子重新开始，不同方向)
    n_walks = max(3, n_steps // 5)
    steps_per_walk = max(5, n_steps // n_walks)

    for _ in range(n_walks):
        pos = np.array([seed_x, seed_y, seed_z], dtype=int)
        for _ in range(steps_per_walk):
            # 随机选择一个方向
            direction = directions_26[rng.randint(len(directions_26))]
            new_pos = pos + direction

            # 边界检查
            if (0 <= new_pos[0] < nx and
                0 <= new_pos[1] < ny and
                0 <= new_pos[2] < nz):
                pos = new_pos
                visited.add((pos[0], pos[1], pos[2]))

    # 将访问过的体素标记为地质体
    for (vx, vy, vz) in visited:
        rho_model[vx, vy, vz] = density
        kappa_model[vx, vy, vz] = susceptibility

    # 形态学膨胀 (使不规则体更饱满，避免过于稀疏)
    from scipy.ndimage import binary_dilation
    mask = rho_model > 0
    struct = np.ones((3, 3, 2), dtype=bool)  # XY 方向膨胀3格，Z方向膨胀2格
    dilated_mask = binary_dilation(mask, structure=struct)

    # 仅对膨胀区域内原为零的地方填充 (使用稍低的物性避免过度增强)
    new_fill = dilated_mask & (~mask)
    if np.any(new_fill):
        fill_density = density * rng.uniform(0.6, 0.9)
        fill_susc = susceptibility * rng.uniform(0.6, 0.9)
        rho_model[new_fill] = fill_density
        kappa_model[new_fill] = fill_susc

    return rho_model, kappa_model


# ============================================================
# 组合模型函数
# ============================================================

def combine_models(models_list: List[Tuple[np.ndarray, np.ndarray]],
                   allow_intersect: bool = True,
                   intersect_mode: str = 'max',
                   rng: Optional[np.random.RandomState] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    组合多个地质体模型。

    参数:
        models_list: [(rho_1, kappa_1), (rho_2, kappa_2), ...]
        allow_intersect: 是否允许相交 (True=允许接触和相交)
        intersect_mode: 相交处处理方式
            'max'   — 取较大值 (默认)
            'sum'   — 求和叠加
            'avg'   — 取平均
        rng: RandomState 实例

    返回:
        combined_rho, combined_kappa
    """
    if len(models_list) == 0:
        raise ValueError("models_list 不能为空")

    if len(models_list) == 1:
        return models_list[0][0].copy(), models_list[0][1].copy()

    if rng is None:
        rng = np.random.RandomState()

    # 初始化为第一个模型
    shapes = [m[0].shape for m in models_list]
    assert all(s == shapes[0] for s in shapes), "所有模型形状必须一致"
    nx, ny, nz = shapes[0]

    combined_rho = np.zeros((nx, ny, nz), dtype=np.float32)
    combined_kappa = np.zeros((nx, ny, nz), dtype=np.float32)

    if allow_intersect:
        if intersect_mode == 'max':
            for rho, kappa in models_list:
                combined_rho = np.maximum(combined_rho, rho)
                combined_kappa = np.maximum(combined_kappa, kappa)
        elif intersect_mode == 'sum':
            for rho, kappa in models_list:
                combined_rho = combined_rho + rho
                combined_kappa = combined_kappa + kappa
        elif intersect_mode == 'avg':
            count = np.zeros((nx, ny, nz), dtype=np.float32)
            for rho, kappa in models_list:
                mask = rho > 0
                combined_rho[mask] += rho[mask]
                combined_kappa[mask] += kappa[mask]
                count[mask] += 1.0
            safe_count = np.maximum(count, 1.0)
            combined_rho /= safe_count
            combined_kappa /= safe_count
        else:
            raise ValueError(f"未知 intersect_mode: {intersect_mode}")
    else:
        # 不允许相交: 后面的模型覆盖前面的 (简单策略)
        for rho, kappa in models_list:
            mask = rho > 0
            combined_rho[mask] = rho[mask]
            combined_kappa[mask] = kappa[mask]

    return combined_rho, combined_kappa


# ============================================================
# 结构相似性标签生成
# ============================================================

def compute_structural_similarity(rho: np.ndarray, kappa: np.ndarray) -> np.ndarray:
    """
    计算结构相似性标签 S。

    S[i,j,k] = 1 if rho[i,j,k] != 0 AND kappa[i,j,k] != 0 else 0

    物理含义: 标记密度异常和磁异常同时存在的区域 (结构一致区)。

    参数:
        rho: (nx, ny, nz) 密度模型
        kappa: (nx, ny, nz) 磁化率模型

    返回:
        structural_sim: (nx, ny, nz) 二值标签, dtype=float32
    """
    sim = ((rho != 0) & (kappa != 0)).astype(np.float32)
    return sim


# ============================================================
# 内置正演备选方案 (当 forward_*.py 不存在时使用)
# ============================================================

def simple_forward_gravity(rho: np.ndarray, obs_height: float = 1.0) -> np.ndarray:
    """
    简化重力正演: 深度加权垂直积分。

    将 3D 密度模型沿 Z 轴投影，每层按深度平方反比加权，
    近似模拟重力异常随深度衰减的物理规律。

    这不是严格的 Blakely (1995) 棱柱体公式，但足以生成
    训练数据的基本空间模式 (高密度区对应高重力异常)。

    参数:
        rho: (nx, ny, nz) 密度模型
        obs_height: 观测面等效高度 (单位: 层间距倍数)

    返回:
        gravity: (nx, ny) 重力异常 (未归一化)
    """
    nz = rho.shape[2]

    # 深度权重: 越深影响越小 (平方反比律近似)
    depths = np.arange(nz, dtype=np.float64) * 10.0 + obs_height * 10.0  # 假设层间距 10m
    weights = 1.0 / (depths ** 2)
    weights = weights.astype(np.float32)
    weights = weights / weights.sum()  # 归一化权重

    # 沿 Z 轴加权求和
    gravity = np.sum(rho * weights[np.newaxis, np.newaxis, :], axis=2)

    return gravity.astype(np.float32)


def simple_forward_magnetic(kappa: np.ndarray,
                            obs_height: float = 1.0,
                            inclination: float = DEFAULT_INCLINATION,
                            declination: float = DEFAULT_DECLINATION) -> np.ndarray:
    """
    简化磁法正演: 深度加权垂直积分 + 方向投影。

    类似重力正演但加入磁倾角/偏角的方向余弦因子，
    使磁异常呈现偶极子特征 (正负伴生)。

    参数:
        kappa: (nx, ny, nz) 磁化率模型
        obs_height: 观测面等效高度
        inclination: 磁倾角 (度)
        declination: 磁偏角 (度)

    返回:
        magnetic: (nx, ny) 磁异常 (未归一化)
    """
    nz = kappa.shape[2]

    # 深度权重 (同重力)
    depths = np.arange(nz, dtype=np.float64) * 10.0 + obs_height * 10.0
    weights = 1.0 / (depths ** 2)
    weights = weights.astype(np.float32)
    weights = weights / weights.sum()

    # 基础投影 (同重力)
    base_proj = np.sum(kappa * weights[np.newaxis, np.newaxis, :], axis=2)

    # 方向余弦因子 (产生偶极子特征)
    inc_rad = np.radians(inclination)
    dec_rad = np.radians(declination)

    # 创建方向梯度模板 (简单的空间导数近似)
    # 磁异常在源体"南侧"通常为正、"北侧"为负 (北半球特征)
    nx_ny = kappa.shape[:2]
    cy, cx = nx_ny[0] / 2.0, nx_ny[1] / 2.0

    yy, xx = np.mgrid[0:nx_ny[0], 0:nx_ny[1]]
    # 相对于中心的归一化距离
    dy = (yy - cy) / max(cy, 1.0)
    dx = (xx - cx) / max(cx, 1.0)

    # 磁化方向投影因子
    # 总场异常 ≈ 基础投影 * (cos(I)^2 + 垂直梯度项)
    cos_inc = np.cos(inc_rad)
    sin_inc = np.sin(inc_rad)
    cos_dec = np.cos(dec_rad)
    sin_dec = np.sin(dec_rad)

    # 简化的偶极子修正: 加入水平梯度分量
    dipole_correction = (
        sin_inc * cos_dec * dx +   # 南北方向分量
        sin_inc * sin_dec * dy +   # 东西方向分量
        cos_inc                      # 垂直分量
    )

    magnetic = base_proj * dipole_correction

    return magnetic.astype(np.float32)


# ============================================================
# 尝试导入外部正演模块 (如果可用则优先使用)
# ============================================================

try:
    from src.data.forward_gravity import forward_gravity as _ext_fg
    from src.data.forward_magnetic import forward_magnetic as _ext_fm
    HAS_EXTERNAL_FORWARD = True
except ImportError:
    HAS_EXTERNAL_FORWARD = False
    _ext_fg = None
    _ext_fm = None


# 默认网格参数 (用于调用外部正演模块)
_DEFAULT_GRID_PARAMS = {
    'dx': 1.0,   # 网格间距 m (归一化坐标下)
    'dy': 1.0,
    'dz': 10.0,  # 深度层间距 m (论文 Section 2.1)
    'x0': 0.0,
    'y0': 0.0,
    'z0': 1.0,   # 观测面高度 m
}

_DEFAULT_MAG_PARAMS = {
    'I': DEFAULT_INCLINATION,
    'D': DEFAULT_DECLINATION,
    'F': DEFAULT_FIELD_INTENSITY,
}


def forward_gravity(rho: np.ndarray, obs_height: float = 1.0) -> np.ndarray:
    """
    重力正演入口: 优先使用外部物理模块，否则用内置简化版。

    外部模块接口: forward_gravity(rho_model, grid_params) -> (nx, ny)
    内置接口:     simple_forward_gravity(rho, obs_height) -> (nx, ny)

    注意: 如果外部模块调用失败 (如参数不兼容)，自动回退到内置简化版。
    """
    if HAS_EXTERNAL_FORWARD and _ext_fg is not None:
        try:
            gp = dict(_DEFAULT_GRID_PARAMS)
            gp['z0'] = obs_height
            result = _ext_fg(rho, gp)
            return result
        except Exception:
            # 外部模块调用失败，回退到内置版
            pass
    return simple_forward_gravity(rho, obs_height=obs_height)


def forward_magnetic(kappa: np.ndarray,
                     obs_height: float = 1.0,
                     inclination: float = DEFAULT_INCLINATION,
                     declination: float = DEFAULT_DECLINATION) -> np.ndarray:
    """
    磁法正演入口: 优先使用外部物理模块，否则用内置简化版。

    外部模块接口: forward_magnetic(kappa_model, grid_params, mag_params) -> (nx, ny)
    内置接口:     simple_forward_magnetic(kappa, obs_height, I, D) -> (nx, ny)

    注意: 如果外部模块调用失败，自动回退到内置简化版。
    """
    if HAS_EXTERNAL_FORWARD and _ext_fm is not None:
        try:
            gp = dict(_DEFAULT_GRID_PARAMS)
            gp['z0'] = obs_height
            mp = dict(_DEFAULT_MAG_PARAMS)
            mp['I'] = inclination
            mp['D'] = declination
            result = _ext_fm(kappa, gp, mp)
            return result
        except Exception:
            # 外部模块调用失败，回退到内置版
            pass
    return simple_forward_magnetic(kappa, obs_height=obs_height,
                                   inclination=inclination, declination=declination)


# ============================================================
# 一致性类型辅助函数
# ============================================================

def generate_consistency_type(consistency: str,
                               model_type: str,
                               nx: int, ny: int, nz: int,
                               density_range: Tuple[float, float],
                               susc_range: Tuple[float, float],
                               rng: Optional[np.random.RandomState] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    根据一致性类型和模型类型生成对应的 (rho, kappa) 模型对。

    一致性类型说明:
      - global (全局一致):      rho 和 kappa 来自同一地质体 → 结构完全一致
      - partial (部分一致):     rho 和 kappa 来自相同类型的组合体，但某些体只有其中一种属性
      - inconsistent (结构不一致): rho 和 kappa 来自不同类型的地质体 → 结构不一致

    参数:
        consistency: 'global' | 'partial' | 'inconsistent'
        model_type: 模型类型字符串
        nx, ny, nz: 网格尺寸
        density_range, susc_range: 物性范围
        rng: RandomState

    返回:
        rho, kappa
    """
    if rng is None:
        rng = np.random.RandomState()

    if consistency == 'global':
        # === 全局一致: 同一地质体同时有密度和磁化率 ===
        if model_type == 'cuboid':
            rho, kappa = generate_cuboid(nx, ny, nz, density_range, susc_range, rng=rng)
        elif model_type == 'tilting':
            rho, kappa = generate_tilting_body(nx, ny, nz, density_range, susc_range, rng=rng)
        elif model_type == 'randomwalk':
            rho, kappa = generate_random_walk_body(nx, ny, nz, density_range, susc_range, rng=rng)
        else:
            raise ValueError(f"未知 model_type: {model_type}")

    elif consistency == 'partial':
        # === 部分一致: 组合体，部分子体同时有两种属性，部分只有一种 ===
        n_bodies = rng.randint(2, 4)
        models = []
        for i in range(n_bodies):
            if model_type in ('combined_cuboid', 'combined_tilting'):
                base_type = 'cuboid' if 'cuboid' in model_type else 'tilting'
            else:
                base_type = 'randomwalk'

            if base_type == 'cuboid':
                r, k = generate_cuboid(nx, ny, nz, density_range, susc_range, rng=rng)
            elif base_type == 'tilting':
                r, k = generate_tilting_body(nx, ny, nz, density_range, susc_range, rng=rng)
            else:
                r, k = generate_random_walk_body(nx, ny, nz, density_range, susc_range, rng=rng)

            # 部分一致: 随机让某些子体只保留一种属性
            if i > 0 and rng.random() < 0.4:
                if rng.random() < 0.5:
                    k = np.zeros_like(k)  # 只保留密度
                else:
                    r = np.zeros_like(r)  # 只保留磁化率

            models.append((r, k))

        rho, kappa = combine_models(models, allow_intersect=True, intersect_mode='max', rng=rng)

    elif consistency == 'inconsistent':
        # === 结构不一致: 密度体和磁化率体来自不同类型/位置的地质体 ===
        if model_type == 'combined_randomwalk':
            # 密度: 长方体组合; 磁化率: 随机游走组合
            n_rho = rng.randint(2, 4)
            n_kappa = rng.randint(2, 4)
            rho_models = [generate_cuboid(nx, ny, nz, density_range, (0.0, 0.0), rng=rng)[0]
                          for _ in range(n_rho)]
            # cuboid 返回 (rho, kappa)，我们只需要 rho 部分
            rho_models_raw = []
            for _ in range(n_rho):
                r, _ = generate_cuboid(nx, ny, nz, density_range, (0.0, 0.0), rng=rng)
                rho_models_raw.append(r)
            kappa_models_raw = []
            for _ in range(n_kappa):
                _, k = generate_random_walk_body(nx, ny, nz, (0.0, 0.0), susc_range, rng=rng)
                kappa_models_raw.append(k)

            rho = combine_models([(m, np.zeros_like(m)) for m in rho_models_raw],
                                 allow_intersect=True, intersect_mode='max')[0]
            kappa = combine_models([(np.zeros_like(m), m) for m in kappa_models_raw],
                                   allow_intersect=True, intersect_mode='max')[1]

        elif model_type == 'combined_multi_rw':
            # 多个独立随机游走体，密度和磁化率的游走路径不同
            n_rho = rng.randint(2, 4)
            n_kappa = rng.randint(2, 4)
            rho_models_raw = []
            for _ in range(n_rho):
                r, _ = generate_random_walk_body(nx, ny, nz, density_range, (0.0, 0.0), rng=rng)
                rho_models_raw.append(r)
            kappa_models_raw = []
            for _ in range(n_kappa):
                _, k = generate_random_walk_body(nx, ny, nz, (0.0, 0.0), susc_range, rng=rng)
                kappa_models_raw.append(k)

            rho = combine_models([(m, np.zeros_like(m)) for m in rho_models_raw],
                                 allow_intersect=True, intersect_mode='max')[0]
            kappa = combine_models([(np.zeros_like(m), m) for m in kappa_models_raw],
                                   allow_intersect=True, intersect_mode='max')[1]

        elif model_type == 'combined_mixed':
            # Type 6 不一致: 混合类型
            n_bodies = rng.randint(3, 5)
            rho_models_raw = []
            kappa_models_raw = []
            for i in range(n_bodies):
                choice = rng.choice(['cuboid', 'tilting', 'randomwalk'])
                if choice == 'cuboid':
                    r, k = generate_cuboid(nx, ny, nz, density_range, susc_range, rng=rng)
                elif choice == 'tilting':
                    r, k = generate_tilting_body(nx, ny, nz, density_range, susc_range, rng=rng)
                else:
                    r, k = generate_random_walk_body(nx, ny, nz, density_range, susc_range, rng=rng)

                # 不一致: 随机分配给 rho 或 kappa (大部分情况分离)
                if rng.random() < 0.7:
                    if rng.random() < 0.5:
                        rho_models_raw.append(r)
                    else:
                        kappa_models_raw.append(k)
                else:
                    rho_models_raw.append(r)
                    kappa_models_raw.append(k)

            if rho_models_raw:
                rho = combine_models([(m, np.zeros_like(m)) for m in rho_models_raw],
                                     allow_intersect=True, intersect_mode='max')[0]
            else:
                rho = np.zeros((nx, ny, nz), dtype=np.float32)

            if kappa_models_raw:
                kappa = combine_models([(np.zeros_like(m), m) for m in kappa_models_raw],
                                       allow_intersect=True, intersect_mode='max')[1]
            else:
                kappa = np.zeros((nx, ny, nz), dtype=np.float32)

        else:
            raise ValueError(f"未知 inconsistent model_type: {model_type}")

    else:
        raise ValueError(f"未知 consistency 类型: {consistency}")

    return rho, kappa


# ============================================================
# 主生成函数
# ============================================================

def generate_dataset(dataset_type: int = 1,
                     n_samples: Optional[int] = None,
                     nx: int = DEFAULT_NX,
                     ny: int = DEFAULT_NY,
                     nz: int = DEFAULT_NZ,
                     noise_gravity: float = DEFAULT_NOISE_GRAVITY,
                     noise_magnetic: float = DEFAULT_NOISE_MAGNETIC,
                     seed: int = 42,
                     verbose: bool = True) -> List[Dict]:
    """
    生成指定类型的数据集。

    参数:
        dataset_type: 1-6 (对应论文 Table I)
        n_samples: None 时使用 Table I 默认数量; 给定整数时按比例缩放
        nx, ny, nz: 网格尺寸 (默认 40x40x20)
        noise_gravity: 重力噪声水平 (归一化后)
        noise_magnetic: 磁异常噪声水平 (归一化后)
        seed: 随机种子
        verbose: 是否打印进度信息

    返回:
        samples: list of dict, 每个 dict 含:
          'rho': (nx,ny,nz) 密度模型
          'kappa': (nx,ny,nz) 磁化率模型
          'structural_sim': (nx,ny,nz) 结构相似性标签
          'gravity': (nx,ny) 重力异常 (已归一化+加噪)
          'magnetic': (nx,ny) 磁异常 (已归一化+加噪)
          'type': 数据集类型编号
          'consistency_type': 'global'/'partial'/'inconsistent'
    """
    set_seed(seed)

    if dataset_type not in DATASET_SPECS:
        raise ValueError(f"dataset_type 必须在 1-6 之间, 得到: {dataset_type}")

    spec = DATASET_SPECS[dataset_type]

    # 缩放样本数量
    if n_samples is not None:
        scale = n_samples / spec['total']
    else:
        scale = 1.0

    # 各一致性类别的样本数
    counts = {}
    for cons_key in ['global', 'partial', 'inconsistent']:
        counts[cons_key] = max(1, int(spec[cons_key]['n'] * scale))

    total = sum(counts.values())

    if verbose:
        print(f"[generate_dataset] Type {dataset_type}: "
              f"global={counts['global']}, partial={counts['partial']}, "
              f"inconsistent={counts['inconsistent']}, total={total}")
        print(f"[generate_dataset] 正演引擎: {'外部模块' if HAS_EXTERNAL_FORWARD else '内置简化版'}")

    # 不同数据集类型的物性参数分布 (Type 4/5/6 有不同的参数范围)
    type_params = _get_type_params(dataset_type)

    samples = []
    sample_idx = 0
    rng = np.random.RandomState(seed)

    for consistency, count in counts.items():
        model_type = spec[consistency]['model_type']

        for i in range(count):
            # 为每个样本创建独立的子rng (保证并行安全且可复现)
            sample_rng = np.random.RandomState(seed + dataset_type * 100000 + sample_idx)

            # 生成模型
            try:
                rho, kappa = generate_consistency_type(
                    consistency, model_type, nx, ny, nz,
                    type_params['density_range'],
                    type_params['susc_range'],
                    rng=sample_rng
                )
            except Exception as e:
                if verbose:
                    print(f"  [WARN] 样本 {sample_idx} 生成失败 ({e}), 用 cuboid 替代")
                rho, kappa = generate_cuboid(nx, ny, nz,
                                             type_params['density_range'],
                                             type_params['susc_range'],
                                             rng=sample_rng)

            # 结构相似性标签
            struct_sim = compute_structural_similarity(rho, kappa)

            # 正演计算观测数据
            gravity_raw = forward_gravity(rho)
            magnetic_raw = forward_magnetic(kappa)

            # 归一化到 [0,1]
            gravity_norm = normalize_data(gravity_raw)
            magnetic_norm = normalize_data(magnetic_raw)

            # 添加噪声
            gravity_noisy = add_noise(gravity_norm, noise_gravity, seed=seed + sample_idx * 2)
            magnetic_noisy = add_noise(magnetic_norm, noise_magnetic, seed=seed + sample_idx * 2 + 1)

            samples.append({
                'rho': rho.astype(np.float32),
                'kappa': kappa.astype(np.float32),
                'structural_sim': struct_sim.astype(np.float32),
                'gravity': gravity_noisy.astype(np.float32),
                'magnetic': magnetic_noisy.astype(np.float32),
                'type': dataset_type,
                'consistency_type': consistency,
            })

            sample_idx += 1

            if verbose and sample_idx % max(1, total // 10) == 0:
                print(f"  进度: {sample_idx}/{total} ({100*sample_idx/total:.1f}%)")

    if verbose:
        print(f"[generate_dataset] Type {dataset_type} 完成: {len(samples)} 个样本")

    return samples


def _get_type_params(dataset_type: int) -> Dict:
    """获取不同数据集类型的物性参数范围"""
    # Type 1-3: 标准参数
    # Type 4: 减半数量，参数不变
    # Type 5: 同 Type 3 但不同的参数分布 (更宽的范围)
    # Type 6: 特殊混合参数
    params = {
        1: {'density_range': (0.5, 1.5), 'susc_range': (0.05, 0.3)},
        2: {'density_range': (0.5, 1.5), 'susc_range': (0.05, 0.3)},
        3: {'density_range': (0.5, 1.5), 'susc_range': (0.05, 0.3)},
        4: {'density_range': (0.5, 1.5), 'susc_range': (0.05, 0.3)},  # 数量减半，参数不变
        5: {'density_range': (0.3, 2.0), 'susc_range': (0.02, 0.5)},  # 更宽的参数范围
        6: {'density_range': (0.4, 1.8), 'susc_range': (0.03, 0.4)},  # 混合参数
    }
    return params.get(dataset_type, params[1])


# ============================================================
# 批量生成全部 6 类数据集
# ============================================================

def generate_all_datasets(n_samples_per_type: Optional[int] = None,
                          nx: int = DEFAULT_NX,
                          ny: int = DEFAULT_NY,
                          nz: int = DEFAULT_NZ,
                          noise_gravity: float = DEFAULT_NOISE_GRAVITY,
                          noise_magnetic: float = DEFAULT_NOISE_MAGNETIC,
                          seed: int = 42,
                          verbose: bool = True) -> Dict[int, List[Dict]]:
    """
    生成全部 6 类数据集。

    参数:
        n_samples_per_type: 每类的样本数 (None=Table I 原始数量)
        其余参数同 generate_dataset

    返回:
        all_datasets: {dataset_type: [samples]}
    """
    all_datasets = {}
    total_start = time.time()

    for dt in range(1, 7):
        t0 = time.time()
        samples = generate_dataset(
            dataset_type=dt,
            n_samples=n_samples_per_type,
            nx=nx, ny=ny, nz=nz,
            noise_gravity=noise_gravity,
            noise_magnetic=noise_magnetic,
            seed=seed + dt * 1000,
            verbose=verbose
        )
        elapsed = time.time() - t0
        all_datasets[dt] = samples
        if verbose:
            print(f"  Type {dt}: {len(samples)} 样本, 耗时 {elapsed:.1f}s")

    total_elapsed = time.time() - total_start
    total_samples = sum(len(s) for s in all_datasets.values())
    if verbose:
        print(f"\n[全部完成] 共 {total_samples} 个样本, 总耗时 {total_elapsed:.1f}s")

    return all_datasets


# ============================================================
# 数据保存/加载
# ============================================================

def save_dataset(samples: List[Dict], filepath: str, compress: bool = True):
    """
    保存数据集到 .npz 文件。

    参数:
        samples: 样本列表
        filepath: 输出文件路径
        compress: 是否压缩
    """
    save_dict = {}
    for i, sample in enumerate(samples):
        prefix = f'sample_{i:06d}'
        save_dict[f'{prefix}_rho'] = sample['rho']
        save_dict[f'{prefix}_kappa'] = sample['kappa']
        save_dict[f'{prefix}_sim'] = sample['structural_sim']
        save_dict[f'{prefix}_gravity'] = sample['gravity']
        save_dict[f'{prefix}_magnetic'] = sample['magnetic']
        save_dict[f'{prefix}_type'] = np.array(sample['type'])
        save_dict[f'{prefix}_consistency'] = sample['consistency_type']

    # 元数据
    meta = {
        'n_samples': len(samples),
        'nx': samples[0]['rho'].shape[0] if samples else 0,
        'ny': samples[0]['rho'].shape[1] if samples else 0,
        'nz': samples[0]['rho'].shape[2] if samples else 0,
        'types': list(set(s['type'] for s in samples)),
        'forward_engine': 'external' if HAS_EXTERNAL_FORWARD else 'builtin_simple',
    }

    if compress:
        np.savez_compressed(filepath, **save_dict, __meta__=json.dumps(meta))
    else:
        np.savez(filepath, **save_dict, __meta__=json.dumps(meta))

    print(f"[保存] {len(samples)} 个样本 -> {filepath} "
          f"({os.path.getsize(filepath) / 1024 / 1024:.1f} MB)")


def load_dataset(filepath: str) -> Tuple[List[Dict], Dict]:
    """
    从 .npz 文件加载数据集。

    返回:
        samples: 样本列表
        meta: 元数据字典
    """
    data = np.load(filepath, allow_pickle=True)
    meta_str = str(data['__meta__'])
    meta = json.loads(meta_str)
    n_samples = meta['n_samples']

    samples = []
    for i in range(n_samples):
        prefix = f'sample_{i:06d}'
        samples.append({
            'rho': data[f'{prefix}_rho'],
            'kappa': data[f'{prefix}_kappa'],
            'structural_sim': data[f'{prefix}_sim'],
            'gravity': data[f'{prefix}_gravity'],
            'magnetic': data[f'{prefix}_magnetic'],
            'type': int(data[f'{prefix}_type']),
            'consistency_type': str(data[f'{prefix}_consistency']),
        })

    data.close()
    return samples, meta


# ============================================================
# 验证函数
# ============================================================

def validate_dataset(samples: List[Dict]) -> Dict[str, any]:
    """
    验证数据集质量。

    检查项:
      1. 样本数量
      2. 数据形状
      3. 数据范围 [0,1]
      4. 无 NaN / Inf
      5. 结构相似性标签逻辑正确
      6. 各类型统计

    返回:
        report: 验证报告字典
    """
    report = {'passed': True, 'errors': [], 'warnings': [], 'stats': {}}

    if len(samples) == 0:
        report['passed'] = False
        report['errors'].append('数据集为空')
        return report

    # ---- 基本统计 ----
    report['stats']['n_samples'] = len(samples)
    report['stats']['shape_rho'] = samples[0]['rho'].shape
    report['stats']['shape_gravity'] = samples[0]['gravity'].shape

    # ---- 逐样本检查 ----
    nan_count = 0
    inf_count = 0
    oor_count = 0  # out of range
    sim_errors = 0
    type_counts = {}
    cons_counts = {}

    gravity_vals = []
    magnetic_vals = []
    rho_vals = []
    kappa_vals = []
    sim_ratios = []  # 每个样本中 sim=1 的比例

    for idx, s in enumerate(samples):
        # NaN 检查
        for key in ['rho', 'kappa', 'structural_sim', 'gravity', 'magnetic']:
            arr = s[key]
            if np.any(np.isnan(arr)):
                nan_count += 1
                report['errors'].append(f'样本 {idx} {key} 含 NaN')
            if np.any(np.isinf(arr)):
                inf_count += 1
                report['errors'].append(f'样本 {idx} {key} 含 Inf')

        # 范围检查 (gravity/magnetic 应在 [0,1])
        g = s['gravity']
        m = s['magnetic']
        if np.any(g < 0) or np.any(g > 1):
            oor_count += 1
        if np.any(m < 0) or np.any(m > 1):
            oor_count += 1

        # 结构相似性逻辑检查
        rho = s['rho']
        kappa = s['kappa']
        sim = s['structural_sim']
        expected_sim = ((rho != 0) & (kappa != 0)).astype(np.float32)
        if not np.array_equal(sim, expected_sim):
            sim_errors += 1
            report['warnings'].append(f'样本 {idx} 结构相似性标签与实际不符')

        # 统计
        t = s['type']
        c = s['consistency_type']
        type_counts[t] = type_counts.get(t, 0) + 1
        cons_counts[c] = cons_counts.get(c, 0) + 1

        gravity_vals.append(g)
        magnetic_vals.append(m)
        rho_vals.append(rho)
        kappa_vals.append(kappa)
        sim_ratios.append(float(sim.mean()))

    report['stats']['nan_samples'] = nan_count
    report['stats']['inf_samples'] = inf_count
    report['stats']['out_of_range_samples'] = oor_count
    report['stats']['sim_label_errors'] = sim_errors
    report['stats']['type_distribution'] = type_counts
    report['stats']['consistency_distribution'] = cons_counts

    # ---- 数据分布统计 ----
    all_g = np.concatenate([v.ravel() for v in gravity_vals])
    all_m = np.concatenate([v.ravel() for v in magnetic_vals])
    all_rho = np.concatenate([v.ravel() for v in rho_vals])
    all_kappa = np.concatenate([v.ravel() for v in kappa_vals])

    report['stats']['gravity'] = {
        'mean': float(all_g.mean()), 'std': float(all_g.std()),
        'min': float(all_g.min()), 'max': float(all_g.max()),
    }
    report['stats']['magnetic'] = {
        'mean': float(all_m.mean()), 'std': float(all_m.std()),
        'min': float(all_m.min()), 'max': float(all_m.max()),
    }
    report['stats']['rho'] = {
        'mean': float(all_rho.mean()), 'std': float(all_rho.std()),
        'min': float(all_rho.min()), 'max': float(all_rho.max()),
    }
    report['stats']['kappa'] = {
        'mean': float(all_kappa.mean()), 'std': float(all_kappa.std()),
        'min': float(all_kappa.min()), 'max': float(all_kappa.max()),
    }
    report['stats']['structural_similarity_ratio'] = {
        'mean': float(np.mean(sim_ratios)),
        'min': float(min(sim_ratios)),
        'max': float(max(sim_ratios)),
    }

    # ---- 判定 ----
    if nan_count > 0 or inf_count > 0:
        report['passed'] = False
    if sim_errors > len(samples) * 0.1:  # 允许少量误差
        report['passed'] = False
        report['errors'].append(f'结构相似性错误过多: {sim_errors}/{len(samples)}')

    return report


if __name__ == '__main__':
    # 快速测试: 生成小规模验证集
    print("=" * 60)
    print("合成数据集生成器 - 快速测试")
    print("=" * 60)

    # 测试各基础模型生成
    print("\n--- 测试基础模型 ---")
    rng_test = np.random.RandomState(42)

    rho_c, kappa_c = generate_cuboid(40, 40, 20, rng=rng_test)
    print(f"长方体: rho shape={rho_c.shape}, nonzero={np.count_nonzero(rho_c)}, "
          f"kappa nonzero={np.count_nonzero(kappa_c)}")

    rho_t, kappa_t = generate_tilting_body(40, 40, 20, rng=np.random.RandomState(42))
    print(f"倾斜体: rho shape={rho_t.shape}, nonzero={np.count_nonzero(rho_t)}, "
          f"kappa nonzero={np.count_nonzero(kappa_t)}")

    rho_rw, kappa_rw = generate_random_walk_body(40, 40, 20, rng=np.random.RandomState(42))
    print(f"随机游走: rho shape={rho_rw.shape}, nonzero={np.count_nonzero(rho_rw)}, "
          f"kappa nonzero={np.count_nonzero(kappa_rw)}")

    # 测试组合
    print("\n--- 测试组合模型 ---")
    models = [
        generate_cuboid(40, 40, 20, rng=np.random.RandomState(10)),
        generate_cuboid(40, 40, 20, rng=np.random.RandomState(20)),
    ]
    rho_comb, kappa_comb = combine_models(models)
    print(f"组合: rho nonzero={np.count_nonzero(rho_comb)}, "
          f"kappa nonzero={np.count_nonzero(kappa_comb)}")

    # 测试结构相似性
    print("\n--- 测试结构相似性 ---")
    sim = compute_structural_similarity(rho_c, kappa_c)
    print(f"S shape={sim.shape}, S=1 比例={sim.mean():.3f}, S=0 比例={1-sim.mean():.3f}")

    # 测试正演
    print("\n--- 测试正演 ---")
    g = forward_gravity(rho_c)
    m = forward_magnetic(kappa_c)
    print(f"重力异常: shape={g.shape}, range=[{g.min():.6f}, {g.max():.6f}]")
    print(f"磁异常: shape={m.shape}, range=[{m.min():.6f}, {m.max():.6f}]")

    # 测试归一化和加噪
    gn = normalize_data(g)
    ga = add_noise(gn, 0.005, seed=42)
    print(f"归一化重力: range=[{gn.min():.4f}, {gn.max():.4f}]")
    print(f"加噪重力: range=[{ga.min():.4f}, {ga.max():.4f}]")

    mn = normalize_data(m)
    ma = add_noise(mn, 0.108, seed=42)
    print(f"归一化磁: range=[{mn.min():.4f}, {mn.max():.4f}]")
    print(f"加噪磁: range=[{ma.min():.4f}, {ma.max():.4f}]")

    # 小规模数据集测试
    print("\n--- 小规模数据集测试 (每类 30 样本) ---")
    small_ds = generate_dataset(dataset_type=1, n_samples=30, seed=42, verbose=True)
    report = validate_dataset(small_ds)
    print(f"\n验证通过: {report['passed']}")
    if report['errors']:
        print(f"错误: {report['errors']}")
    if report['warnings']:
        print(f"警告 ({len(report['warnings'])}): {report['warnings'][:3]}...")
    print(f"\n统计摘要:")
    for k, v in report['stats'].items():
        if isinstance(v, dict) and 'mean' in v:
            print(f"  {k}: mean={v['mean']:.4f}, std={v.get('std', 0):.4f}, "
                  f"range=[{v.get('min', 0):.4f}, {v.get('max', 0):.4f}]")
        elif not isinstance(v, dict):
            print(f"  {k}: {v}")

    print("\n" + "=" * 60)
    print("快速测试完成!")
    print("=" * 60)
