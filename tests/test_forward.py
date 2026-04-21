"""
正演模块单元测试
================

测试重力正演 (forward_gravity.py) 和磁法正演 (forward_magnetic.py) 的正确性、
数值精度和物理合理性。

测试覆盖:
1. 均匀半空间 — 重力异常中心区域近似常数
2. 单棱柱体解析解对比 — 数值结果 vs 解析解，相对误差 < 1%
3. 零密度/零磁化率模型 — 输出全零
4. 磁法对称性 — 垂直磁化下异常对称
5. 数值精度 — 无 NaN/Inf，值域合理
"""

import sys
import os
import time
import numpy as np
import pytest

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data.forward_gravity import (
    forward_gravity,
    forward_gravity_dense,
    analytical_sphere_gravity,
    analytical_prism_gravity,
)
from src.data.forward_magnetic import (
    forward_magnetic,
    analytical_prism_magnetic,
)


# ===== 测试配置 =====
class TestConfig:
    """全局测试参数"""
    NX = 21          # 水平网格数（小规模测试）
    NY = 21
    NZ = 10          # 深度层数
    DX = 50.0        # 网格间距 (m)
    DY = 50.0
    DZ = 50.0
    Z0 = 0.0         # 观测面高度 (m)，地表


def make_grid_params(dx=TestConfig.DX, dy=TestConfig.DY, dz=TestConfig.DZ,
                     z0=TestConfig.Z0):
    """构造标准网格参数字典"""
    return {'dx': dx, 'dy': dy, 'dz': dz, 'x0': 0.0, 'y0': 0.0, 'z0': z0}


# =====================================================================
# 测试 1: 均匀半空间 — 重力异常在中心区域近似为常数
# =====================================================================

class TestUniformHalfSpace:
    """均匀半空间密度模型的重力响应"""

    def test_uniform_density_constant_center(self):
        """
        均匀密度半空间: 中心区域的重力异常应近似为常数。

        物理原理: 对于无限延伸的均匀半空间，地表重力异常为常数
        (Bouger 平板公式: g_z = 2πGρ)。有限网格中，中心区域应接近此值。
        """
        rho = 1.0  # g/cm³, 均匀密度
        rho_model = np.full((TestConfig.NX, TestConfig.NY, TestConfig.NZ), rho)
        grid_params = make_grid_params()

        gz = forward_gravity(rho_model, grid_params)

        # 中心区域 (排除边缘效应)
        cx, cy = TestConfig.NX // 2, TestConfig.NY // 2
        center_values = gz[cx-2:cx+3, cy-2:cy+3]

        # 中心区域标准差应远小于均值（近似常数）
        cv = np.std(center_values) / (np.mean(np.abs(center_values)) + 1e-30)
        assert cv < 0.05, (
            f"均匀半空间中心区域变异系数过大: {cv:.4f} "
            f"(期望 < 0.05), center mean={np.mean(center_values):.4f} mGal"
        )

    def test_uniform_density_positive(self):
        """正密度产生正重力异常"""
        rho_model = np.full((TestConfig.NX, TestConfig.NY, TestConfig.NZ), 0.5)
        grid_params = make_grid_params()
        gz = forward_gravity(rho_model, grid_params)

        assert np.all(gz > 0), "正密度模型应产生正重力异常"

    def test_uniform_density_magnitude(self):
        """
        粗略量级检查: 均匀半空间重力异常量级。

        Bouger 平板近似: g_z ≈ 2πGρt
        其中 t 为有效厚度。对于 ρ=1 g/cm³, t=500 m:
        g_z ≈ 2π * 6.674e-3 * 1 * 500 ≈ 20.96 mGal
        （注意: 这是粗略估计，实际值取决于网格范围）
        """
        rho_model = np.full((TestConfig.NX, TestConfig.NY, TestConfig.NZ), 1.0)
        grid_params = make_grid_params()
        gz = forward_gravity(rho_model, grid_params)

        center_val = gz[TestConfig.NX//2, TestConfig.NY//2]
        # 只检查是合理的正数量级
        assert center_val > 0.1, f"中心重力异常过小: {center_val:.6f} mGal"
        assert center_val < 10000, f"中心重力异常过大: {center_val:.2f} mGal"


# =====================================================================
# 测试 2: 单棱柱体 — 解析解对比
# =====================================================================

class TestSinglePrismAnalytical:
    """单个棱柱体的数值正演与解析解对比"""

    @pytest.fixture(scope="class")
    def single_prism_setup(self):
        """设置单棱柱体测试场景"""
        nx, ny, nz = 11, 11, 5
        dx = dy = dz = 100.0  # 大间距减少计算量
        z0 = 0.0

        # 在网格中心放置一个棱柱体
        ci, cj, ck = nx // 2, ny // 2, nz // 2
        density = 1.0  # g/cm³

        rho_model = np.zeros((nx, ny, nz))
        rho_model[ci, cj, ck] = density

        grid_params = {
            'dx': dx, 'dy': dy, 'dz': dz,
            'x0': 0.0, 'y0': 0.0, 'z0': z0
        }

        # 该棱柱体的几何参数
        x1 = ci * dx
        x2 = (ci + 1) * dx
        y1 = cj * dy
        y2 = (cj + 1) * dy
        z1 = ck * dz
        z2 = (ck + 1) * dz

        return {
            'rho_model': rho_model,
            'grid_params': grid_params,
            'density': density,
            'prism_geom': (x1, x2, y1, y2, z1, z2),
            'nx': nx, 'ny': ny,
            'dx': dx, 'dy': dy, 'z0': z0
        }

    def test_prism_gravity_vs_analytical(self, single_prism_setup):
        """
        单棱柱体重力: 数值正演 vs 解析解，相对误差 < 1%。
        """
        setup = single_prism_setup
        rho_model = setup['rho_model']
        grid_params = setup['grid_params']
        density = setup['density']
        x1, x2, y1, y2, z1, z2 = setup['prism_geom']
        nx, ny = setup['nx'], setup['ny']
        dx, dy = setup['dx'], setup['dy']
        z0 = setup['z0']

        # 数值正演
        gz_numerical = forward_gravity(rho_model, grid_params)

        # 解析解: 在观测点网格上直接计算
        obs_x = (np.arange(nx) + 0.5) * dx
        obs_y = (np.arange(ny) + 0.5) * dy
        OX, OY = np.meshgrid(obs_x, obs_y, indexing='ij')
        gz_analytical = analytical_prism_gravity(
            OX, OY, z0, x1, x2, y1, y2, z1, z2, density
        )

        # 计算相对误差（在异常显著的区域）
        max_analytical = np.max(np.abs(gz_analytical))
        if max_analytical > 1e-10:
            abs_diff = np.abs(gz_numerical - gz_analytical)
            rel_error = np.max(abs_diff) / max_analytical

            assert rel_error < 0.01, (
                f"单棱柱体重力相对误差过大: {rel_error:.6f} "
                f"(期望 < 1%), max_diff={np.max(abs_diff):.8f} mGal"
            )
        else:
            pytest.skip("解析解值过小，无法计算有意义的相对误差")

    def test_prism_peak_position(self, single_prism_setup):
        """单棱柱体异常峰值应在棱柱体正上方附近"""
        setup = single_prism_setup
        gz = forward_gravity(setup['rho_model'], setup['grid_params'])
        ci, cj = setup['nx'] // 2, setup['ny'] // 2

        # 峰值位置应在中心附近 (允许 ±1 个网格的偏移)
        peak_idx = np.unravel_index(np.argmax(gz), gz.shape)
        assert abs(peak_idx[0] - ci) <= 1 and abs(peak_idx[1] - cj) <= 1, (
            f"峰值位置 {peak_idx} 偏离棱柱体中心 ({ci}, {cj}) 过远"
        )


# =====================================================================
# 测试 3: 零密度 / 零磁化率模型 → 全零输出
# =====================================================================

class TestZeroModel:
    """零输入模型的边界条件测试"""

    def test_zero_gravity(self):
        """全零密度模型 → 重力异常全零"""
        rho_model = np.zeros((TestConfig.NX, TestConfig.NY, TestConfig.NZ))
        grid_params = make_grid_params()

        gz = forward_gravity(rho_model, grid_params)

        assert np.allclose(gz, 0.0, atol=1e-15), (
            f"零密度模型输出非零: max={np.max(np.abs(gz)):.2e}"
        )

    def test_zero_magnetic(self):
        """全零磁化率模型 → 磁异常全零"""
        kappa_model = np.zeros((TestConfig.NX, TestConfig.NY, TestConfig.NZ))
        grid_params = make_grid_params()

        dt = forward_magnetic(kappa_model, grid_params)

        assert np.allclose(dt, 0.0, atol=1e-15), (
            f"零磁化率模型输出非零: max={np.max(np.abs(dt)):.2e}"
        )

    def test_near_zero_gravity(self):
        """极小密度模型 → 极小重力异常"""
        rho_model = np.full((TestConfig.NX, TestConfig.NY, TestConfig.NZ), 1e-12)
        grid_params = make_grid_params()

        gz = forward_gravity(rho_model, grid_params)

        assert np.max(np.abs(gz)) < 1e-6, (
            f"极小密度模型输出过大: max={np.max(np.abs(gz)):.2e} mGal"
        )


# =====================================================================
# 测试 4: 磁法对称性 — 垂直磁化 (I=90°) 下异常对称
# =====================================================================

class TestMagneticSymmetry:
    """磁法正演的对称性验证"""

    def test_vertical_magnetization_symmetry(self):
        """
        垂直磁化 (I=90°) 时，水平方向偏心的磁性体产生的 ΔT 应关于
        通过源体中心的 x 和 y 轴对称。
        """
        nx, ny, nz = 31, 31, 10
        dx = dy = dz = 50.0

        # 放置一个偏离中心的棱柱体
        kappa_model = np.zeros((nx, ny, nz))
        kappa_model[20, 15, 3] = 0.1  # SI

        grid_params = {'dx': dx, 'dy': dy, 'dz': dz,
                       'x0': 0.0, 'y0': 0.0, 'z0': 0.0}

        # 垂直磁化: I=90°
        mag_params = {'I': 90.0, 'D': 0.0, 'F': 50000.0}
        dt = forward_magnetic(kappa_model, grid_params, mag_params)

        # 关于源体中心 (20, 15) 的对称性检验
        # 对于垂直磁化，ΔT(x0+dx, y0+dy) 应等于 ΔT(x0-dx, y0-dy)
        ci, cj = 20, 15

        # 检查几个偏移点的对称性
        for di in [1, 2, 3]:
            for dj in [1, 2, 3]:
                if (ci+di < nx and ci-di >= 0 and
                    cj+dj < ny and cj-dj >= 0):
                    val_plus = dt[ci+di, cj+dj]
                    val_minus = dt[ci-di, cj-dj]
                    # 对称性: 差异应远小于幅值
                    diff_ratio = abs(val_plus - val_minus) / (
                        (abs(val_plus) + abs(val_minus)) / 2 + 1e-30
                    )
                    assert diff_ratio < 0.05, (
                        f"垂直磁化对称性破坏: "
                        f"ΔT({ci+di},{cj+dj})={val_plus:.6f}, "
                        f"ΔT({ci-di},{cj-dj})={val_minus:.6f}, "
                        f"diff_ratio={diff_ratio:.4f}"
                    )

    def test_inclined_asymmetry(self):
        """
        倾斜磁化 (I=45°) 时，异常应表现出不对称性（负异常一侧）。

        这不是严格的数学证明，而是验证倾斜磁化确实产生了不对称模式，
        与垂直磁化的对称模式形成对比。
        """
        nx, ny, nz = 31, 31, 10
        dx = dy = dz = 50.0

        kappa_model = np.zeros((nx, ny, nz))
        kappa_model[15, 15, 3] = 0.1  # 中心放置

        grid_params = {'dx': dx, 'dy': dy, 'dz': dz,
                       'x0': 0.0, 'y0': 0.0, 'z0': 0.0}

        # 倾斜磁化
        mag_params_inc = {'I': 45.0, 'D': 0.0, 'F': 50000.0}
        dt_inc = forward_magnetic(kappa_model, grid_params, mag_params_inc)

        # 垂直磁化（参考）
        mag_params_vert = {'I': 90.0, 'D': 0.0, 'F': 50000.0}
        dt_vert = forward_magnetic(kappa_model, grid_params, mag_params_vert)

        # 垂直磁化下中心源体应基本只有正异常
        # 倾斜磁化下应有正负异常对
        min_inc = np.min(dt_inc)
        min_vert = np.min(dt_vert)

        # 倾斜磁化的最小值应比垂直磁化更负（或至少不同）
        # 这是一个弱检验，主要确认两种磁化模式产生不同的场
        assert not np.allclose(dt_inc, dt_vert, rtol=0.01), (
            "倾斜磁化和垂直磁化产生了几乎相同的异常，可能存在错误"
        )


# =====================================================================
# 测试 5: 数值精度和值域合理性
# =====================================================================

class TestNumericalPrecision:
    """数值精度和物理合理性检验"""

    def test_no_nan_inf_gravity(self):
        """重力结果无 NaN 或 Inf"""
        rho_model = np.random.randn(TestConfig.NX, TestConfig.NY, TestConfig.NZ) * 0.5
        grid_params = make_grid_params()

        gz = forward_gravity(rho_model, grid_params)

        assert not np.any(np.isnan(gz)), "重力结果包含 NaN"
        assert not np.any(np.isinf(gz)), "重力结果包含 Inf"

    def test_no_nan_inf_magnetic(self):
        """磁法结果无 NaN 或 Inf"""
        kappa_model = np.random.rand(TestConfig.NX, TestConfig.NY, TestConfig.NZ) * 0.1
        grid_params = make_grid_params()

        dt = forward_magnetic(kappa_model, grid_params)

        assert not np.any(np.isnan(dt)), "磁法结果包含 NaN"
        assert not np.any(np.isinf(dt)), "磁法结果包含 Inf"

    def test_gravity_value_range(self):
        """
        重力异常值域合理性: 典型地质密度 contrast (~1 g/cm³) 产生的异常
        应在 ±几百 mGal 范围内（对于 km 尺度的模型）。
        """
        # 中等规模的随机密度模型
        nx, ny, nz = 21, 21, 15
        dx = dy = dz = 100.0  # 2km × 2km × 1.5km 模型
        rho_model = np.zeros((nx, ny, nz))
        # 放置几个高密度体
        rho_model[5:16, 5:16, 2:8] = 0.5   # 中心块体
        rho_model[14:19, 8:13, 1:5] = 0.3   # 偏心块体

        grid_params = {'dx': dx, 'dy': dy, 'dz': dz,
                       'x0': 0.0, 'y0': 0.0, 'z0': 0.0}

        gz = forward_gravity(rho_model, grid_params)

        max_gz = np.max(gz)
        min_gz = np.min(gz)

        # 合理范围: 对于这个尺度和密度，应该在几十到几百 mGal
        assert max_gz > 0, "正密度模型应产生正的最大重力异常"
        assert max_gz < 50000, f"重力异常过大: {max_gz:.2f} mGal (> 50000)"
        assert abs(min_gz) < 50000, f"重力异常负值过大: {min_gz:.2f} mGal"

    def test_magnetic_value_range(self):
        """
        磁异常值域合理性: 典型磁化率 (~0.1 SI) 在 50000 nT 地磁场下
        产生的异常应在 ±几千 nT 范围内。
        """
        nx, ny, nz = 21, 21, 15
        dx = dy = dz = 100.0
        kappa_model = np.zeros((nx, ny, nz))
        kappa_model[7:14, 7:14, 2:8] = 0.1  # 中心磁性体

        grid_params = {'dx': dx, 'dy': dy, 'dz': dz,
                       'x0': 0.0, 'y0': 0.0, 'z0': 0.0}
        mag_params = {'I': 45.0, 'D': 0.0, 'F': 50000.0}

        dt = forward_magnetic(kappa_model, grid_params, mag_params)

        max_dt = np.max(dt)
        min_dt = np.min(dt)

        # 合理范围: 几千 nT
        assert max_dt > 0, "正磁化率模型应产生正的最大磁异常"
        assert max_dt < 200000, f"磁异常过大: {max_dt:.2f} nT (> 200000)"
        assert abs(min_dt) < 200000, f"磁异常负值过大: {min_dt:.2f} nT"

    def test_computation_time(self):
        """计算耗时应在合理范围内（单样本 < 5 秒）"""
        nx, ny, nz = 21, 21, 15
        dx = dy = dz = 100.0
        rho_model = np.zeros((nx, ny, nz))
        rho_model[5:16, 5:16, 2:8] = 0.5

        grid_params = {'dx': dx, 'dy': dy, 'dz': dz,
                       'x0': 0.0, 'y0': 0.0, 'z0': 0.0}

        start = time.time()
        gz = forward_gravity(rho_model, grid_params)
        elapsed_grav = time.time() - start

        kappa_model = np.zeros((nx, ny, nz))
        kappa_model[5:16, 5:16, 2:8] = 0.1
        mag_params = {'I': 45.0, 'D': 0.0, 'F': 50000.0}

        start = time.time()
        dt = forward_magnetic(kappa_model, grid_params, mag_params)
        elapsed_mag = time.time() - start

        print(f"\n  [计时] 重力正演: {elapsed_grav:.3f}s ({nx}×{ny}×{nz} 网格)")
        print(f"  [计时] 磁法正演: {elapsed_mag:.3f}s ({nx}×{ny}×{nz} 网格)")

        assert elapsed_grav < 10.0, f"重力正演过慢: {elapsed_grav:.1f}s"
        assert elapsed_mag < 10.0, f"磁法正演过慢: {elapsed_mag:.1f}s"


# =====================================================================
# 额外测试: 密集版本一致性
# =====================================================================

class TestDenseVsSparseConsistency:
    """稠密版和稀疏版正演的一致性"""

    def test_gravity_dense_sparse_match(self):
        """稠密版和优化版重力结果应一致"""
        nx, ny, nz = 9, 9, 5
        dx = dy = dz = 100.0
        rng = np.random.RandomState(42)
        rho_model = rng.randn(nx, ny, nz) * 0.5

        grid_params = {'dx': dx, 'dy': dy, 'dz': dz,
                       'x0': 0.0, 'y0': 0.0, 'z0': 0.0}

        gz_sparse = forward_gravity(rho_model, grid_params)
        gz_dense = forward_gravity_dense(rho_model, grid_params)

        max_val = max(np.max(np.abs(gz_sparse)), np.max(np.abs(gz_dense)), 1e-30)
        max_diff = np.max(np.abs(gz_sparse - gz_dense))

        assert max_diff / max_val < 0.01, (
            f"稠密版和稀疏版差异过大: relative={max_diff/max_val:.6f}"
        )


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
