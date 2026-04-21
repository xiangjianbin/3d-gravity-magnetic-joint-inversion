# 正演模块验证报告

> **生成时间**: 2026-04-21
> **模块**: `src/data/forward_gravity.py` + `src/data/forward_magnetic.py`
> **测试文件**: `tests/test_forward.py`

---

## 1. 测试结果总览

| 测试类 | 测试项 | 结果 | 说明 |
|--------|--------|------|------|
| TestUniformHalfSpace | test_uniform_density_constant_center | **PASS** | 均匀半空间中心区域 CV < 5% |
| TestUniformHalfSpace | test_uniform_density_positive | **PASS** | 正密度 -> 正异常 |
| TestUniformHalfSpace | test_uniform_density_magnitude | **PASS** | 量级合理 (0.1~10000 mGal) |
| TestSinglePrismAnalytical | test_prism_gravity_vs_analytical | **PASS** | 数值 vs 解析解相对误差 < 1% |
| TestSinglePrismAnalytical | test_prism_peak_position | **PASS** | 峰值位置正确 |
| TestZeroModel | test_zero_gravity | **PASS** | 零密度 -> 全零 |
| TestZeroModel | test_zero_magnetic | **PASS** | 零磁化率 -> 全零 |
| TestZeroModel | test_near_zero_gravity | **PASS** | 极小密度 -> 极小异常 |
| TestMagneticSymmetry | test_vertical_magnetization_symmetry | **PASS** | I=90° 对称性成立 |
| TestMagneticSymmetry | test_inclined_asymmetry | **PASS** | I=45° 与 I=90° 异常不同 |
| TestNumericalPrecision | test_no_nan_inf_gravity | **PASS** | 无 NaN/Inf |
| TestNumericalPrecision | test_no_nan_inf_magnetic | **PASS** | 无 NaN/Inf |
| TestNumericalPrecision | test_gravity_value_range | **PASS** | 重力值域合理 |
| TestNumericalPrecision | test_magnetic_value_range | **PASS** | 磁法值域合理 |
| TestNumericalPrecision | test_computation_time | **PASS** | 重力 0.235s, 磁法 0.580s |
| TestDenseVsSparseConsistency | test_gravity_dense_sparse_match | **PASS** | 稠密/稀疏版一致 |

**总计: 16/16 PASS**

---

## 2. 计算精度

### 2.1 单棱柱体解析解对比

- **测试配置**: 11×11×5 网格, dx=dy=dz=100m, 单个中心棱柱体, rho=1.0 g/cm³
- **相对误差**: < 1% (满足阈值要求)
- **峰值位置**: 正确位于棱柱体正上方 (±1 网格容差内)

### 2.2 均匀半空间

- **测试配置**: 21×21×10 网格, dx=dy=dz=50m, 均匀密度 1.0 g/cm³
- **中心区域变异系数**: < 5% (符合 Bouger 平板近似的常数特性)
- **物理一致性**: 正密度产生正重力异常

### 2.3 磁法对称性

- **垂直磁化 (I=90°)**: 关于源体中心的对称性偏差 < 5%
- **倾斜磁化 (I=45°)**: 与垂直磁化结果显著不同（确认方向效应生效）

---

## 3. 计算耗时

基于 21×21×15 网格 (中等规模) 的单样本计时:

| 模块 | 耗时 | 网格规模 | 备注 |
|------|------|----------|------|
| 重力正演 | **0.235 s** | 21×21×15 | 含非零掩码优化 |
| 磁法正演 | **0.580 s** | 21×21×15 | 6 个张量分量，约 2.5x 重力 |

### 性能分析

- 当前实现采用"外层单元循环 + 内核观测点向量化"策略
- 对于论文规模 (81×81×20 = 131,220 cells)，预估:
  - 若全部非零: ~30-60s/样本 (需进一步优化)
  - 实际地质模型通常稀疏 (<10% 非零): ~3-6s/样本
- 后续可考虑: Numba JIT、Cython 或分块并行加速

---

## 4. 实现的文件清单

| 文件路径 | 功能 | 行数 |
|----------|------|------|
| `src/data/forward_gravity.py` | Blakely 棱柱体重力正演 + 解析解辅助函数 | ~230 行 |
| `src/data/forward_magnetic.py` | 总场异常 ΔT 磁法正演 + 张量核函数 | ~190 行 |
| `tests/test_forward.py` | 16 项单元测试 (5 类) | ~380 行 |

### 核心函数签名

```python
# 重力正演
def forward_gravity(rho_model: ndarray(nx,ny,nz), grid_params: dict) -> ndarray(nx,ny)  # mGal

# 磁法正演
def forward_magnetic(kappa_model: ndarray(nx,ny,nz), grid_params: dict,
                     mag_params: Optional[dict]=None) -> ndarray(nx,ny)  # nT

# 辅助: 解析解（用于验证）
def analytical_prism_gravity(x0, y0, z0, x1,x2,y1,y2,z1,z2, density) -> gz  # mGal
def analytical_prism_magnetic(x0, y0, z0, x1,x2,y1,y2,z1,z2, kappa, I,D,F) -> dT  # nT
```

---

## 5. 物理参数与假设

| 参数 | 值 | 来源 |
|------|-----|------|
| 引力常数 G | 6.674e-3 mGal·cm³/(g·m²) | CGS-SI 混合单位转换 |
| 默认磁倾角 I | 45° | Gap 7: 中国中纬度典型值 |
| 默认磁偏角 D | 0° | Gap 7: 简化假设 |
| 默认地磁场 F | 50000 nT | Gap 7: 典型地磁强度 |
| z 向下为正 | 是 | 地球物理标准约定 |

---

## 6. 已知限制与后续优化方向

1. **性能**: 大规模全稠密模型 (>50×50×20) 计算较慢，建议后续加入 Numba JIT 加速
2. **z=0 观测面**: 当观测点恰好在棱柱体顶面 (z0=z1) 时，atan 项使用小量正则化避免奇异性，精度略有损失但数值稳定
3. **磁法公式**: 采用感应磁化近似 (M = κH)，未考虑剩磁和自退磁效应
4. **坐标系**: 统一使用 x(Easting)-y(Northing)-z(向下深度)，与论文一致
