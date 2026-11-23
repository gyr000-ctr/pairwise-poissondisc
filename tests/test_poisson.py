"""泊松盘采样（Bridson）+ 变密度的验收测试。

覆盖：最小间距（均匀/变密度）、区域边界、同种子复现、变密度场密度误差
<=10%、1 万点耗时 <=400ms、并发独立、极小半径、退化区域、种子边界、
非法半径/密度/区域参数报错。
"""

import math
import threading
import time

import pytest

import poissondisc
from poissondisc import PoissonDisc


# ---------- 辅助 ----------

def _grid_violations(points, radii, cell_scale=1.0):
    """O(n) 校验任意点对距离 >= min(各自半径)。

    以全场最小半径 / sqrt(2) 建格，每个被占用格只存一个点；
    对每个点只需检查半径可达范围内的邻居格。
    """
    if len(points) < 2:
        return 0
    r_min = min(radii) * cell_scale
    cell = r_min / math.sqrt(2.0)
    occupied = {}
    for i, (x, y) in enumerate(points):
        occupied[(int(x / cell), int(y / cell))] = i

    violations = 0
    for i, (x, y) in enumerate(points):
        ri = radii[i]
        m = int(math.ceil(ri / cell))
        cx, cy = int(x / cell), int(y / cell)
        for gx in range(cx - m, cx + m + 1):
            for gy in range(cy - m, cy + m + 1):
                j = occupied.get((gx, gy))
                if j is None or j <= i:
                    continue
                qx, qy = points[j]
                d = math.hypot(x - qx, y - qy)
                if d < min(ri, radii[j]) - 1e-9:
                    violations += 1
    return violations


# ---------- 基础约束 ----------

def test_poisson_nonempty_and_types():
    pts = PoissonDisc(0.5, 1).sample(10.0, 10.0, k=10)
    assert isinstance(pts, list)
    assert len(pts) > 0
    assert all(len(p) == 2 for p in pts)
    assert all(isinstance(v, float) for p in pts for v in p)


def test_poisson_min_distance_uniform():
    r = 0.5
    pts = PoissonDisc(r, 2).sample(12.0, 12.0, k=30)
    assert len(pts) > 100
    assert _grid_violations(pts, [r] * len(pts)) == 0


def test_poisson_in_bounds():
    w, h = 10.0, 7.5
    pts = PoissonDisc(0.3, 3).sample(w, h)
    assert all(0.0 <= x <= w and 0.0 <= y <= h for x, y in pts)


def test_poisson_deterministic_same_seed():
    a = PoissonDisc(0.5, 123).sample(20.0, 20.0)
    b = PoissonDisc(0.5, 123).sample(20.0, 20.0)
    assert a == b
    # 同一实例重复采样也一致（无可变状态）
    p = PoissonDisc(0.5, 123)
    assert p.sample(20.0, 20.0) == p.sample(20.0, 20.0)


def test_poisson_different_seed_different_result():
    a = PoissonDisc(0.5, 1).sample(20.0, 20.0)
    b = PoissonDisc(0.5, 2).sample(20.0, 20.0)
    assert a != b


def test_poisson_density_uniform_count():
    # r=0.5 在 10x10 内至少应有几十个点
    pts = PoissonDisc(0.5, 3).sample(10.0, 10.0)
    assert len(pts) >= 100


# ---------- 变密度 ----------

def test_poisson_variable_density_min_distance():
    field = lambda x, y: 0.2 + 0.6 * (x / 40.0) + 0.2 * (y / 40.0)
    pts = PoissonDisc(field, 7).sample(40.0, 40.0, k=30)
    assert len(pts) > 1000
    radii = [field(x, y) for x, y in pts]
    assert _grid_violations(pts, radii) == 0
    # 稀疏区（半径大）点数应明显少于密集区（半径小）
    fine = [p for p in pts if p[0] < 10.0]
    coarse = [p for p in pts if p[0] > 30.0]
    assert len(fine) > len(coarse) * 1.5


def test_poisson_variable_density_error_within_10pct():
    """密度误差 <=10%：局部点密度应与 1/r(x,y)^2 成正比。

    用平滑线性梯度场（避免阶跃边界的硬墙效应），在相距很远、各自局部
    半径近似恒定的两个内缩条带内统计点数；基线取与条带同尺寸、相同
    局部半径的独立均匀采样（样本量同量级，避免缩放噪声）。
    """
    w, h = 120.0, 20.0
    r_fine, r_coarse = 0.35, 0.7

    def field(x, y):
        # 平滑梯度：左端 r_fine，右端 r_coarse
        return r_fine + (r_coarse - r_fine) * (x / w)

    # 两个内缩条带：x 方向远离外边界与彼此，y 方向内缩一个粗半径
    band_w = 8.0
    fine_x = (6.0, 6.0 + band_w)
    coarse_x = (w - 6.0 - band_w, w - 6.0)
    y_lo, y_hi = r_coarse, h - r_coarse

    def count_in(pts, x_span):
        return sum(1 for x, y in pts
                   if x_span[0] < x < x_span[1] and y_lo < y < y_hi)

    errors = []
    for seed in (21, 22, 23):
        pts = PoissonDisc(field, seed).sample(w, h, k=30)
        r_f = field((fine_x[0] + fine_x[1]) / 2.0, 0.0)
        r_c = field((coarse_x[0] + coarse_x[1]) / 2.0, 0.0)

        got_f = count_in(pts, fine_x)
        got_c = count_in(pts, coarse_x)
        assert got_f > 100 and got_c > 30

        # 同尺寸独立均匀场基线，3 个种子取均值降噪
        bf = [count_in(PoissonDisc(r_f, s).sample(w, h, k=30), fine_x)
              for s in (301, 302, 303)]
        bc = [count_in(PoissonDisc(r_c, s).sample(w, h, k=30), coarse_x)
              for s in (301, 302, 303)]
        expected_f = sum(bf) / len(bf)
        expected_c = sum(bc) / len(bc)

        errors.append(abs(got_f - expected_f) / expected_f)
        errors.append(abs(got_c - expected_c) / expected_c)

    assert max(errors) <= 0.10, f"密度误差超出 10%: {errors}"


def test_poisson_variable_field_reproducible():
    field = lambda x, y: 0.3 + 0.4 * (0.5 + 0.5 * math.sin(x * 0.3))
    a = PoissonDisc(field, 55).sample(30.0, 30.0)
    b = PoissonDisc(field, 55).sample(30.0, 30.0)
    assert a == b


# ---------- 性能 ----------

def test_poisson_10k_points_under_400ms():
    # r=0.5 在 64x64 上稳定产出 1 万+ 点
    t0 = time.perf_counter()
    pts = PoissonDisc(0.5, 42).sample(64.0, 64.0, k=30)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    assert len(pts) >= 10000
    assert elapsed_ms < 400.0, f"耗时 {elapsed_ms:.1f} ms 超过 400 ms"


def test_poisson_variable_10k_points_under_400ms():
    field = lambda x, y: 0.4 + 0.2 * (x / 64.0)
    t0 = time.perf_counter()
    pts = PoissonDisc(field, 42).sample(64.0, 64.0, k=30)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    assert len(pts) >= 10000
    assert elapsed_ms < 400.0, f"变密度耗时 {elapsed_ms:.1f} ms 超过 400 ms"


# ---------- 并发独立 ----------

def test_poisson_concurrent_independent():
    results = {}
    errors = []

    def worker(seed):
        try:
            local = PoissonDisc(0.4, seed).sample(25.0, 25.0, k=20)
            results[seed] = local
        except Exception as exc:  # pragma: no cover - 失败路径
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(s,)) for s in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors
    assert len(results) == 8
    # 同种子结果与串行调用完全一致
    for seed, got in results.items():
        assert got == PoissonDisc(0.4, seed).sample(25.0, 25.0, k=20)
    # 不同种子互不相同
    assert len({tuple(v) for v in results.values()}) == 8


def test_poisson_shared_instance_thread_safe():
    sampler = PoissonDisc(0.4, 99)
    out = []
    lock = threading.Lock()

    def worker():
        pts = sampler.sample(20.0, 20.0, k=15)
        with lock:
            out.append(pts)

    threads = [threading.Thread(target=worker) for _ in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    expected = sampler.sample(20.0, 20.0, k=15)
    assert len(out) == 6
    assert all(pts == expected for pts in out)


# ---------- 极小半径 ----------

def test_poisson_tiny_radius_dense_and_valid():
    r = 0.02
    pts = PoissonDisc(r, 4).sample(2.0, 2.0, k=30)
    # 面积 4，理论上限约 4/(2/sqrt(3)*r^2) ≈ 8660，实际数千点
    assert len(pts) > 2000
    assert _grid_violations(pts, [r] * len(pts)) == 0
    assert all(0.0 <= x <= 2.0 and 0.0 <= y <= 2.0 for x, y in pts)


def test_poisson_tiny_radius_variable():
    field = lambda x, y: 0.02 + 0.03 * (x / 2.0)
    pts = PoissonDisc(field, 5).sample(2.0, 2.0, k=20)
    assert len(pts) > 1000
    radii = [field(x, y) for x, y in pts]
    assert _grid_violations(pts, radii) == 0


# ---------- 退化区域 ----------

def test_poisson_degenerate_area_returns_list():
    # 区域远小于半径：只能放下 0 或 1 个点，且不卡死
    pts = PoissonDisc(10.0, 6).sample(0.1, 0.1, k=5)
    assert isinstance(pts, list)
    assert len(pts) <= 1
    assert all(0.0 <= x <= 0.1 and 0.0 <= y <= 0.1 for x, y in pts)


def test_poisson_thin_strip():
    # 窄条区域：宽度小于半径，点只能沿条带分布
    pts = PoissonDisc(0.5, 7).sample(10.0, 0.2, k=20)
    assert isinstance(pts, list)
    assert all(0.0 <= x <= 10.0 and 0.0 <= y <= 0.2 for x, y in pts)
    if len(pts) >= 2:
        assert _grid_violations(pts, [0.5] * len(pts)) == 0


# ---------- 种子边界 ----------

@pytest.mark.parametrize("seed", [0, 1, -1, 2**63, -(2**63), 3.14, "", "seed",
                                  True, b"\x00\x01"])
def test_poisson_seed_boundaries(seed):
    a = PoissonDisc(0.5, seed).sample(8.0, 8.0, k=10)
    b = PoissonDisc(0.5, seed).sample(8.0, 8.0, k=10)
    assert a == b
    assert all(0.0 <= x <= 8.0 and 0.0 <= y <= 8.0 for x, y in a)


def test_poisson_seed_none_is_nondeterministic_but_valid():
    # Python 语义：seed=None 委托 OS 熵，两次不同，但每次输出都合法
    a = PoissonDisc(0.5, None).sample(8.0, 8.0, k=10)
    b = PoissonDisc(0.5, None).sample(8.0, 8.0, k=10)
    assert a != b
    for pts in (a, b):
        assert all(0.0 <= x <= 8.0 and 0.0 <= y <= 8.0 for x, y in pts)


def test_poisson_large_seed_reproducible():
    seed = 123456789012345678901234567890
    a = PoissonDisc(0.5, seed).sample(10.0, 10.0)
    b = PoissonDisc(0.5, seed).sample(10.0, 10.0)
    assert a == b


# ---------- 非法参数 ----------

@pytest.mark.parametrize("bad", [0.0, -0.5, float("nan"), float("inf"),
                                 float("-inf"), "0.5", None, [0.5], object()])
def test_poisson_invalid_radius_raises(bad):
    with pytest.raises(ValueError):
        PoissonDisc(bad, 1)


def test_poisson_invalid_dimensions_raise():
    p = PoissonDisc(0.5, 1)
    with pytest.raises(ValueError):
        p.sample(0.0, 10.0)
    with pytest.raises(ValueError):
        p.sample(10.0, -1.0)
    with pytest.raises(ValueError):
        p.sample(float("nan"), 10.0)
    with pytest.raises(ValueError):
        p.sample(10.0, float("inf"))
    with pytest.raises(ValueError):
        p.sample("10", 10.0)


def test_poisson_invalid_k_raises():
    p = PoissonDisc(0.5, 1)
    with pytest.raises(ValueError):
        p.sample(10.0, 10.0, k=0)
    with pytest.raises(ValueError):
        p.sample(10.0, 10.0, k=-3)
    with pytest.raises(ValueError):
        p.sample(10.0, 10.0, k=1.5)
    with pytest.raises(ValueError):
        p.sample(10.0, 10.0, k=True)


@pytest.mark.parametrize("bad_value", [0.0, -1.0, float("nan"), float("inf")])
def test_poisson_field_illegal_value_raises(bad_value):
    field = lambda x, y: bad_value
    with pytest.raises(ValueError):
        PoissonDisc(field, 1).sample(10.0, 10.0, k=5)


def test_poisson_field_non_numeric_raises():
    with pytest.raises((ValueError, TypeError)):
        PoissonDisc(lambda x, y: "0.5", 1).sample(10.0, 10.0, k=5)
    with pytest.raises((ValueError, TypeError)):
        PoissonDisc(lambda x, y: None, 1).sample(10.0, 10.0, k=5)


def test_poisson_field_late_failure_raises():
    # 场在采样过程中才返回非法值，也要报错而不是静默产出
    calls = {"n": 0}

    def field(x, y):
        calls["n"] += 1
        if calls["n"] > 50:
            return 0.0
        return 0.5

    with pytest.raises(ValueError):
        PoissonDisc(field, 1).sample(20.0, 20.0, k=30)


# ---------- 网格健壮性回归 ----------

def _brute_violations(points, field):
    """独立 O(n^2) 校验：任意点对距离 >= min(各自半径)，不依赖网格。"""
    bad = 0
    for i in range(len(points)):
        xi, yi = points[i]
        ri = field(xi, yi)
        for j in range(i + 1, len(points)):
            xj, yj = points[j]
            if math.hypot(xi - xj, yi - yj) < min(ri, field(xj, yj)) - 1e-9:
                bad += 1
    return bad


def test_poisson_forced_rebuild_completeness(monkeypatch):
    # 把预扫描探针压到 1 个，使估计最小半径几乎必然偏大，主循环被迫
    # 频繁重建网格；重建后邻居搜索仍须完备（链表网格每格可存多点）。
    import poissondisc.core as core
    monkeypatch.setattr(core, "_PROBE_COUNT", 1)

    fields = {
        "linear": lambda x, y: 0.1 + 0.5 * (x / 6.0),
        "sine": lambda x, y: 0.2 + 0.3 * (0.5 + 0.5 * math.sin(x * 0.8)),
        "step": lambda x, y: 0.12 if x < 3.0 else 0.45,
    }
    for field in fields.values():
        for seed in range(3):
            pts = PoissonDisc(field, seed).sample(6.0, 6.0, k=20)
            assert len(pts) > 50
            assert _brute_violations(pts, field) == 0
            assert all(0.0 <= x <= 6.0 and 0.0 <= y <= 6.0 for x, y in pts)


def test_poisson_sparse_grid_branch(monkeypatch):
    # 降低稠密网格格数上限，强制走稀疏字典网格（含链表邻居搜索）。
    import poissondisc.core as core
    monkeypatch.setattr(core, "_CELL_LIMIT", 1000)

    pts = PoissonDisc(0.05, 2).sample(6.0, 6.0, k=15)
    assert len(pts) > 1000
    assert _grid_violations(pts, [0.05] * len(pts)) == 0

    field = lambda x, y: 0.05 + 0.15 * (x / 6.0)
    pts = PoissonDisc(field, 5).sample(6.0, 6.0, k=15)
    assert _grid_violations(pts, [field(*p) for p in pts]) == 0


def test_poisson_grid_allows_multiple_points_per_cell():
    # 变密度骤变时两个合法点可能落入同一网格格；链表存储不得丢点。
    # 用强反差场在小区域内生成密集点，再用暴力校验确认无遗漏冲突。
    field = lambda x, y: 0.05 if x < 1.5 else 1.2
    pts = PoissonDisc(field, 1).sample(4.0, 4.0, k=20)
    assert len(pts) > 1000
    assert _brute_violations(pts, field) == 0
