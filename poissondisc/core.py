"""变密度泊松盘采样（Bridson 算法）。

支持两种半径模式：

* 标量半径：全场统一最小间距；
* 可调用密度场 ``radius(x, y) -> float``：半径随位置变化，数值越大越稀疏。

间距约束（同位置最小半径语义）：对任意两个采样点 ``a``、``b``，

    dist(a, b) >= min(radius(a), radius(b))

只需在候选点插入时检查它与所有已有点的距离不小于候选点自身半径
``rc``：因 ``min(rc, rj) <= rc``，成对约束自动成立（旧点插入时已对
自身半径做过同样的检查）。

空间索引采用边长 ``r_min / sqrt(2)`` 的网格，**每格以扁平链表存放
任意多个点**（``heads`` + ``nxt``），因此即使变密度场的局部半径
骤变、两个点落入同一格也不会互相覆盖，邻居搜索始终完备。网格格数
超过 ``_CELL_LIMIT`` 时自动退化为稀疏字典网格，避免极小半径下
O((L/r)^2) 的内存爆炸。

同一种子生成的点序列完全一致；每次 :meth:`PoissonDisc.sample` 都
从独立的本地 :class:`random.Random` 出发，实例无可变状态，可在
多个线程中安全并发调用。
"""

import hashlib
import math
import random

__all__ = ["PoissonDisc"]

_SQRT2 = math.sqrt(2.0)
_SQRT3 = math.sqrt(3.0)
_TWO_PI = 2.0 * math.pi
_CELL_LIMIT = 1 << 22   # 稠密网格最多约 400 万格，超出后改用稀疏字典网格
_PROBE_COUNT = 1024     # 变密度场预扫描点数
_REBUILD_FACTOR = 0.75  # 新最小半径低于当前尺度的 75% 才重建网格
_PROBE_SALT = 0x5EEDCAFE


def _is_finite_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) \
        and math.isfinite(value)


def _coerce_radius(value):
    """校验并转换密度场返回的半径值，非法时抛 ValueError。"""
    try:
        positive = value > 0.0
    except TypeError:
        raise ValueError("密度场返回的半径必须为正的有限数") from None
    if not positive or not math.isfinite(value):
        raise ValueError("密度场返回的半径必须为正的有限数")
    return float(value)


def _probe_seed(seed):
    """把任意合法种子确定性映射为 int，供预扫描的独立随机源使用。"""
    if isinstance(seed, bool):
        return int(seed) ^ _PROBE_SALT
    if isinstance(seed, int):
        return seed ^ _PROBE_SALT
    if seed is None:
        return _PROBE_SALT
    if isinstance(seed, float):
        raw = repr(seed).encode()
    elif isinstance(seed, (bytes, bytearray)):
        raw = b"probe:" + bytes(seed)
    elif isinstance(seed, str):
        raw = b"probe:" + seed.encode()
    else:
        try:
            raw = repr(seed).encode()
        except Exception as exc:  # pragma: no cover - 极端自定义类型
            raise ValueError("不支持的随机种子类型") from exc
    return int.from_bytes(hashlib.sha512(raw).digest(), "little")


class PoissonDisc:
    """Bridson 泊松盘采样器。

    Parameters
    ----------
    radius:
        标量正半径，或形如 ``radius(x, y) -> float`` 的可调用变密度场，
        返回值必须为正的有限数，否则构造时（标量）或采样时（场）抛
        :class:`ValueError`。
    seed:
        随机种子，任何 :class:`random.Random` 接受的值均可
        （含 int/float/str/bytes/None）；``None`` 委托系统熵，
        两次采样结果不同。
    """

    def __init__(self, radius, seed):
        if callable(radius):
            self._field = radius
            self._constant = None
        elif _is_finite_number(radius) and radius > 0.0:
            self._field = None
            self._constant = float(radius)
        else:
            raise ValueError(
                "radius 必须为正的有限数，或返回正有限数的可调用密度场"
            )
        self._seed = seed

    @property
    def seed(self):
        return self._seed

    def radius_at(self, x, y):
        """返回位置 (x, y) 处的最小半径（非法场值会在此抛 ValueError）。"""
        if self._field is not None:
            return _coerce_radius(self._field(x, y))
        return self._constant

    def sample(self, width, height, k=30):
        """在 ``[0, width] × [0, height]`` 内生成采样点。

        Returns
        -------
        list[tuple[float, float]]
            采样点列表，坐标均为 ``float`` 且全部落在闭区域内。区域
            相对半径过于局促、无法容纳第二个点时，返回仅含首点的列表。
        """
        if not _is_finite_number(width) or width <= 0.0:
            raise ValueError("width 必须为正的有限数")
        if not _is_finite_number(height) or height <= 0.0:
            raise ValueError("height 必须为正的有限数")
        if not isinstance(k, int) or isinstance(k, bool) or k < 1:
            raise ValueError("k 必须为 >= 1 的整数")

        width = float(width)
        height = float(height)
        if self._constant is not None:
            return _sample_uniform(self._constant, self._seed,
                                   width, height, k)
        return _sample_variable(self._field, self._seed, width, height, k)


def _sample_uniform(r, seed, width, height, k):
    """固定半径的快速路径。"""
    rng = random.Random(seed)  # 每次采样独立随机源，保证并发独立
    rng_random = rng.random
    rng_randrange = rng.randrange
    sqrt = math.sqrt
    cos = math.cos
    sin = math.sin

    cell = r / _SQRT2
    nx = max(1, int(math.ceil(width / cell)))
    ny = max(1, int(math.ceil(height / cell)))
    dense = nx * ny <= _CELL_LIMIT
    heads = [-1] * (nx * ny) if dense else {}
    nxt = []
    nxm1 = nx - 1
    nym1 = ny - 1
    r2 = r * r

    x = rng_random() * width
    y = rng_random() * height
    xs = [x]
    ys = [y]
    gx0 = int(x / cell)
    gy0 = int(y / cell)
    if gx0 > nxm1:
        gx0 = nxm1
    if gy0 > nym1:
        gy0 = nym1
    heads[gx0 * ny + gy0] = 0
    nxt.append(-1)
    active = [0]
    count = 1

    while active:
        ai = rng_randrange(len(active))
        idx = active[ai]
        px = xs[idx]
        py = ys[idx]
        accepted = False

        for _ in range(k):
            # 环形均匀采样 [r, 2r)：半径密度正比于半径（面积均匀）
            ang = rng_random() * _TWO_PI
            rr = r * sqrt(_SQRT3 * rng_random() + 1.0)
            x = px + rr * cos(ang)
            y = py + rr * sin(ang)
            if x < 0.0 or x > width or y < 0.0 or y > height:
                continue

            cx = int(x / cell)
            cy = int(y / cell)
            if cx > nxm1:
                cx = nxm1
            if cy > nym1:
                cy = nym1
            gxlo = cx - 2
            if gxlo < 0:
                gxlo = 0
            gxhi = cx + 2
            if gxhi > nxm1:
                gxhi = nxm1
            gylo = cy - 2
            if gylo < 0:
                gylo = 0
            gyhi = cy + 2
            if gyhi > nym1:
                gyhi = nym1

            conflict = False
            if dense:
                for gx in range(gxlo, gxhi + 1):
                    row = gx * ny
                    for gy in range(gylo, gyhi + 1):
                        j = heads[row + gy]
                        while j >= 0:
                            dx = x - xs[j]
                            dy = y - ys[j]
                            if dx * dx + dy * dy < r2:
                                conflict = True
                                break
                            j = nxt[j]
                        if conflict:
                            break
                    if conflict:
                        break
            else:
                get = heads.get
                for gx in range(gxlo, gxhi + 1):
                    row = gx * ny
                    for gy in range(gylo, gyhi + 1):
                        j = get(row + gy, -1)
                        while j >= 0:
                            dx = x - xs[j]
                            dy = y - ys[j]
                            if dx * dx + dy * dy < r2:
                                conflict = True
                                break
                            j = nxt[j]
                        if conflict:
                            break
                    if conflict:
                        break
            if conflict:
                continue

            j = count
            count += 1
            xs.append(x)
            ys.append(y)
            key = cx * ny + cy
            nxt.append(heads[key] if dense else heads.get(key, -1))
            heads[key] = j
            active.append(j)
            accepted = True
            break

        if accepted:
            active.pop(ai)
            active.append(idx)
        else:
            active.pop(ai)

    return list(zip(xs, ys))


def _scan_field_min(field, seed, width, height):
    """用独立随机源预扫描密度场，估计全场最小半径。

    估计值只用于初始网格尺度：主循环中若出现更小的合法半径，仍会按
    阈值触发重建，因此预扫描不改变任何正确性保证。场返回非法值时
    抛 ValueError。
    """
    probe_rng = random.Random(_probe_seed(seed))
    random_ = probe_rng.random
    r_min = float("inf")
    for _ in range(_PROBE_COUNT):
        r = _coerce_radius(field(random_() * width, random_() * height))
        if r < r_min:
            r_min = r
    return r_min


def _sample_variable(field, seed, width, height, k):
    """半径随位置变化的变密度路径。"""
    rng = random.Random(seed)  # 每次采样独立随机源，保证并发独立
    rng_random = rng.random
    rng_randrange = rng.randrange
    sqrt = math.sqrt
    cos = math.cos
    sin = math.sin
    isfinite = math.isfinite

    # 预扫描估计全场最小半径（独立随机源，不消费主 rng，序列仍可复现）
    r_min = _scan_field_min(field, seed, width, height)

    x0 = rng_random() * width
    y0 = rng_random() * height
    r_first = _coerce_radius(field(x0, y0))
    if r_first < r_min:
        r_min = r_first

    cell = r_min / _SQRT2
    nx = max(1, int(math.ceil(width / cell)))
    ny = max(1, int(math.ceil(height / cell)))
    dense = nx * ny <= _CELL_LIMIT
    heads = [-1] * (nx * ny) if dense else {}
    nxt = [-1]
    nxm1 = nx - 1
    nym1 = ny - 1

    xs = [x0]               # 点 i 坐标为 xs[i] / ys[i]，半径为 radii[i]
    ys = [y0]
    radii = [r_first]
    gx0 = int(x0 / cell)
    gy0 = int(y0 / cell)
    if gx0 > nxm1:
        gx0 = nxm1
    if gy0 > nym1:
        gy0 = nym1
    heads[gx0 * ny + gy0] = 0
    active = [0]
    count = 1
    rebuild_limit = r_min * _REBUILD_FACTOR

    while active:
        ai = rng_randrange(len(active))
        idx = active[ai]
        px = xs[idx]
        py = ys[idx]
        pr = radii[idx]
        accepted = False

        for _ in range(k):
            # 环形均匀采样 [pr, 2pr)
            ang = rng_random() * _TWO_PI
            rr = pr * sqrt(_SQRT3 * rng_random() + 1.0)
            x = px + rr * cos(ang)
            y = py + rr * sin(ang)
            if x < 0.0 or x > width or y < 0.0 or y > height:
                continue

            rc = field(x, y)
            # 内联校验：一次原生比较 + 一次 isfinite，避免每候选的调用开销
            try:
                positive = rc > 0.0
            except TypeError:
                raise ValueError("密度场返回的半径必须为正的有限数") from None
            if not positive or not isfinite(rc):
                raise ValueError("密度场返回的半径必须为正的有限数")
            if type(rc) is not float:
                rc = float(rc)

            # 出现明显更小半径时按新尺度重建网格；未达阈值时沿用当前
            # 网格，邻居搜索范围按 rc 自适应，完备性不受影响
            if rc < rebuild_limit:
                cell = rc / _SQRT2
                nx = max(1, int(math.ceil(width / cell)))
                ny = max(1, int(math.ceil(height / cell)))
                dense = dense and nx * ny <= _CELL_LIMIT
                heads = [-1] * (nx * ny) if dense else {}
                nxm1 = nx - 1
                nym1 = ny - 1
                for j in range(count):
                    gx = int(xs[j] / cell)
                    gy = int(ys[j] / cell)
                    if gx > nxm1:
                        gx = nxm1
                    if gy > nym1:
                        gy = nym1
                    key = gx * ny + gy
                    nxt[j] = heads[key] if dense else heads.get(key, -1)
                    heads[key] = j
                r_min = rc
                rebuild_limit = r_min * _REBUILD_FACTOR

            # 候选格坐标用精确除法（仅 2 次），避免逆元乘法的舍入风险
            cx = int(x / cell)
            cy = int(y / cell)
            if cx > nxm1:
                cx = nxm1
            if cy > nym1:
                cy = nym1
            # ceil(rc / cell)；用除法+余数保证向上取整不被浮点误差破坏
            m = int(rc / cell)
            if m * cell < rc:
                m += 1
            gxlo = cx - m
            if gxlo < 0:
                gxlo = 0
            gxhi = cx + m
            if gxhi > nxm1:
                gxhi = nxm1
            gylo = cy - m
            if gylo < 0:
                gylo = 0
            gyhi = cy + m
            if gyhi > nym1:
                gyhi = nym1
            rc2 = rc * rc
            conflict = False

            if dense:
                for gx in range(gxlo, gxhi + 1):
                    row = gx * ny
                    for gy in range(gylo, gyhi + 1):
                        j = heads[row + gy]
                        while j >= 0:
                            dx = x - xs[j]
                            dy = y - ys[j]
                            if dx * dx + dy * dy < rc2:
                                conflict = True
                                break
                            j = nxt[j]
                        if conflict:
                            break
                    if conflict:
                        break
            else:
                get = heads.get
                for gx in range(gxlo, gxhi + 1):
                    row = gx * ny
                    for gy in range(gylo, gyhi + 1):
                        j = get(row + gy, -1)
                        while j >= 0:
                            dx = x - xs[j]
                            dy = y - ys[j]
                            if dx * dx + dy * dy < rc2:
                                conflict = True
                                break
                            j = nxt[j]
                        if conflict:
                            break
                    if conflict:
                        break
            if conflict:
                continue

            j = count
            count += 1
            xs.append(x)
            ys.append(y)
            radii.append(rc)
            key = cx * ny + cy
            nxt.append(heads[key] if dense else heads.get(key, -1))
            heads[key] = j
            active.append(j)
            accepted = True
            break

        if accepted:
            active.pop(ai)
            active.append(idx)
        else:
            active.pop(ai)

    return list(zip(xs, ys))
