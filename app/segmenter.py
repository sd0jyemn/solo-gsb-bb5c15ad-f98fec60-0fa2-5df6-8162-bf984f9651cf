"""字幕断句核心:全局动态规划选择分段,使加权总代价最低。

代价项(均可加权):
  - 镜头切点偏差: 分段边界偏离最近镜头切点的距离(归一化), 以及分段跨越切点的惩罚
  - 行长不均:     两行长度差(归一化)
  - 不自然停顿:   断点处词间间隔过小(归一化)

硬约束:
  - 每段最多 2 行, 每行不超过 max_chars_per_line 个字符
  - 阅读速度(字符/秒)不超过 max_cps
  - 每段时长在 [min_duration, max_duration] 内
  - 不可拆短语(phrase)作为原子单元, 不得被分段或换行拆开
  - 边界约束(boundaries): 指定词间隙必须断开(must_break)或禁止断开(no_break);
    不可拆短语内部的词间隙不得设置边界
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Config:
    max_chars_per_line: int = 42
    max_cps: float = 20.0
    min_duration: float = 0.8
    max_duration: float = 7.0
    pause_threshold: float = 0.3      # 词间间隔小于该值视为不自然断点
    shot_dev_cap: float = 0.5         # 镜头切点偏差归一化上限(秒)
    crossing_penalty: float = 1.0     # 分段跨越一个镜头切点的惩罚(归一化)
    w_shot: float = 1.0
    w_imbalance: float = 1.0
    w_pause: float = 1.0


@dataclass
class Unit:
    """原子单元: 单个词或一个不可拆短语。"""
    word_indices: list[int]
    text: str
    start: float
    end: float


def build_units(words: list[dict], phrases: list[list[int]]) -> list[Unit]:
    """把词序列与不可拆短语(词下标区间)合并为原子单元序列。"""
    n = len(words)
    owner: dict[int, list[int]] = {}
    for ph in phrases:
        if not ph:
            continue
        if ph != list(range(ph[0], ph[-1] + 1)):
            raise ValueError(f"短语下标不连续: {ph}")
        for i in ph:
            if i < 0 or i >= n:
                raise ValueError(f"短语下标越界: {i}")
            if i in owner:
                raise ValueError(f"短语重叠于词 {i}")
        for i in ph:
            owner[i] = ph

    units: list[Unit] = []
    i = 0
    while i < n:
        if i in owner:
            ph = owner[i]
            units.append(Unit(
                word_indices=ph,
                text=" ".join(words[k]["text"] for k in ph),
                start=words[ph[0]]["start"],
                end=words[ph[-1]]["end"],
            ))
            i = ph[-1] + 1
        else:
            units.append(Unit(
                word_indices=[i],
                text=words[i]["text"],
                start=words[i]["start"],
                end=words[i]["end"],
            ))
            i += 1
    return units


def resolve_boundaries(boundaries: dict | None, n_words: int,
                       units: list[Unit]) -> tuple[set[int], set[int]]:
    """校验词间隙边界约束, 映射为单元下标集合 (must, forbid)。

    边界 i 表示词 i-1 与词 i 之间的间隙(1 <= i <= n_words-1);
    映射后为单元下标 u, 表示单元 u-1 与单元 u 之间的间隙。
    """
    boundaries = boundaries or {}
    must = set(boundaries.get("must_break") or [])
    forbid = set(boundaries.get("no_break") or [])
    for i in sorted(must | forbid):
        if not isinstance(i, int) or isinstance(i, bool) or i < 1 or i >= n_words:
            raise ValueError(f"边界下标越界: {i}(应在 1..{n_words - 1})")
    both = sorted(must & forbid)
    if both:
        raise ValueError(f"同一边界不能既必须断开又禁止断开: {both}")
    inner = set()
    for u in units:
        inner.update(u.word_indices[1:])  # 单元(短语)内部的词下标
    for i in sorted(must | forbid):
        if i in inner:
            raise ValueError(f"不可拆短语内部不得设置边界: 词间隙 {i}")
    unit_of = {u.word_indices[0]: k for k, u in enumerate(units)}
    return {unit_of[i] for i in must}, {unit_of[i] for i in forbid}


def layout_segment(units: list[Unit], a: int, b: int, max_chars: int):
    """尝试把 units[a:b] 排成至多两行, 返回 (lines, imbalance) 或 None。

    单元不可拆行; 枚举所有换行位置, 取行长差最小者。
    """
    texts = [u.text for u in units[a:b]]
    if any(len(t) > max_chars for t in texts):
        return None

    def join(ts):
        return " ".join(ts)

    best = None
    # 单行
    one = join(texts)
    if len(one) <= max_chars:
        best = ([one], 0.0)
    # 两行
    for k in range(1, len(texts)):
        l1, l2 = join(texts[:k]), join(texts[k:])
        if len(l1) <= max_chars and len(l2) <= max_chars:
            imb = abs(len(l1) - len(l2)) / max_chars
            if best is None or imb < best[1]:
                best = ([l1, l2], imb)
    return best


def segment_violations(units: list[Unit], a: int, b: int, cfg: Config) -> list[str]:
    """列出段 units[a..b](闭区间)违反的硬约束(空列表表示可行)。"""
    seg = units[a:b + 1]
    dur = seg[-1].end - seg[0].start
    chars = sum(len(u.text) for u in seg) + len(seg) - 1
    v = []
    if dur > cfg.max_duration + 1e-9:
        v.append(f"时长 {dur:.2f}s 超过上限 {cfg.max_duration}s")
    if dur < cfg.min_duration - 1e-9:
        v.append(f"时长 {dur:.2f}s 低于下限 {cfg.min_duration}s")
    if dur > 1e-9 and chars / dur > cfg.max_cps + 1e-9:
        v.append(f"阅读速度 {chars / dur:.1f} 字/秒 超过上限 {cfg.max_cps}")
    if layout_segment(units, a, b + 1, cfg.max_chars_per_line) is None:
        v.append(f"无法在 2 行 x {cfg.max_chars_per_line} 字符内排下")
    return v


def boundary_shot_cost(t: float, shots: list[float], cfg: Config) -> float:
    if not shots:
        return 0.0
    d = min(abs(t - s) for s in shots)
    return min(d, cfg.shot_dev_cap) / cfg.shot_dev_cap


def crossing_count(start: float, end: float, shots: list[float]) -> int:
    return sum(1 for s in shots if start + 1e-9 < s < end - 1e-9)


def pause_cost(gap: float, cfg: Config) -> float:
    if gap >= cfg.pause_threshold:
        return 0.0
    return min(1.0, (cfg.pause_threshold - gap) / cfg.pause_threshold)


def solve(words: list[dict], phrases: list[list[int]], shots: list[float],
          cfg: Config, boundaries: dict | None = None) -> dict:
    if not words:
        return {"ok": True, "total_cost": 0.0, "segments": []}
    for i, w in enumerate(words):
        if w["end"] < w["start"]:
            raise ValueError(f"词 {i} 的结束时间早于开始时间")
        if i and w["start"] < words[i - 1]["start"] - 1e-9:
            raise ValueError(f"词 {i} 的开始时间早于前一个词")

    units = build_units(words, phrases)
    n = len(units)
    shots = sorted(shots)
    must, forbid = resolve_boundaries(boundaries, len(words), units)

    # seg_cost[a][b]: 段 units[a:b+1] 的可行性与代价分解
    INF = float("inf")
    feasible = [[False] * n for _ in range(n)]
    cost_parts = [[None] * n for _ in range(n)]
    for a in range(n):
        for b in range(a, n):
            if units[b].end - units[a].start > cfg.max_duration + 1e-9:
                break  # 再延长只会更超时长
            if segment_violations(units, a, b, cfg):
                continue
            lay = layout_segment(units, a, b, cfg.max_chars_per_line)
            lines, imb = lay
            start, end = units[a].start, units[b].end
            dur = end - start
            chars = sum(len(u.text) for u in units[a:b + 1]) + (b - a)
            cross = crossing_count(start, end, shots)
            feasible[a][b] = True
            cost_parts[a][b] = {
                "lines": lines,
                "imbalance": imb,
                "crossing": cross,
                "duration": dur,
                "cps": chars / dur if dur > 1e-9 else 0.0,
            }

    # next_must[i]: 严格大于 i 的最近必须断开边界(单元下标), 不存在则为 n+1
    next_must = [n + 1] * (n + 1)
    for i in range(n - 1, -1, -1):
        next_must[i] = i + 1 if (i + 1) in must else next_must[i + 1]

    def edge_ok(i: int, j: int) -> bool:
        """段 units[i:j] 是否满足边界硬约束(时长/行宽等可行性另判)。"""
        if i > 0 and i in forbid:
            return False  # 禁止断开处不得起新段
        return next_must[i] >= j  # 段不得跨越必须断开边界

    def edge_cost(i: int, j: int) -> float:
        p = cost_parts[i][j - 1]
        c_shot = p["crossing"] * cfg.crossing_penalty
        c_pause = 0.0
        if i > 0:  # 段首边界(非全文开头): 切点偏差 + 停顿
            t = units[i].start
            c_shot += boundary_shot_cost(t, shots, cfg)
            c_pause = pause_cost(units[i].start - units[i - 1].end, cfg)
        return (cfg.w_shot * c_shot
                + cfg.w_imbalance * p["imbalance"]
                + cfg.w_pause * c_pause)

    # DP: dp[j] = 前 j 个单元的最小总代价
    dp = [INF] * (n + 1)
    dp[0] = 0.0
    prev = [-1] * (n + 1)
    for j in range(1, n + 1):
        for i in range(j):
            if dp[i] == INF or not feasible[i][j - 1] or not edge_ok(i, j):
                continue
            total = dp[i] + edge_cost(i, j)
            if total < dp[j]:
                dp[j] = total
                prev[j] = i

    if dp[n] == INF:
        if must:
            # 反向 DP: rev[i] = 从单元 i 起(在 i 处断开)完成剩余单元的最小代价
            rev = [INF] * (n + 1)
            rev[n] = 0.0
            for i in range(n - 1, -1, -1):
                if not edge_ok(i, i + 1):
                    continue  # i 处禁止断开, 不能在此起段
                for j in range(i + 1, n + 1):
                    if rev[j] == INF or not feasible[i][j - 1] or not edge_ok(i, j):
                        continue
                    c = edge_cost(i, j) + rev[j]
                    if c < rev[i]:
                        rev[i] = c
            for m in sorted(must):
                if dp[m] == INF or rev[m] == INF:
                    return _boundary_conflict(units, cfg, m, dp, rev, next_must,
                                              forbid)
        return _conflict(units, words, dp, cfg)

    # 回溯分段
    bounds = []
    j = n
    while j > 0:
        i = prev[j]
        bounds.append((i, j - 1))
        j = i
    bounds.reverse()

    segments = []
    for a, b in bounds:
        p = cost_parts[a][b]
        start, end = units[a].start, units[b].end
        c_shot = p["crossing"] * cfg.crossing_penalty
        c_pause = 0.0
        if a > 0:
            c_shot += boundary_shot_cost(start, shots, cfg)
            c_pause = pause_cost(units[a].start - units[a - 1].end, cfg)
        c_shot_w = cfg.w_shot * c_shot
        c_imb_w = cfg.w_imbalance * p["imbalance"]
        c_pause_w = cfg.w_pause * c_pause
        segments.append({
            "word_range": [units[a].word_indices[0], units[b].word_indices[-1]],
            "start": round(start, 3),
            "end": round(end, 3),
            "duration": round(p["duration"], 3),
            "cps": round(p["cps"], 1),
            "lines": p["lines"],
            "cost": {
                "shot": round(c_shot_w, 4),
                "imbalance": round(c_imb_w, 4),
                "pause": round(c_pause_w, 4),
                "total": round(c_shot_w + c_imb_w + c_pause_w, 4),
            },
        })
    return {"ok": True, "total_cost": round(dp[n], 4), "segments": segments}


def _boundary_conflict(units: list[Unit], cfg: Config, m: int,
                       dp: list[float], rev: list[float],
                       next_must: list[int], forbid: set[int]) -> dict:
    """必须断开边界 m(单元下标)不可行: 报告失败侧(左/右)最近尝试段的违例原因。"""
    INF = float("inf")
    n = len(units)
    if dp[m] == INF:
        # 左侧不可行: 最近的可达前驱 a 到 m 的末段必然违反硬约束
        cands = [i for i in range(m)
                 if dp[i] != INF and (i == 0 or i not in forbid)
                 and next_must[i] >= m]
        lo, hi = (max(cands) if cands else m - 1), m - 1
        side = "left"
    else:
        # 右侧不可行: m 到最近的可达后继 b 的首段必然违反硬约束
        cands = [j for j in range(m + 1, n + 1)
                 if rev[j] != INF and next_must[m] >= j]
        b = min(cands) if cands else min(next_must[m], n)
        lo, hi = m, b - 1
        side = "right"
    reasons = segment_violations(units, lo, hi, cfg)
    if not reasons:
        reasons = ["该侧在边界约束下无法完成可行分段"]
    return {
        "ok": False,
        "conflict": {
            "boundary": units[m].word_indices[0],
            "side": side,
            "word_range": [units[lo].word_indices[0], units[hi].word_indices[-1]],
            "start": round(units[lo].start, 3),
            "end": round(units[hi].end, 3),
            "text": " ".join(u.text for u in units[lo:hi + 1]),
            "reasons": reasons,
        },
    }


def _conflict(units: list[Unit], words: list[dict], dp: list[float],
              cfg: Config) -> dict:
    """无可行方案时, 定位首个无法被任何可行分段覆盖的单元区间。"""
    INF = float("inf")
    n = len(units)
    j_star = next(j for j in range(1, n + 1) if dp[j] == INF)
    # j_star-1 是首个无法结束的单元; 收集以它为结尾/起点的最短冲突区间的原因
    a = j_star - 1
    reasons = segment_violations(units, a, a, cfg)
    if not reasons:
        # 单元单独可行, 但与前方可达边界拼接均失败: 汇总最近一段的违例
        for i in range(a - 1, -1, -1):
            if dp[i] != INF:
                reasons = segment_violations(units, i, a, cfg)
                break
        if not reasons:
            reasons = ["与相邻单元组合后无可行布局"]
    wi = units[a].word_indices
    return {
        "ok": False,
        "conflict": {
            "word_range": [wi[0], wi[-1]],
            "start": round(units[a].start, 3),
            "end": round(units[a].end, 3),
            "text": units[a].text,
            "reasons": reasons,
        },
    }
