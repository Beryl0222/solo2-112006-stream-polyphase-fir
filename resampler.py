"""直播字幕链路的流式 FIR 变采样与多轨时间交接引擎。

数值口径见《交接协议.md》。引擎按事件驱动:

- ``clock``   上游采样率映射(有理数), 高修订可覆盖未确认区间;
- ``chunk``   按轨编号的采集块, 输入编号连续, 全零块是真实静音而非断流;
- ``gap``     显式声明的断流区间, 与"块未到"的未知状态严格区分;
- ``safe``    安全区: 该输出点前的映射与已知观测不再修订, 缺报仍可补;
- ``preview`` 字幕预览: 可暂用值、已知系数部分和、缺口与等待原因;
- ``publish`` 档案交付: 仅推进双轨稳定且支撑已定的连续前缀, 断流点交付空档。

已交付区间不可再被修订, 可随时回取; 每轨热历史只保留 ``retention``
个输入编号跨度, 越界且不能安全修正时明确请求重送或拒绝。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from math import gcd
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

PHASES = (0, 1, 2, 3)


class _Gap:
    """档案空档标记: 非零滤波支撑内确认断流的输出点。"""

    _instance: Optional["_Gap"] = None

    def __new__(cls) -> "_Gap":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:
        return "GAP"

    def __bool__(self) -> bool:
        return False


GAP = _Gap()


def _floor(x: Fraction) -> int:
    return x.numerator // x.denominator


@dataclass
class Outcome:
    """状态变更事件的裁决: 是否接受、机器可读码与可读细节。"""

    accepted: bool
    code: str = "ok"
    detail: Dict[str, Any] = field(default_factory=dict)

    def __bool__(self) -> bool:
        return self.accepted


@dataclass(frozen=True)
class ClockMap:
    """一条 clock 区间: 输出半开区间 [start, end) 上的有理数输入坐标。"""

    revision: int
    start: int
    end: int
    num: int
    den: int
    stride: int

    @property
    def slope(self) -> Fraction:
        return Fraction(self.stride, self.den)

    @property
    def intercept(self) -> Fraction:
        return Fraction(self.num - self.start * self.stride, self.den)

    def coordinate(self, t: int) -> Fraction:
        return Fraction(self.num + (t - self.start) * self.stride, self.den)


@dataclass
class _Point:
    status: str  # ready / waiting / gap / unmapped
    coordinate: Optional[Fraction]
    value: Any
    partial: float
    missing: List[int]


@dataclass
class _Track:
    name: str
    clocks: List[ClockMap] = field(default_factory=list)
    samples: Dict[int, float] = field(default_factory=dict)
    gaps: set = field(default_factory=set)
    safe_before: int = 0
    evicted_below: Optional[int] = None
    evicted_above: Optional[int] = None


class Handoff:
    """流式变采样交接引擎。

    参数
    ----
    tracks:    轨名序列(如 ["host", "call"]), 会话开始即确定;
    offsets:   滤波偏移, 相对输入坐标向下取整;
    taps:      相位 -> 系数表, 相位为输入坐标分数部分 * 4, 必须覆盖 0..3;
    retention: 每轨热历史保留的输入编号跨度上限。
    """

    def __init__(
        self,
        tracks: Iterable[str],
        offsets: Sequence[int],
        taps: Dict[Any, Sequence[float]],
        retention: int,
        log_limit: Optional[int] = 10000,
    ) -> None:
        names = [str(name) for name in tracks]
        if not names:
            raise ValueError("至少需要一条轨")
        if len(set(names)) != len(names):
            raise ValueError("轨名不能重复")
        self.offsets = [self._check_int(o, "offsets") for o in offsets]
        if not self.offsets:
            raise ValueError("offsets 不能为空")
        self.taps: Dict[int, List[float]] = {}
        for key, coeffs in taps.items():
            if isinstance(key, str) and key.lstrip("-").isdigit():
                key = int(key)
            phase = self._check_int(key, "taps 相位")
            if phase not in PHASES:
                raise ValueError(f"非法相位 {key!r}")
            row = [self._check_number(c, "taps 系数") for c in coeffs]
            if len(row) != len(self.offsets):
                raise ValueError("taps 与 offsets 长度不一致")
            self.taps[phase] = row
        if sorted(self.taps) != list(PHASES):
            raise ValueError("taps 必须覆盖相位 0、1、2、3")
        self.retention = self._check_int(retention, "retention")
        if self.retention < 0:
            raise ValueError("retention 不能为负")
        nonzero = set()
        for row in self.taps.values():
            for off, coeff in zip(self.offsets, row):
                if coeff != 0.0:
                    nonzero.add(off)
        if not nonzero:
            raise ValueError("滤波器没有非零系数")
        self._nz_lo = min(nonzero)
        self._nz_hi = max(nonzero)
        self._log_limit = log_limit
        self._tracks = {name: _Track(name) for name in names}
        self.delivered = 0
        # 已交付档案: 协议要求可回取, 长期运行时下游应自行转存
        self.archive: Dict[str, List[Any]] = {name: [] for name in names}
        self.log: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # 基础校验

    @staticmethod
    def _check_int(value: Any, field_name: str) -> int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{field_name} 必须是整数, 收到 {value!r}")
        return value

    @staticmethod
    def _check_number(value: Any, field_name: str) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{field_name} 必须是数值, 收到 {value!r}")
        result = float(value)
        if result != result or result in (float("inf"), float("-inf")):
            raise ValueError(f"{field_name} 必须是有限数值, 收到 {value!r}")
        return result

    def _track(self, track: str) -> _Track:
        try:
            return self._tracks[track]
        except KeyError:
            raise KeyError(f"未知轨 {track!r}") from None

    def _record(self, event: Sequence[Any], outcome: Optional[Outcome] = None, **extra: Any) -> None:
        entry: Dict[str, Any] = {"event": list(event)}
        if outcome is not None:
            entry["accepted"] = outcome.accepted
            entry["code"] = outcome.code
            if outcome.detail:
                entry["detail"] = outcome.detail
        entry.update(extra)
        self.log.append(entry)
        if self._log_limit is not None and len(self.log) > self._log_limit:
            del self.log[: len(self.log) - self._log_limit]

    # ------------------------------------------------------------------
    # 映射解析与滤波支撑

    def _resolve(self, tr: _Track, t: int) -> Optional[ClockMap]:
        """输出点 t 的有效映射: 覆盖它的最高修订区间。"""
        best = None
        for cm in tr.clocks:
            if cm.start <= t < cm.end and (best is None or cm.revision > best.revision):
                best = cm
        return best

    def _segments(self, tr: _Track, lo: int, hi: int) -> List[Tuple[int, int]]:
        """[lo, hi) 内由 clock 区间端点切出的原子段, 段内解析结果不变。"""
        points = set()
        for cm in tr.clocks:
            a, b = max(cm.start, lo), min(cm.end, hi)
            if a < b:
                points.add(a)
                points.add(b)
        ordered = sorted(points)
        return [(a, b) for a, b in zip(ordered, ordered[1:])]

    def _phase(self, x: Fraction) -> int:
        frac = x - _floor(x)
        phase = frac * 4
        if phase.denominator != 1:
            raise RuntimeError("时钟分母未通过协议校验")
        return phase.numerator

    def _support(self, tr: _Track, t: int):
        """输出点 t 的 (映射, 输入坐标, 非零支撑 [(输入编号, 系数)]); 未映射时为 None。"""
        cm = self._resolve(tr, t)
        if cm is None:
            return None, None, None
        x = cm.coordinate(t)
        base = _floor(x)
        coeffs = self.taps[self._phase(x)]
        support = [
            (base + off, coeff)
            for off, coeff in zip(self.offsets, coeffs)
            if coeff != 0.0
        ]
        return cm, x, support

    def _point(self, tr: _Track, t: int) -> _Point:
        cm, x, support = self._support(tr, t)
        if cm is None:
            return _Point("unmapped", None, None, 0.0, [])
        partial = 0.0
        missing: List[int] = []
        gap_hit = False
        for index, coeff in support:
            if index in tr.gaps:
                gap_hit = True
            elif index in tr.samples:
                partial += coeff * tr.samples[index]
            else:
                missing.append(index)
        if gap_hit:
            return _Point("gap", x, GAP, partial, missing)
        if missing:
            return _Point("waiting", x, None, partial, missing)
        return _Point("ready", x, partial, partial, missing)

    # ------------------------------------------------------------------
    # 热历史窗口(retention)

    def _need_range(self, tr: _Track) -> Tuple[Optional[int], Optional[int]]:
        """未交付且已映射的输出点所需的输入编号包络 [lo, hi]; 无需求时为 (None, None)。"""
        lo = hi = None
        for a, b in self._segments(tr, self.delivered, 1 << 62):
            cm = self._resolve(tr, a)
            if cm is None:
                continue
            for t in (a, b - 1):
                floor = _floor(cm.coordinate(t))
                cand_lo, cand_hi = floor + self._nz_lo, floor + self._nz_hi
                lo = cand_lo if lo is None else min(lo, cand_lo)
                hi = cand_hi if hi is None else max(hi, cand_hi)
        return lo, hi

    def _evict(self, tr: _Track) -> None:
        """裁剪热历史: 丢弃需求包络之外的编号; 需求未知时窗口随流整体滑动。"""
        lo, hi = self._need_range(tr)
        kept = sorted(set(tr.samples) | set(tr.gaps))
        if not kept:
            return
        drop: set = set()
        if lo is not None:
            for index in kept:
                if index < lo or index > hi:
                    drop.add(index)
            # 需求包络内超窗的情形由 ingest 拒绝, 这里不丢需求内资料
        else:
            rest = kept
            while rest and rest[-1] - rest[0] > self.retention:
                drop.add(rest.pop(0))
        if not drop:
            return
        for index in drop:
            tr.samples.pop(index, None)
            tr.gaps.discard(index)
        low = [i for i in drop if lo is None or i < lo]
        high = [i for i in drop if lo is not None and i > hi]
        if low:
            boundary = max(low) + 1
            if tr.evicted_below is None or boundary > tr.evicted_below:
                tr.evicted_below = boundary
        if high:
            boundary = min(high) - 1
            if tr.evicted_above is None or boundary < tr.evicted_above:
                tr.evicted_above = boundary

    def _fits(self, tr: _Track, new: set) -> bool:
        """模拟接纳新编号后能否守住窗口: 需求包络内的资料一个也不能丢。"""
        lo, hi = self._need_range(tr)
        if lo is None:
            return True  # 需求未知时窗口可整体滑动, 总能容纳
        kept = {i for i in (set(tr.samples) | set(tr.gaps) | set(new)) if lo <= i <= hi}
        if not kept:
            return True
        return max(kept) - min(kept) <= self.retention

    def _resend(self, tr: _Track, missing: Iterable[int]) -> List[int]:
        """缺失编号中曾被热历史丢弃、必须请求上游补送的部分。"""
        out = []
        for index in missing:
            if tr.evicted_below is not None and index < tr.evicted_below:
                out.append(index)
            elif tr.evicted_above is not None and index > tr.evicted_above:
                out.append(index)
        return out

    def _frozen_input(self, tr: _Track, index: int) -> bool:
        """该输入编号是否落在安全区内某输出点的非零支撑内。"""
        # 快速排除: 不在安全区支撑的外包区间内
        outer = False
        for a, b in self._segments(tr, 0, tr.safe_before):
            cm = self._resolve(tr, a)
            if cm is None:
                continue
            fa = _floor(cm.coordinate(a))
            fb = _floor(cm.coordinate(b - 1))
            lo, hi = (fa, fb) if fa <= fb else (fb, fa)
            if lo + self._nz_lo <= index <= hi + self._nz_hi:
                outer = True
                break
        if not outer:
            return False
        # 精确核对: 逐点检查安全区内输出点的非零支撑
        for t in range(tr.safe_before):
            cm, _, support = self._support(tr, t)
            if cm is None:
                continue
            if any(i == index for i, _ in support):
                return True
        return False

    def _prune_clocks(self, tr: _Track) -> None:
        """丢弃处处被更高修订覆盖、不再参与解析的区间。"""
        kept = []
        for cm in tr.clocks:
            remaining = [(cm.start, cm.end)]
            for other in tr.clocks:
                if other.revision <= cm.revision:
                    continue
                nxt = []
                for a, b in remaining:
                    if other.end <= a or other.start >= b:
                        nxt.append((a, b))
                        continue
                    if a < other.start:
                        nxt.append((a, min(other.start, b)))
                    if other.end < b:
                        nxt.append((max(other.end, a), b))
                remaining = nxt
                if not remaining:
                    break
            if remaining:
                kept.append(cm)
        tr.clocks = kept

    @staticmethod
    def _merge_ranges(ranges: List[List[int]]) -> List[List[int]]:
        merged: List[List[int]] = []
        for a, b in ranges:
            if merged and merged[-1][1] == a:
                merged[-1][1] = b
            else:
                merged.append([a, b])
        return merged

    def _differs_ranges(self, tr: _Track, cm: ClockMap, lo: int, hi: int, include_unmapped: bool) -> List[List[int]]:
        """[lo, hi) 内现有解析与 cm 坐标不同的输出点区间。

        按原子段解析计算: 段内两坐标之差是仿射函数, 整数零点至多一个,
        因此无需逐点扫描, 长区间也能常量时间完成。
        """
        out: List[List[int]] = []
        cursor = lo
        for a, b in self._segments(tr, lo, hi):
            if include_unmapped and cursor < a:
                out.append([cursor, a])  # 尚无映射的缝隙
            old = self._resolve(tr, a)
            if old is None:
                if include_unmapped:
                    out.append([a, b])  # 段内整体无映射
            else:
                d_slope = old.slope - cm.slope
                d_intercept = old.intercept - cm.intercept
                if d_slope == 0:
                    if d_intercept != 0:
                        out.append([a, b])
                else:
                    zero = -d_intercept / d_slope  # 唯一可能相等的点
                    if zero.denominator == 1 and a <= zero.numerator < b:
                        point = zero.numerator
                        if a < point:
                            out.append([a, point])
                        if point + 1 < b:
                            out.append([point + 1, b])
                    else:
                        out.append([a, b])
            cursor = b
        if include_unmapped and cursor < hi:
            out.append([cursor, hi])
        return self._merge_ranges(out)

    # ------------------------------------------------------------------
    # 事件: clock / chunk / gap / safe

    def clock(
        self,
        track: str,
        revision: int,
        start: int,
        end: int,
        num: int,
        den: int,
        stride: int,
    ) -> Outcome:
        tr = self._track(track)
        event = ["clock", track, revision, start, end, num, den, stride]
        revision, start, end, num, den, stride = (
            self._check_int(v, "clock 参数")
            for v in (revision, start, end, num, den, stride)
        )

        def reject(code: str, **detail: Any) -> Outcome:
            outcome = Outcome(False, code, detail)
            self._record(event, outcome)
            return outcome

        if start < 0 or end <= start:
            return reject("bad-range", start=start, end=end)
        if den <= 0:
            return reject("bad-denominator", denominator=den)
        if tr.clocks and revision <= max(cm.revision for cm in tr.clocks):
            return reject("stale-revision", revision=revision)
        cm = ClockMap(revision, start, end, num, den, stride)
        # 协议只允许分母可约至 1、2 或 4: 检查所有可达剩余类
        step = gcd(stride, den)
        for j in range(den // step):
            residue = (num + j * step) % den
            reduced = den // gcd(residue, den)
            if reduced not in (1, 2, 4):
                return reject("bad-denominator", denominator=den, reduced=reduced)
        # 冻结区校验: 依照最终支撑(逐点输入坐标)而非仅起点判断是否影响已交付输出
        frozen_to = max(tr.safe_before, self.delivered)
        if start < frozen_to:
            conflicts = self._differs_ranges(tr, cm, start, min(end, frozen_to), include_unmapped=False)
            if conflicts:
                t = conflicts[0][0]
                old = self._resolve(tr, t)
                code = "delivered-conflict" if t < self.delivered else "safe-conflict"
                return reject(
                    code,
                    t=t,
                    old=str(old.coordinate(t)) if old is not None else None,
                    new=str(cm.coordinate(t)),
                )
        # 实际改变的输出区间(未确认部分), 供"为何只影响后半段"查询
        changed = self._differs_ranges(tr, cm, start, end, include_unmapped=True)
        tr.clocks.append(cm)
        self._prune_clocks(tr)
        self._evict(tr)
        outcome = Outcome(True, "ok", {
            "changed": changed,
            "frozen": [start, min(end, frozen_to)] if start < frozen_to else None,
        })
        self._record(event, outcome)
        return outcome

    def chunk(self, track: str, start: int, values: Sequence[float]) -> Outcome:
        tr = self._track(track)
        values = list(values)  # 先物化, 避免生成器被消费两次
        event = ["chunk", track, start, len(values)]  # 日志只存摘要, 不复制样本
        start = self._check_int(start, "chunk 起点")
        values = [self._check_number(v, "chunk 样本") for v in values]
        if not values:
            raise ValueError("chunk 不能为空")

        def reject(code: str, **detail: Any) -> Outcome:
            outcome = Outcome(False, code, detail)
            self._record(event, outcome)
            return outcome

        corrections = []
        for k, value in enumerate(values):
            index = start + k
            if index in tr.gaps or (index in tr.samples and tr.samples[index] != value):
                if self._frozen_input(tr, index):
                    return reject("frozen-conflict", index=index)
                corrections.append(index)
        new = set(range(start, start + len(values)))
        if not self._fits(tr, new):
            lo, hi = self._need_range(tr)
            return reject(
                "retention-exceeded",
                need=[lo, hi],
                evicted_below=tr.evicted_below,
                evicted_above=tr.evicted_above,
            )
        for k, value in enumerate(values):
            index = start + k
            tr.gaps.discard(index)
            tr.samples[index] = value
        self._evict(tr)
        outcome = Outcome(True, "ok", {
            "stored": [start, start + len(values)],
            "corrected": corrections,
            "evicted_below": tr.evicted_below,
        })
        self._record(event, outcome)
        return outcome

    def gap(self, track: str, start: int, end: int) -> Outcome:
        tr = self._track(track)
        event = ["gap", track, start, end]
        start = self._check_int(start, "gap 起点")
        end = self._check_int(end, "gap 终点")
        if end <= start:
            raise ValueError("gap 区间不能为空")

        def reject(code: str, **detail: Any) -> Outcome:
            outcome = Outcome(False, code, detail)
            self._record(event, outcome)
            return outcome

        corrections = []
        for index in range(start, end):
            if index in tr.samples:
                if self._frozen_input(tr, index):
                    return reject("frozen-conflict", index=index)
                corrections.append(index)
        new = set(range(start, end))
        if not self._fits(tr, new):
            lo, hi = self._need_range(tr)
            return reject(
                "retention-exceeded",
                need=[lo, hi],
                evicted_below=tr.evicted_below,
                evicted_above=tr.evicted_above,
            )
        for index in range(start, end):
            tr.samples.pop(index, None)
            tr.gaps.add(index)
        self._evict(tr)
        outcome = Outcome(True, "ok", {
            "declared": [start, end],
            "corrected": corrections,
            "evicted_below": tr.evicted_below,
        })
        self._record(event, outcome)
        return outcome

    def safe(self, track: str, output_before: int) -> Outcome:
        tr = self._track(track)
        event = ["safe", track, output_before]
        output_before = self._check_int(output_before, "safe 边界")
        if output_before < 0:
            raise ValueError("safe 边界不能为负")
        if output_before < tr.safe_before:
            outcome = Outcome(False, "safe-regression", {"safe_before": tr.safe_before})
            self._record(event, outcome)
            return outcome
        tr.safe_before = output_before
        outcome = Outcome(True, "ok", {"safe_before": output_before})
        self._record(event, outcome)
        return outcome

    # ------------------------------------------------------------------
    # 查询与交付

    def preview(self, start: int, end: int) -> Dict[str, List[Dict[str, Any]]]:
        """字幕预览: 已交付点给档案值, 其余给可暂用值/部分和/缺口与状态。"""
        start = self._check_int(start, "preview 起点")
        end = self._check_int(end, "preview 终点")
        if start < 0 or end < start:
            raise ValueError("preview 区间非法")
        out: Dict[str, List[Dict[str, Any]]] = {}
        for name, tr in self._tracks.items():
            rows = []
            for t in range(start, end):
                if t < self.delivered:
                    rows.append({
                        "t": t,
                        "status": "delivered",
                        "stable": True,
                        "value": self.archive[name][t],
                        "partial": None,
                        "missing": [],
                        "resend": [],
                        "coordinate": None,
                    })
                    continue
                pt = self._point(tr, t)
                rows.append({
                    "t": t,
                    "status": pt.status,
                    "stable": t < tr.safe_before,
                    "value": pt.value if pt.status in ("ready", "gap") else None,
                    "partial": pt.partial,
                    "missing": pt.missing,
                    "resend": self._resend(tr, pt.missing),
                    "coordinate": str(pt.coordinate) if pt.coordinate is not None else None,
                })
            out[name] = rows
        return out

    def publish(self) -> Dict[str, Any]:
        """交付从未交付点起连续、双轨均稳定且支撑已定的前缀。"""
        start = self.delivered
        values: Dict[str, List[Any]] = {name: [] for name in self._tracks}
        t = start
        while True:
            row: Dict[str, Any] = {}
            for name, tr in self._tracks.items():
                if t >= tr.safe_before:
                    break
                pt = self._point(tr, t)
                if pt.status not in ("ready", "gap"):
                    break
                row[name] = pt.value
            else:
                for name, value in row.items():
                    self.archive[name].append(value)
                    values[name].append(value)
                t += 1
                continue
            break
        self.delivered = t
        for tr in self._tracks.values():
            self._evict(tr)
        result = {
            "delivered": [start, t],
            "values": values,
            "blockers": self.blockers(),
        }
        self._record(["publish"], None, result=result)
        return result

    def published(self, start: int, end: int) -> Dict[str, List[Any]]:
        """回取已交付时间段; 范围含未交付点时拒绝。"""
        start = self._check_int(start, "回取起点")
        end = self._check_int(end, "回取终点")
        if start < 0 or end < start or end > self.delivered:
            raise ValueError(
                f"回取范围 [{start}, {end}) 包含未交付点 (已交付 {self.delivered})"
            )
        return {name: list(self.archive[name][start:end]) for name in self._tracks}

    def blockers(self) -> Dict[str, Dict[str, Any]]:
        """当前交付前沿为何停滞: 每轨在该输出点的状态(含等待另一轨的证据)。"""
        t = self.delivered
        out: Dict[str, Dict[str, Any]] = {}
        for name, tr in self._tracks.items():
            if t >= tr.safe_before:
                out[name] = {"reason": "unsafe", "safe_before": tr.safe_before}
                continue
            pt = self._point(tr, t)
            if pt.status == "waiting":
                out[name] = {
                    "reason": "waiting",
                    "missing": pt.missing,
                    "resend": self._resend(tr, pt.missing),
                }
            elif pt.status == "unmapped":
                out[name] = {"reason": "unmapped"}
            else:
                out[name] = {"reason": pt.status}
        return out

    def why_waiting(self, track: str, t: int) -> Dict[str, Any]:
        """查询某轨某输出点当前为何不能进入档案(或已交付)。"""
        tr = self._track(track)
        t = self._check_int(t, "输出点")
        if t < 0:
            raise ValueError("输出点不能为负")
        if t < self.delivered:
            return {"track": track, "t": t, "state": "delivered"}
        pt = self._point(tr, t)
        return {
            "track": track,
            "t": t,
            "state": pt.status,
            "stable": t < tr.safe_before,
            "missing": pt.missing,
            "resend": self._resend(tr, pt.missing),
            "coordinate": str(pt.coordinate) if pt.coordinate is not None else None,
        }

    def track_state(self, track: str) -> Dict[str, Any]:
        """热历史与安全区快照, 供窗口与状态核查。"""
        tr = self._track(track)
        kept = sorted(set(tr.samples) | set(tr.gaps))
        return {
            "safe_before": tr.safe_before,
            "evicted_below": tr.evicted_below,
            "evicted_above": tr.evicted_above,
            "kept_span": (kept[-1] - kept[0]) if kept else None,
            "kept_count": len(kept),
            "clocks": [(cm.revision, cm.start, cm.end) for cm in tr.clocks],
        }

    def decisions(self) -> List[Dict[str, Any]]:
        """决策日志: 每次接受/拒绝及理由, 供"为何被拒/只影响后半段"查询。"""
        return list(self.log)
