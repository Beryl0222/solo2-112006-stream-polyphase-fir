from pathlib import Path
import unittest

import numpy as np

from read_capture import read_capture
from replay import build_engine, replay
from resampler import GAP, Handoff


ROOT = Path(__file__).resolve().parents[1]
CAPTURE = ROOT / "samples" / "capture.json"

OFFSETS = [-1, 0, 1, 2]
TAPS = {
    0: [0, 1, 0, 0],
    1: [-0.125, 1, 0.25, -0.125],
    2: [-0.125, 0.625, 0.625, -0.125],
    3: [-0.125, 0.25, 1, -0.125],
}


def make_engine(tracks=("a", "b"), retention=8):
    return Handoff(tracks=tracks, offsets=OFFSETS, taps=TAPS, retention=retention)


def identity_clock(engine, track, end, revision=0):
    """x = t 的恒等映射, 覆盖 [0, end)。"""
    return engine.clock(track, revision, 0, end, 0, 4, 4)


def assert_values(case, actual, expected):
    """按协议口径 atol=rtol=1e-7 比较, GAP 单独处理。"""
    case.assertEqual(len(actual), len(expected))
    for got, want in zip(actual, expected):
        if want is GAP:
            case.assertIs(got, GAP)
        else:
            case.assertTrue(
                np.isclose(got, want, atol=1e-7, rtol=1e-7),
                f"{got} != {want}",
            )


class CaptureReplayTests(unittest.TestCase):
    """按接收顺序回放合成采集, 核对发布、预览、拒绝与窗口。"""

    @classmethod
    def setUpClass(cls):
        cls.capture = read_capture(CAPTURE)
        cls.engine, cls.transcript = replay(cls.capture)

    def test_first_publish_delivers_shared_prefix(self):
        result = self.transcript[7]["result"]  # 第 8 个事件是首次 publish
        self.assertEqual(result["delivered"], [0, 2])
        assert_values(self, result["values"]["host"], [0, 2])
        assert_values(self, result["values"]["call"], [2, -0.5])
        # call 轨安全区只到 2, host 已就绪也只能等另一轨
        self.assertEqual(result["blockers"]["host"]["reason"], "ready")
        self.assertEqual(result["blockers"]["call"]["reason"], "unsafe")

    def test_second_publish_marks_gap(self):
        result = self.transcript[13]["result"]
        self.assertEqual(result["delivered"], [2, 6])
        assert_values(self, result["values"]["host"], [6, 3, 4, 2])
        assert_values(self, result["values"]["call"], [0, 6.25, 8.625, GAP])

    def test_revision_touching_delivered_is_rejected(self):
        entry = self.transcript[14]
        self.assertEqual(entry["event"][0], "clock")
        self.assertFalse(entry["accepted"])
        self.assertEqual(entry["code"], "delivered-conflict")
        self.assertEqual(entry["detail"]["t"], 1)
        self.assertEqual(entry["detail"]["old"], "3/4")
        self.assertEqual(entry["detail"]["new"], "0")

    def test_final_preview_states(self):
        result = self.transcript[16]["result"]
        host, call = result["host"], result["call"]
        # host 新映射指向从未送达的负编号: 只能等待并请求补送, 不给假值
        self.assertEqual(host[0]["status"], "waiting")
        self.assertIsNone(host[0]["value"])
        self.assertEqual(host[0]["missing"], [-4])
        self.assertEqual(host[0]["resend"], [-4])
        self.assertEqual(host[1]["missing"], [-3])
        # call 非零支撑内有已声明断流: 预览标记空档
        self.assertEqual(call[0]["status"], "gap")
        self.assertIs(call[0]["value"], GAP)
        self.assertTrue(np.isclose(call[0]["partial"], 1.125, atol=1e-7, rtol=1e-7))
        self.assertEqual(call[1]["status"], "gap")

    def test_preview_partial_sums_before_publish(self):
        result = self.transcript[6]["result"]  # 首个 preview 事件(0 基)
        call = result["call"]
        self.assertEqual(call[2]["status"], "waiting")
        self.assertTrue(np.isclose(call[2]["partial"], 2.875, atol=1e-7, rtol=1e-7))
        self.assertEqual(call[2]["missing"], [3])
        self.assertEqual(call[4]["missing"], [3])
        self.assertFalse(call[2]["stable"])  # 超出安全区, 只是可暂用值
        host = result["host"]
        self.assertEqual(host[6]["status"], "ready")
        self.assertFalse(host[6]["stable"])

    def test_archive_immutable_and_retrievable(self):
        engine = self.engine
        self.assertEqual(engine.delivered, 6)
        archive = engine.published(0, 6)
        assert_values(self, archive["host"], [0, 2, 6, 3, 4, 2])
        assert_values(self, archive["call"], [2, -0.5, 0, 6.25, 8.625, GAP])
        # 已交付区间可回取; 含未交付点的范围被拒
        with self.assertRaises(ValueError):
            engine.published(0, 7)
        # 已交付点的预览命中档案, 不再受后续状态影响
        preview = engine.preview(0, 2)
        self.assertEqual(preview["host"][0]["status"], "delivered")
        self.assertTrue(np.isclose(preview["host"][1]["value"], 2, atol=1e-7, rtol=1e-7))

    def test_window_never_exceeds_retention(self):
        capture = self.capture
        engine = build_engine(capture)
        for event in capture["events"]:
            kind = event[0]
            if kind == "clock":
                engine.clock(*event[1:])
            elif kind == "chunk":
                engine.chunk(event[1], event[2], event[3])
            elif kind == "gap":
                engine.gap(event[1], event[2], event[3])
            elif kind == "safe":
                engine.safe(event[1], event[2])
            elif kind == "publish":
                engine.publish()
            for name in ("host", "call"):
                span = engine.track_state(name)["kept_span"]
                if span is not None:
                    self.assertLessEqual(span, capture["retention"])

    def test_rejection_is_queryable(self):
        decisions = self.engine.decisions()
        rejected = [d for d in decisions if d.get("accepted") is False]
        self.assertEqual(len(rejected), 1)
        self.assertEqual(rejected[0]["code"], "delivered-conflict")
        self.assertEqual(rejected[0]["event"][0], "clock")
        # 事后可逐点查询等待原因
        state = self.engine.why_waiting("host", 6)
        self.assertEqual(state["state"], "waiting")
        self.assertEqual(state["missing"], [-4])
        self.assertEqual(self.engine.why_waiting("host", 0)["state"], "delivered")


class ChunkSizeInvarianceTests(unittest.TestCase):
    """改变采集块大小不应改变有效的重采样结果。"""

    def resplit(self, events, size):
        out = []
        for event in events:
            if event[0] == "chunk":
                track, start, values = event[1], event[2], event[3]
                for offset in range(0, len(values), size):
                    piece = values[offset:offset + size]
                    out.append(["chunk", track, start + offset, piece])
            else:
                out.append(list(event))
        return out

    def replay_events(self, events):
        capture = dict(read_capture(CAPTURE))
        capture["events"] = events
        engine = build_engine(capture)
        for event in events:
            kind = event[0]
            if kind == "clock":
                engine.clock(*event[1:])
            elif kind == "chunk":
                engine.chunk(event[1], event[2], event[3])
            elif kind == "gap":
                engine.gap(event[1], event[2], event[3])
            elif kind == "safe":
                engine.safe(event[1], event[2])
            elif kind == "publish":
                engine.publish()
        return engine

    def test_singleton_and_triplet_chunks_match_original(self):
        events = read_capture(CAPTURE)["events"]
        base = self.replay_events([list(e) for e in events])
        for size in (1, 3):
            other = self.replay_events(self.resplit(events, size))
            self.assertEqual(other.delivered, base.delivered)
            for track in ("host", "call"):
                assert_values(
                    self,
                    other.published(0, other.delivered)[track],
                    base.published(0, base.delivered)[track],
                )
                for got, want in zip(other.preview(6, 8)[track], base.preview(6, 8)[track]):
                    self.assertEqual(got["status"], want["status"])
                    self.assertEqual(got["missing"], want["missing"])
                    if want["value"] is GAP:
                        self.assertIs(got["value"], GAP)
                    elif want["value"] is not None:
                        self.assertTrue(
                            np.isclose(got["value"], want["value"], atol=1e-7, rtol=1e-7)
                        )


class ClockSwitchTests(unittest.TestCase):
    """速率切换落在块中间: 输出整数连续, 不丢点也不多出一点。"""

    def test_switch_keeps_output_points_continuous(self):
        engine = make_engine(retention=32)
        for track in ("a", "b"):
            # [0,4): x = 2t; [4,8): x = 2t+1 —— 切换点取该点版本
            self.assertTrue(engine.clock(track, 0, 0, 4, 0, 1, 2))
            self.assertTrue(engine.clock(track, 1, 4, 8, 9, 1, 2))
            engine.chunk(track, 0, list(range(17)))
            self.assertTrue(engine.safe(track, 8))
        result = engine.publish()
        self.assertEqual(result["delivered"], [0, 8])
        expected = [0, 2, 4, 6, 9, 11, 13, 15]  # 每个输出点恰好一次
        for track in ("a", "b"):
            assert_values(self, result["values"][track], expected)
            self.assertEqual(len(engine.archive[track]), 8)

    def test_unmapped_points_wait_instead_of_skipping(self):
        engine = make_engine()
        for track in ("a", "b"):
            identity_clock(engine, track, 2)
            engine.chunk(track, 0, [1, 2])
            engine.safe(track, 4)  # 安全区越过未映射区
        result = engine.publish()
        self.assertEqual(result["delivered"], [0, 2])  # 映射外不丢点也不多出
        for track in ("a", "b"):
            self.assertEqual(result["blockers"][track]["reason"], "unmapped")


class RevisionRuleTests(unittest.TestCase):
    """时钟修订: 可改未确认预览, 不可改已交付; 按最终支撑判断。"""

    def test_correction_changes_undelivered_preview(self):
        engine = make_engine()
        identity_clock(engine, "a", 4)
        identity_clock(engine, "b", 4)
        engine.chunk("a", 0, [1, 2, 3, 4])
        engine.chunk("b", 0, [5, 6, 7, 8])
        before = engine.preview(2, 3)["a"][0]["value"]
        self.assertTrue(np.isclose(before, 3, atol=1e-7, rtol=1e-7))
        # 未确认区间的修订: x = t + 1
        outcome = engine.clock("a", 1, 0, 4, 4, 4, 4)
        self.assertTrue(outcome)
        after = engine.preview(2, 3)["a"][0]["value"]
        self.assertTrue(np.isclose(after, 4, atol=1e-7, rtol=1e-7))

    def test_identical_mapping_on_delivered_part_is_accepted(self):
        engine = make_engine()
        identity_clock(engine, "a", 6)
        identity_clock(engine, "b", 6)
        engine.chunk("a", 0, [1, 2, 3, 4, 5, 6])
        engine.chunk("b", 0, [1, 2, 3, 4, 5, 6])
        engine.safe("a", 2)
        engine.safe("b", 2)
        engine.publish()
        self.assertEqual(engine.delivered, 2)
        # 区间起点在已交付范围内, 但最终支撑未变: 不能仅凭起点拒绝
        outcome = engine.clock("a", 1, 1, 4, 4, 4, 4)  # x = t, 与原映射一致
        self.assertTrue(outcome)
        self.assertEqual(outcome.detail["changed"], [])
        self.assertEqual(outcome.detail["frozen"], [1, 2])

    def test_revision_only_affects_unconfirmed_tail(self):
        engine = make_engine()
        identity_clock(engine, "a", 4)
        engine.safe("a", 4)
        # [2,4) 与既有映射一致(冻结区不变), [4,8) 是新映射: 只影响后半段
        outcome = engine.clock("a", 1, 2, 8, 8, 4, 4)  # x = t
        self.assertTrue(outcome)
        self.assertEqual(outcome.detail["changed"], [[4, 8]])
        self.assertEqual(outcome.detail["frozen"], [2, 4])

    def test_crossing_lines_judged_by_final_support(self):
        engine = make_engine()
        identity_clock(engine, "a", 8)  # x = t
        engine.safe("a", 1)             # 只有 t=0 被冻结
        # 新线 x = t/2 与旧线恰在 t=0 交于同一坐标: 该点最终支撑不变, 不能仅凭起点拒绝
        outcome = engine.clock("a", 1, 0, 8, 0, 8, 4)
        self.assertTrue(outcome)
        self.assertEqual(outcome.detail["changed"], [[1, 8]])
        self.assertEqual(outcome.detail["frozen"], [0, 1])

    def test_changed_ranges_skip_coincident_points(self):
        # 采集里 call 的 rev1 与 rev0 在 t=3 坐标同为 9/4: 该点支撑不变
        engine = build_engine(read_capture(CAPTURE))
        engine.clock("call", 0, 0, 8, 0, 4, 3)
        outcome = engine.clock("call", 1, 2, 8, 5, 4, 4)
        self.assertTrue(outcome)
        self.assertEqual(outcome.detail["changed"], [[2, 3], [4, 8]])

    def test_revision_inside_safe_region_is_rejected(self):
        engine = make_engine()
        identity_clock(engine, "a", 6)
        engine.safe("a", 4)
        # 未交付但已冻结: 改变 [2,6) 的映射被拒
        outcome = engine.clock("a", 1, 2, 6, 4, 4, 4)  # x = t + 1
        self.assertFalse(outcome)
        self.assertEqual(outcome.code, "safe-conflict")
        self.assertEqual(outcome.detail["t"], 2)

    def test_stale_revision_and_bad_denominator(self):
        engine = make_engine()
        identity_clock(engine, "a", 4)
        self.assertEqual(engine.clock("a", 0, 0, 4, 0, 4, 4).code, "stale-revision")
        self.assertEqual(engine.clock("a", 1, 0, 4, 0, 3, 1).code, "bad-denominator")
        # 分母可约至 2: 合法
        self.assertTrue(engine.clock("a", 1, 0, 4, 0, 8, 4))

    def test_safe_regression_is_rejected(self):
        engine = make_engine()
        self.assertTrue(engine.safe("a", 4))
        self.assertEqual(engine.safe("a", 2).code, "safe-regression")
        self.assertTrue(engine.safe("a", 4))  # 同值是幂等的

    def test_chunk_accepts_one_pass_iterables(self):
        engine = make_engine()
        identity_clock(engine, "a", 2)
        outcome = engine.chunk("a", 0, iter([1, 2]))  # 生成器只能消费一次
        self.assertTrue(outcome)
        value = engine.preview(1, 2)["a"][0]["value"]
        self.assertTrue(np.isclose(value, 2, atol=1e-7, rtol=1e-7))

    def test_decision_log_is_bounded(self):
        engine = Handoff(tracks=["a"], offsets=OFFSETS, taps=TAPS, retention=8, log_limit=3)
        for _ in range(5):
            engine.safe("a", 0)
        self.assertEqual(len(engine.decisions()), 3)


class ObservationStateTests(unittest.TestCase):
    """缺报、显式断流与真实静音是三种不同状态。"""

    def test_silence_gap_and_missing_are_distinct(self):
        engine = make_engine()
        identity_clock(engine, "a", 2)
        identity_clock(engine, "b", 2)
        engine.chunk("a", 0, [0, 0])  # 真实静音: 有数据的零
        engine.gap("b", 0, 1)         # 显式断流
        engine.safe("a", 2)
        engine.safe("b", 2)
        result = engine.publish()
        # t=0: a 交付真实 0, b 交付空档; t=1: b 缺报, 只能等待
        self.assertEqual(result["delivered"], [0, 1])
        assert_values(self, result["values"]["a"], [0.0])
        assert_values(self, result["values"]["b"], [GAP])
        self.assertEqual(result["blockers"]["a"]["reason"], "ready")
        self.assertEqual(result["blockers"]["b"]["reason"], "waiting")
        self.assertEqual(result["blockers"]["b"]["missing"], [1])
        # 缺报可补: 补送后另一轨才能推进
        engine.chunk("b", 1, [7])
        result = engine.publish()
        self.assertEqual(result["delivered"], [1, 2])
        assert_values(self, result["values"]["a"], [0.0])
        assert_values(self, result["values"]["b"], [7.0])
        self.assertEqual(engine.delivered, 2)

    def test_frozen_observation_cannot_be_revised(self):
        engine = make_engine()
        identity_clock(engine, "a", 4)
        engine.chunk("a", 0, [1, 2, 3, 4])
        engine.safe("a", 2)
        outcome = engine.chunk("a", 0, [9])  # 与已冻结观测矛盾
        self.assertFalse(outcome)
        self.assertEqual(outcome.code, "frozen-conflict")
        # 未冻结的观测可更正, 并改变未交付预览
        outcome = engine.chunk("a", 3, [30])
        self.assertTrue(outcome)
        self.assertEqual(outcome.detail["corrected"], [3])
        value = engine.preview(3, 4)["a"][0]["value"]
        self.assertTrue(np.isclose(value, 30, atol=1e-7, rtol=1e-7))

    def test_gap_over_known_sample_is_a_correction(self):
        engine = make_engine()
        identity_clock(engine, "a", 2)
        engine.chunk("a", 0, [5, 6])
        outcome = engine.gap("a", 1, 2)  # 未冻结: 实样更正为断流
        self.assertTrue(outcome)
        self.assertEqual(outcome.detail["corrected"], [1])
        self.assertEqual(engine.preview(1, 2)["a"][0]["status"], "gap")
        engine.safe("a", 2)
        outcome = engine.chunk("a", 1, [6])  # 冻结后不得翻案
        self.assertFalse(outcome)
        self.assertEqual(outcome.code, "frozen-conflict")


class RetentionTests(unittest.TestCase):
    """热历史只保留协议窗口; 越界且不能安全修正时请求重送或拒绝。"""

    def test_chunk_beyond_window_is_rejected(self):
        engine = make_engine(retention=4)
        identity_clock(engine, "a", 8)
        outcome = engine.chunk("a", 0, [1, 2, 3, 4, 5, 6])  # 跨度 5 > 4
        self.assertFalse(outcome)
        self.assertEqual(outcome.code, "retention-exceeded")
        self.assertTrue(engine.chunk("a", 0, [1, 2, 3, 4, 5]))  # 跨度 4 可行

    def test_evicted_dependency_becomes_resend_request(self):
        engine = make_engine(retention=8)
        identity_clock(engine, "a", 4)
        identity_clock(engine, "b", 4)
        engine.chunk("a", 0, [1, 2, 3, 4])
        engine.chunk("b", 0, [1, 1, 1, 1])
        engine.safe("a", 4)
        engine.safe("b", 4)
        engine.publish()
        self.assertEqual(engine.delivered, 4)
        # 新块把窗口推向高处, 低编号被裁出热历史
        self.assertTrue(engine.chunk("a", 4, [9, 9, 9, 9]))
        self.assertTrue(engine.chunk("a", 8, [9, 9, 9, 9]))
        self.assertEqual(engine.track_state("a")["evicted_below"], 3)
        # 新映射让较早送过的输入重新成为滤波支撑: 只能请求补送
        self.assertTrue(engine.safe("a", 8))
        self.assertTrue(engine.safe("b", 8))
        self.assertTrue(engine.clock("a", 1, 4, 8, 0, 1, 1))  # x = t - 4
        row = engine.preview(4, 5)["a"][0]
        self.assertEqual(row["status"], "waiting")
        self.assertIsNone(row["value"])  # 超窗且资料不足: 不给假值
        self.assertEqual(row["missing"], [0])
        self.assertEqual(row["resend"], [0])
        blockers = engine.blockers()
        self.assertEqual(blockers["a"]["reason"], "waiting")
        self.assertEqual(blockers["a"]["resend"], [0])
        self.assertEqual(blockers["b"]["reason"], "unmapped")
        # 上游按请求补送后恢复可交付
        self.assertTrue(engine.clock("b", 1, 4, 8, 0, 1, 1))
        engine.chunk("b", 4, [2, 2, 2, 2])
        self.assertTrue(engine.chunk("a", 0, [1]))
        engine.chunk("b", 0, [1])
        result = engine.publish()
        self.assertEqual(result["delivered"], [4, 5])
        assert_values(self, result["values"]["a"], [1.0])
        assert_values(self, result["values"]["b"], [1.0])

    def test_window_slides_without_loading_whole_stream(self):
        engine = make_engine(retention=4)
        identity_clock(engine, "a", 8)
        identity_clock(engine, "b", 8)
        # 分小块持续交接, 全程不整段载入
        for start in range(0, 8, 2):
            self.assertTrue(engine.chunk("a", start, [start + 1, start + 2]))
            self.assertTrue(engine.chunk("b", start, [start + 1, start + 2]))
            engine.safe("a", start + 2)
            engine.safe("b", start + 2)
            engine.publish()
            for track in ("a", "b"):
                span = engine.track_state(track)["kept_span"]
                if span is not None:
                    self.assertLessEqual(span, 4)
        self.assertEqual(engine.delivered, 8)
        assert_values(self, engine.published(0, 8)["a"], [1, 2, 3, 4, 5, 6, 7, 8])


class QueryTests(unittest.TestCase):
    """决策可查询: 为何被拒、为何等待另一轨、为何只影响后半段。"""

    def test_decisions_log_records_reasons(self):
        engine = make_engine()
        identity_clock(engine, "a", 4)
        engine.safe("a", 4)
        engine.clock("a", 1, 2, 6, 4, 4, 4)  # 被拒: safe-conflict
        rejected = [d for d in engine.decisions() if d.get("accepted") is False]
        self.assertEqual(len(rejected), 1)
        self.assertEqual(rejected[0]["code"], "safe-conflict")
        self.assertIn("t", rejected[0]["detail"])

    def test_blockers_explain_waiting_for_the_other_track(self):
        engine = make_engine()
        identity_clock(engine, "a", 2)
        identity_clock(engine, "b", 2)
        engine.chunk("a", 0, [1, 2])
        engine.safe("a", 2)
        engine.safe("b", 2)
        result = engine.publish()
        self.assertEqual(result["delivered"], [0, 0])  # 一无所交
        self.assertEqual(result["blockers"]["a"]["reason"], "ready")
        self.assertEqual(result["blockers"]["b"]["reason"], "waiting")
        self.assertEqual(result["blockers"]["b"]["missing"], [0])


if __name__ == "__main__":
    unittest.main()
