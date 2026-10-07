# -*- coding: utf-8 -*-
"""pytest：检索侧离线链路（快照字段与写读往返 / 阈值离线标定）。

覆盖：
  1) retrieval_pass.hit_record() 的字段完整性（chunk_id/doc/section_path/page/score）；
  2) 快照 写 -> 读 往返（eval_retrieval.load_snapshot）；
  3) 阈值离线标定：正/负类分组、可分离区间与建议值、候选权衡表数字。

运行：.venv_rag311\\Scripts\\python.exe -m pytest tests/ -q
"""
import json
import shutil
from pathlib import Path
from uuid import uuid4

import pytest

# 用例临时目录：放工作区内（系统临时区在受限环境下不可写），用例结束整目录清理，不留痕。
ROOT = Path(__file__).resolve().parents[1]
WORK_DIR = ROOT / ".pytest_work"


@pytest.fixture
def workdir():
    d = WORK_DIR / uuid4().hex[:8]
    d.mkdir(parents=True, exist_ok=True)
    yield d
    shutil.rmtree(WORK_DIR, ignore_errors=True)


def _hit(cid, doc, score, section="S", page=(1, 1)):
    return {"id": cid, "score": score, "text": "片段正文（不落盘）",
            "meta": {"doc": doc, "section_path": section,
                     "start_page": page[0], "end_page": page[1]}}


# ---------- 1. 快照记录字段 ----------

def test_hit_record_fields():
    from scripts.retrieval_pass import hit_record
    r = hit_record(_hit("chunk_000001", "D", 0.9, "D / 活动", (2, 3)))
    assert set(r) == {"chunk_id", "doc", "section_path", "page", "score"}
    assert r["chunk_id"] == "chunk_000001" and r["doc"] == "D"
    assert r["section_path"] == "D / 活动" and r["page"] == [2, 3]
    assert "text" not in r                       # 正文不落盘


def test_hit_record_no_page():
    from scripts.retrieval_pass import hit_record
    r = hit_record(_hit("chunk_000002", "D", 0.5, page=(0, 0)))
    assert r["page"] is None                     # 无页码写 None，不写 [0, 0]


# ---------- 2. 快照写读往返 ----------

def test_snapshot_write_read_roundtrip(workdir, monkeypatch):
    import scripts.eval_retrieval as er
    from scripts.retrieval_pass import write_snapshot

    rows = [
        {"qid": "S01", "set": "simple", "answerable": True, "route": "vector_only", "k": 5,
         "hits": [{"chunk_id": "chunk_000001", "doc": "D1", "section_path": "D1", "page": [1, 1],
                   "score": 0.8123}]},
        {"qid": "S28", "set": "simple", "answerable": False, "route": "vector_only", "k": 5,
         "hits": []},
    ]
    write_snapshot(workdir / "retrieval_vector_only.jsonl", rows)

    monkeypatch.setattr(er, "EVAL_DIR", workdir)
    snap = er.load_snapshot("vector_only")
    assert set(snap) == {"S01", "S28"}
    assert snap["S01"] == rows[0]
    assert snap["S28"]["hits"] == []             # 无命中也是合法快照
    assert snap["S01"]["hits"][0]["score"] == 0.8123


def test_snapshot_missing_returns_none(workdir, monkeypatch, capsys):
    import scripts.eval_retrieval as er
    monkeypatch.setattr(er, "EVAL_DIR", workdir)
    assert er.load_snapshot("hybrid_rerank") is None
    assert "找不到快照" in capsys.readouterr().out


# ---------- 3. 阈值离线标定 ----------

def _write_threshold_snapshot(workdir, pos, neg):
    rows = []
    for i, s in enumerate(pos):
        rows.append({"qid": f"A{i}", "set": "simple", "answerable": True,
                     "route": "vector_only", "k": 5,
                     "hits": [{"chunk_id": "c", "doc": "D", "section_path": "D",
                               "page": [1, 1], "score": s}]})
    for i, s in enumerate(neg):
        rows.append({"qid": f"N{i}", "set": "simple", "answerable": False,
                     "route": "vector_only", "k": 5,
                     "hits": [{"chunk_id": "c", "doc": "D", "section_path": "D",
                               "page": [1, 1], "score": s}]})
    p = workdir / "retrieval_vector_only.jsonl"
    p.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")


def test_threshold_overlap_tradeoff_table(workdir, monkeypatch):
    """重叠分布：候选权衡表取决策面相异的最小阈值。"""
    import scripts.calibrate_threshold as ct
    monkeypatch.setattr(ct, "EVAL_DIR", workdir)
    _write_threshold_snapshot(workdir, pos=[0.9, 0.8, 0.6], neg=[0.7, 0.3])

    result = ct.build("vector_only", qa_by_id={})
    assert result["pos"] == {"n": 3, "min": 0.6, "p25": 0.6, "median": 0.8,
                             "p75": 0.9, "max": 0.9}
    assert result["neg"]["max"] == 0.7
    assert result["separable"] is False                   # 正类下沿 0.6 <= 负类上沿 0.7
    assert result["suggested_min_score"] is None
    assert result["feasible_window"] is None
    # 每个「漏拦」水平只留误杀最少的一档；同一档给出等价阈值区间
    assert [(c["threshold"], c["false_reject"], c["leak"]) for c in result["candidates"]] == [
        (0.3, 0, 2), (0.6, 0, 1), (0.8, 1, 0)]
    assert [c["interval"] for c in result["candidates"]] == [
        [None, 0.3], [0.3, 0.6], [0.7, 0.8]]
    assert len(result["per_question"]) == 5


def test_threshold_separable_window_and_midpoint(workdir, monkeypatch):
    """可分离：给出可行区间与建议中点。"""
    import scripts.calibrate_threshold as ct
    monkeypatch.setattr(ct, "EVAL_DIR", workdir)
    _write_threshold_snapshot(workdir, pos=[0.7, 0.9], neg=[0.2, 0.5])

    result = ct.build("vector_only", qa_by_id={})
    assert result["separable"] is True
    assert result["feasible_window"] == [0.5, 0.7]
    assert result["suggested_min_score"] == 0.6           # (0.5 + 0.7) / 2
    # 建议值确实 0 误杀 0 漏拦
    assert sum(1 for x in [0.7, 0.9] if x < 0.6) == 0
    assert sum(1 for x in [0.2, 0.5] if x >= 0.6) == 0

