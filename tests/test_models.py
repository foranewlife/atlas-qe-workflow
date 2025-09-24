import time
from aqflow.core.models import BoardModel, BoardMeta, TaskModel
import pytest


def test_board_model_roundtrip():
    now = time.time()
    meta = BoardMeta(
        run_id="run_001",
        root="/tmp",
        resources_file="/tmp/resources.yaml",
        tool="eos",
        args=["aqflow", "eos"],
        start_time=now,
        last_update=now,
    )
    t = TaskModel(
        id="t1",
        name="qe test",
        type="qe",
        workdir="/tmp/w",
        status="queued",
    )
    board = BoardModel(meta=meta, tasks={t.id: t})
    raw = board.model_dump()
    # Validate again
    board2 = BoardModel.model_validate(raw)
    assert board2.meta.run_id == "run_001"
    assert "t1" in board2.tasks


def test_task_model_minimal():
    t = TaskModel(
        id="x",
        name="atlas",
        type="atlas",
        workdir="/tmp/x",
        status="queued",
    )
    assert t.status == "queued"


def test_task_validation_times_and_status():
    # running must have start_time
    with pytest.raises(Exception):
        TaskModel(id="id-1", name="n", type="qe", workdir="/w", status="running")
    # succeeded must have end_time & exit_code
    with pytest.raises(Exception):
        TaskModel(id="id-2", name="n", type="atlas", workdir="/w", status="succeeded", start_time=1.0)
    # ok case
    t = TaskModel(id="ok-1", name="n", type="qe", workdir="/w", status="succeeded", start_time=1.0, end_time=2.0, exit_code=0)
    assert t.exit_code == 0
