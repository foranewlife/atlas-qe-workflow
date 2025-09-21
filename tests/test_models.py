import time
from aqflow.core.models import BoardModel, BoardMeta, TaskModel


def test_board_model_roundtrip():
    now = time.time()
    meta = BoardMeta(
        run_id="run_001",
        root="/tmp",
        resources_file="config/resources.yaml",
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
