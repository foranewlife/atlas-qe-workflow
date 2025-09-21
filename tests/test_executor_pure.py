from aqflow.core.executor import Executor


def test_choose_resource_capacity(tmp_path):
    ex = Executor(resources_yaml=tmp_path / "res.yaml", board_path=tmp_path / "aqflow" / "board.json", run_meta={"tool": "test"})
    resources = [
        {"name": "r1", "type": "local", "cores": 4, "software": {"qe": {"path": "/bin/echo", "cores": 2}}},
        {"name": "r2", "type": "local", "cores": 2, "software": {"qe": {"path": "/bin/echo", "cores": 4}}},
    ]
    chosen, req = ex._choose_resource("qe", resources, running={})
    assert chosen["name"] == "r1"
    assert req == 2


def test_update_status_states(tmp_path):
    ex = Executor(resources_yaml=tmp_path / "res.yaml", board_path=tmp_path / "aqflow" / "board.json", run_meta={"tool": "test"})
    t = {"status": "running"}
    ex._update_status(t, 0)
    assert t["status"] == "succeeded"
    ex._update_status(t, 124)
    assert t["status"] == "timeout"
    ex._update_status(t, 1)
    assert t["status"] == "failed"


def test_should_exit(tmp_path):
    ex = Executor(resources_yaml=tmp_path / "res.yaml", board_path=tmp_path / "aqflow" / "board.json", run_meta={"tool": "test"})
    ex.board["tasks"] = {
        "a": {"status": "succeeded"},
        "b": {"status": "failed"},
    }
    assert ex._should_exit() is True
    ex.board["tasks"]["c"] = {"status": "running"}
    assert ex._should_exit() is False


def test_plan_pull_outputs(tmp_path):
    ex = Executor(resources_yaml=tmp_path / "res.yaml", board_path=tmp_path / "aqflow" / "board.json", run_meta={"tool": "test"})
    # Mock a remote running record
    run = type("R", (), {})()
    run.resource = {"type": "remote", "transfer": {"pull_all": False}, "user": "u", "host": "h"}
    run.remote_dir = "/remote/jobdir"
    task = {"workdir": "/local/work"}
    cmds = ex._plan_pull_outputs(run, task)
    assert len(cmds) == 1 and "scp" in cmds[0]
    # pull_all=True branch
    run.resource["transfer"] = {"pull_all": True}
    cmds2 = ex._plan_pull_outputs(run, task)
    assert len(cmds2) == 1 and "-r" in cmds2[0]
