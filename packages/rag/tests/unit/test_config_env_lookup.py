# Test values are synthetic fixtures for software behavior.
"""回归：容器内 config.py 层级不足会让 parents[2] 抛 IndexError，导致服务启动失败。"""

from pathlib import Path

from config import locate_repo_env


def test_locate_repo_env_finds_ancestor_env(tmp_path):
    (tmp_path / ".env").write_text("LLM_API_KEY=test", encoding="utf-8")
    nested = tmp_path / "packages" / "rag"
    nested.mkdir(parents=True)
    module_file = nested / "config.py"
    module_file.write_text("", encoding="utf-8")

    found = locate_repo_env(module_file)
    assert found == (tmp_path / ".env").resolve()


def test_locate_repo_env_does_not_raise_on_shallow_layout(tmp_path):
    """模拟容器里的 /app/config.py：层级不足以取 parents[2]，必须优雅返回。"""
    module_file = tmp_path / "config.py"
    module_file.write_text("", encoding="utf-8")

    # 关键是不抛 IndexError；结果可能为 None 或某个祖先目录下的 .env
    result = locate_repo_env(module_file)
    assert result is None or isinstance(result, Path)


def test_config_module_still_loads_repo_env_locally():
    """本地布局下应能找到 <repo>/packages/rag 之上的 .env（若存在）。"""
    found = locate_repo_env(Path(__file__))
    if found is not None:
        assert found.name == ".env"
        # 找到的应是仓库根（其下含 packages/ 目录），而不是某个无关目录
        assert (found.parent / "packages").is_dir()
