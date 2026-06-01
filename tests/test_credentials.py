"""Tests for credential configuration: the config-scaffold helper / CLI and
the per-user config discovery path used by pip-installed users."""

from pathlib import Path

import pytest

import fieldqkit.api.platform_credentials as pc


@pytest.fixture(autouse=True)
def _reset_config_cache():
    """Keep the module-level config cache from leaking across tests."""
    pc._cached_config = None
    yield
    pc._cached_config = None


# ─────────────────────────────────────────────────────────────
#  write_example_config / init_config
# ─────────────────────────────────────────────────────────────

class TestWriteExampleConfig:
    def test_writes_template_with_all_platforms(self, tmp_path):
        target = tmp_path / "creds.yaml"
        written = pc.write_example_config(target)
        assert written == target
        text = target.read_text(encoding="utf-8")
        assert "credentials:" in text
        for platform in ("quafu", "tianyan", "guodun", "tencent", "origin", "fieldquantum"):
            assert f"{platform}:" in text

    def test_template_is_valid_yaml_with_empty_tokens(self, tmp_path):
        yaml = pytest.importorskip("yaml")
        target = pc.write_example_config(tmp_path / "creds.yaml")
        data = yaml.safe_load(target.read_text(encoding="utf-8"))
        assert set(data["credentials"]) == {
            "quafu", "tianyan", "guodun", "tencent", "origin", "fieldquantum"
        }
        # All tokens start empty.
        assert all(v["api_token"] == "" for v in data["credentials"].values())

    def test_refuses_overwrite_without_force(self, tmp_path):
        target = tmp_path / "creds.yaml"
        pc.write_example_config(target)
        with pytest.raises(FileExistsError):
            pc.write_example_config(target)

    def test_force_overwrites(self, tmp_path):
        target = tmp_path / "creds.yaml"
        pc.write_example_config(target)
        target.write_text("clobbered", encoding="utf-8")
        pc.write_example_config(target, force=True)
        assert "credentials:" in target.read_text(encoding="utf-8")

    def test_creates_parent_directories(self, tmp_path):
        target = tmp_path / "nested" / "dir" / "creds.yaml"
        pc.write_example_config(target)
        assert target.is_file()

    def test_init_config_is_alias(self):
        assert pc.init_config is pc.write_example_config

    def test_default_user_config_path(self):
        assert pc.default_user_config_path() == Path.home() / ".quantum_hw.yaml"


# ─────────────────────────────────────────────────────────────
#  fieldqkit-config-init CLI
# ─────────────────────────────────────────────────────────────

class TestConfigInitCli:
    def test_cli_writes_file(self, tmp_path, capsys):
        target = tmp_path / "out.yaml"
        rc = pc._config_init_cli(["--path", str(target)])
        assert rc == 0
        assert target.is_file()
        assert "Wrote credentials template" in capsys.readouterr().out

    def test_cli_refuses_existing(self, tmp_path, capsys):
        target = tmp_path / "out.yaml"
        assert pc._config_init_cli(["--path", str(target)]) == 0
        rc = pc._config_init_cli(["--path", str(target)])
        assert rc == 1
        assert "already exists" in capsys.readouterr().err

    def test_cli_force(self, tmp_path):
        target = tmp_path / "out.yaml"
        assert pc._config_init_cli(["--path", str(target)]) == 0
        assert pc._config_init_cli(["--path", str(target), "--force"]) == 0


# ─────────────────────────────────────────────────────────────
#  Per-user config discovery (pip users)
# ─────────────────────────────────────────────────────────────

class TestUserConfigDiscovery:
    def test_user_paths_listed(self, monkeypatch, tmp_path):
        monkeypatch.setattr(pc.Path, "home", staticmethod(lambda: tmp_path))
        paths = pc._user_config_paths()
        assert paths[0] == tmp_path / ".quantum_hw.yaml"
        assert paths[1] == tmp_path / ".config" / "fieldqkit" / "credentials.yaml"

    def test_candidates_include_home(self, monkeypatch, tmp_path):
        monkeypatch.setattr(pc.Path, "home", staticmethod(lambda: tmp_path))
        candidates = pc._iter_config_candidates()
        assert (tmp_path / ".quantum_hw.yaml").resolve() in candidates

    def test_home_config_is_discovered(self, monkeypatch, tmp_path):
        pytest.importorskip("yaml")
        fake_home = tmp_path / "home"
        fake_cwd = tmp_path / "cwd"
        fake_home.mkdir()
        fake_cwd.mkdir()

        # Isolate from the developer's real cwd/home and any env tokens.
        monkeypatch.setattr(pc.Path, "home", staticmethod(lambda: fake_home))
        monkeypatch.setattr(pc.Path, "cwd", staticmethod(lambda: fake_cwd))
        monkeypatch.delenv("QUANTUM_HW_CONFIG", raising=False)
        for env in ("QUAFU_API_TOKEN", "TIANYAN_API_TOKEN", "GUODUN_API_TOKEN",
                    "TENCENT_API_TOKEN", "ORIGIN_API_TOKEN", "FIELDQUANTUM_API_TOKEN"):
            monkeypatch.delenv(env, raising=False)

        (fake_home / ".quantum_hw.yaml").write_text(
            "credentials:\n  quafu:\n    api_token: home-token\n", encoding="utf-8"
        )
        pc._cached_config = None
        assert pc.get_quafu_api_token() == "home-token"

    def test_env_var_used_when_no_config_file(self, monkeypatch):
        # Simulate "no config file found anywhere" so the env-var fallback is
        # exercised in isolation (independent of any developer .quantum_hw.yaml).
        monkeypatch.setattr(pc, "_load_config", lambda *a, **k: {})
        monkeypatch.setenv("TIANYAN_API_TOKEN", "env-token")
        assert pc.get_tianyan_api_token() == "env-token"
