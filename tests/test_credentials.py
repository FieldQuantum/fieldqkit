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
        _real_iter = pc._iter_config_candidates
        monkeypatch.setattr(
            pc, "_iter_config_candidates",
            lambda: [p for p in _real_iter() if str(p).startswith(str(tmp_path))],
        )
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


# ─────────────────────────────────────────────────────────────
#  Credential resolution: precedence, all platforms, boundaries
# ─────────────────────────────────────────────────────────────

# (getter, platform key, config section, env var) for all six platforms.
_PLATFORM_GETTERS = [
    (pc.get_quafu_api_token, "quafu", "quafu", "QUAFU_API_TOKEN"),
    (pc.get_tianyan_api_token, "tianyan", "tianyan", "TIANYAN_API_TOKEN"),
    (pc.get_guodun_api_token, "guodun", "guodun", "GUODUN_API_TOKEN"),
    (pc.get_tencent_api_token, "tencent", "tencent", "TENCENT_API_TOKEN"),
    (pc.get_origin_api_token, "origin", "origin", "ORIGIN_API_TOKEN"),
    (pc.get_fieldquantum_api_token, "fieldquantum", "fieldquantum", "FIELDQUANTUM_API_TOKEN"),
]

_ALL_ENV_VARS = [env for _, _, _, env in _PLATFORM_GETTERS]


def _no_config(monkeypatch):
    """Force credential lookup to behave as if no config file exists anywhere."""
    monkeypatch.setattr(pc, "_load_config", lambda *a, **k: {})


def _clear_all_token_env(monkeypatch):
    for env in _ALL_ENV_VARS:
        monkeypatch.delenv(env, raising=False)


class TestCredentialPrecedence:
    @pytest.mark.parametrize("getter,key,section,env", _PLATFORM_GETTERS)
    def test_env_var_fallback_all_platforms(self, monkeypatch, getter, key, section, env):
        """Every platform falls back to its env var when no config file exists."""
        _no_config(monkeypatch)
        _clear_all_token_env(monkeypatch)
        monkeypatch.setenv(env, f"{key}-env-token")
        assert getter() == f"{key}-env-token"

    @pytest.mark.parametrize("getter,key,section,env", _PLATFORM_GETTERS)
    def test_config_file_takes_priority_over_env(self, monkeypatch, getter, key, section, env):
        """Config-file value wins even when the env var is also set."""
        monkeypatch.setattr(
            pc, "_load_config",
            lambda *a, **k: {"credentials": {section: {"api_token": f"{key}-cfg"}}},
        )
        monkeypatch.setenv(env, f"{key}-env")
        assert getter() == f"{key}-cfg"

    @pytest.mark.parametrize("getter,key,section,env", _PLATFORM_GETTERS)
    def test_empty_config_token_falls_through_to_env(self, monkeypatch, getter, key, section, env):
        """An empty token in the config file is treated as absent (env wins)."""
        monkeypatch.setattr(
            pc, "_load_config",
            lambda *a, **k: {"credentials": {section: {"api_token": ""}}},
        )
        monkeypatch.setenv(env, f"{key}-env-after-empty")
        assert getter() == f"{key}-env-after-empty"

    @pytest.mark.parametrize("getter,key,section,env", _PLATFORM_GETTERS)
    def test_missing_credential_raises_helpful_error(self, monkeypatch, getter, key, section, env):
        """With neither config nor env, a ValueError naming the env var is raised."""
        _no_config(monkeypatch)
        _clear_all_token_env(monkeypatch)
        with pytest.raises(ValueError) as excinfo:
            getter()
        msg = str(excinfo.value)
        assert env in msg
        assert "fieldqkit-config-init" in msg

    def test_platform_key_is_case_sensitive(self, monkeypatch):
        """Credential map keys are lowercase; an uppercased key is unknown."""
        with pytest.raises(KeyError):
            pc._get_credential("QUAFU")

    def test_credential_map_covers_exactly_six_platforms(self):
        assert set(pc._CREDENTIAL_MAP) == {
            "quafu", "tianyan", "guodun", "tencent", "origin", "fieldquantum"
        }
        # Labels exist for every mapped platform (used in the error message).
        assert set(pc._PLATFORM_LABELS) == set(pc._CREDENTIAL_MAP)


class TestConfigFileDiscoveryBoundaries:
    def test_explicit_env_config_takes_priority(self, monkeypatch, tmp_path):
        """$QUANTUM_HW_CONFIG points at a valid file -> its tokens are used."""
        pytest.importorskip("yaml")
        cfg = tmp_path / "explicit.yaml"
        cfg.write_text(
            "credentials:\n  guodun:\n    api_token: explicit-token\n",
            encoding="utf-8",
        )
        monkeypatch.setenv("QUANTUM_HW_CONFIG", str(cfg))
        _clear_all_token_env(monkeypatch)
        pc._cached_config = None
        assert pc.get_guodun_api_token() == "explicit-token"

    def test_config_absent_uses_env(self, monkeypatch, tmp_path):
        """When the only candidate is an absent file, env var is the source."""
        # Point the explicit override at a non-existent path, and isolate
        # home/cwd to empty tmp dirs so no real config is discovered.
        missing = tmp_path / "does_not_exist.yaml"
        empty_home = tmp_path / "home"
        empty_cwd = tmp_path / "cwd"
        empty_home.mkdir()
        empty_cwd.mkdir()
        monkeypatch.setattr(pc.Path, "home", staticmethod(lambda: empty_home))
        monkeypatch.setattr(pc.Path, "cwd", staticmethod(lambda: empty_cwd))
        # Restrict candidates to just the (missing) explicit path so the
        # developer's real package-dir config can never be picked up.
        monkeypatch.setattr(pc, "_iter_config_candidates", lambda: [missing])
        _clear_all_token_env(monkeypatch)
        monkeypatch.setenv("ORIGIN_API_TOKEN", "origin-env")
        pc._cached_config = None
        assert pc.get_origin_api_token() == "origin-env"

    def test_malformed_yaml_handled_gracefully(self, monkeypatch, tmp_path):
        """A malformed config file is skipped (logged, not raised)."""
        pytest.importorskip("yaml")
        bad = tmp_path / "bad.yaml"
        bad.write_text("credentials: [unclosed", encoding="utf-8")
        # Only candidate is the malformed file -> _load_config returns {} and
        # never falls through to the developer's real config.
        monkeypatch.setattr(pc, "_iter_config_candidates", lambda: [bad])
        pc._cached_config = None
        assert pc._load_config() == {}

    def test_write_then_read_back_roundtrip(self, monkeypatch, tmp_path):
        """write_example_config produces a file _load_config can discover."""
        pytest.importorskip("yaml")
        target = tmp_path / "roundtrip.yaml"
        written = pc.write_example_config(target)
        assert written == target
        monkeypatch.setattr(pc, "_iter_config_candidates", lambda: [target])
        pc._cached_config = None
        cfg = pc._load_config()
        assert set(cfg["credentials"]) == set(pc._CREDENTIAL_MAP)
        # Template tokens are empty, so a getter on it falls back to ValueError
        # when no env var is set.
        _clear_all_token_env(monkeypatch)
        with pytest.raises(ValueError):
            pc.get_quafu_api_token()

    def test_write_example_config_default_path(self, monkeypatch, tmp_path):
        """With no path, write_example_config targets ~/.quantum_hw.yaml."""
        fake_home = tmp_path / "home"
        fake_home.mkdir()
        monkeypatch.setattr(pc.Path, "home", staticmethod(lambda: fake_home))
        written = pc.write_example_config()
        assert written == fake_home / ".quantum_hw.yaml"
        assert written.is_file()

    def test_force_overwrites_existing_via_write_example(self, tmp_path):
        target = tmp_path / "creds.yaml"
        pc.write_example_config(target)
        # FileExistsError without force.
        with pytest.raises(FileExistsError):
            pc.write_example_config(target)
        # Overwrites with force.
        target.write_text("garbage", encoding="utf-8")
        pc.write_example_config(target, force=True)
        assert "credentials:" in target.read_text(encoding="utf-8")
