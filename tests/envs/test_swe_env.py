"""Tests for SWEEnv fault-tolerance: wait-for-pool-ready and retry logic."""

import json
import os
import sys
import types
import unittest
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Minimal package stubs so swe.py can be imported without the full rllm stack
# or ML frameworks (torch, transformers, etc.).
# ---------------------------------------------------------------------------
_RLLM_PATH = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "rllm")
)

def _stub_rllm_pkg():
    """Inject lightweight stub modules to short-circuit heavy rllm imports."""
    # Top-level rllm package (skip its __init__.py)
    rllm = types.ModuleType("rllm")
    rllm.__path__ = [_RLLM_PATH]
    rllm.__package__ = "rllm"
    sys.modules.setdefault("rllm", rllm)

    # rllm.environments (skip __init__.py which re-exports many things)
    envs_path = os.path.join(_RLLM_PATH, "environments")
    envs = types.ModuleType("rllm.environments")
    envs.__path__ = [envs_path]
    envs.__package__ = "rllm.environments"
    sys.modules.setdefault("rllm.environments", envs)

    # rllm.environments.swe (package stub; real submodules loaded normally)
    swe_path = os.path.join(envs_path, "swe")
    swe_pkg = types.ModuleType("rllm.environments.swe")
    swe_pkg.__path__ = [swe_path]
    swe_pkg.__package__ = "rllm.environments.swe"
    sys.modules.setdefault("rllm.environments.swe", swe_pkg)

    # rllm.environments.base (package stub; real submodules loaded normally)
    base_path = os.path.join(envs_path, "base")
    base_pkg = types.ModuleType("rllm.environments.base")
    base_pkg.__path__ = [base_path]
    base_pkg.__package__ = "rllm.environments.base"
    sys.modules.setdefault("rllm.environments.base", base_pkg)


def _stub_arl():
    """Inject a minimal arl stub with the SandboxSession class."""
    arl = types.ModuleType("arl")

    class SandboxSession:
        def __init__(self, **kwargs):
            self.session_id = "test-session-id"

        def create_sandbox(self):
            pass

        def delete_sandbox(self):
            pass

        def execute(self, steps):
            raise NotImplementedError

        @staticmethod
        def attach(session_id, **kwargs):
            s = SandboxSession()
            s.session_id = session_id
            return s

    arl.SandboxSession = SandboxSession
    sys.modules.setdefault("arl", arl)


def _stub_datasets():
    datasets = types.ModuleType("datasets")
    datasets.Dataset = object
    datasets.load_dataset = lambda *a, **kw: None
    sys.modules.setdefault("datasets", datasets)


_stub_rllm_pkg()
_stub_arl()
_stub_datasets()

# ---------------------------------------------------------------------------
# Now import the environment and session pool.
# ---------------------------------------------------------------------------
from rllm.environments.swe.swe import (  # noqa: E402
    SWEEnv,
    _POOL_TRANSIENT_ERRORS,
)
from rllm.environments.swe.session_pool import SessionPool  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
_DUMMY_ENTRY = {
    "instance_id": "test-instance-1",
    "repo_name": "test-repo",
    "commit_hash": "abc123def456",
    "docker_image": "test-image:latest",
    "problem_statement": "Fix the bug.",
}


def _make_env(**kwargs) -> SWEEnv:
    """Create a SWEEnv with a dummy entry and no network calls."""
    defaults = {
        "entry": _DUMMY_ENTRY,
        "pool_scale_retry_delay": 0.0,  # no real sleeping in tests
        "pool_ready_timeout": 0,        # disable proactive wait by default
    }
    defaults.update(kwargs)
    return SWEEnv(**defaults)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestSWEEnvInit(unittest.TestCase):
    """Test that constructor parameters are stored correctly."""

    def test_default_parameters(self):
        env = SWEEnv(entry=_DUMMY_ENTRY, pool_scale_retry_delay=0.0)
        self.assertEqual(env.pool_scale_retries, 5)
        self.assertEqual(env.pool_scale_retry_delay, 0.0)
        self.assertEqual(env.pool_ready_timeout, 600)

    def test_custom_retry_parameters(self):
        env = _make_env(pool_scale_retries=3, pool_scale_retry_delay=10.0)
        self.assertEqual(env.pool_scale_retries, 3)
        self.assertEqual(env.pool_scale_retry_delay, 10.0)

    def test_zero_retries(self):
        env = _make_env(pool_scale_retries=0)
        self.assertEqual(env.pool_scale_retries, 0)

    def test_custom_pool_ready_timeout(self):
        env = _make_env(pool_ready_timeout=120)
        self.assertEqual(env.pool_ready_timeout, 120)


class TestPoolTransientErrors(unittest.TestCase):
    """Sanity-check the _POOL_TRANSIENT_ERRORS constant."""

    def test_known_patterns_present(self):
        patterns = {p.lower() for p in _POOL_TRANSIENT_ERRORS}
        self.assertIn("pool became unhealthy", patterns)
        self.assertIn("errimagepull", patterns)
        self.assertIn("pull qps exceeded", patterns)
        self.assertIn("no ready replicas", patterns)


class TestPoolReadyReplicas(unittest.TestCase):
    """Unit tests for _pool_ready_replicas()."""

    def setUp(self):
        SessionPool._instance = None

    def _resp_mock(self, payload: dict):
        mock = MagicMock()
        mock.read.return_value = json.dumps(payload).encode()
        mock.__enter__ = lambda s: s
        mock.__exit__ = MagicMock(return_value=False)
        return mock

    def test_returns_none_on_network_error(self):
        env = _make_env()
        with patch("urllib.request.urlopen", side_effect=OSError("refused")):
            self.assertIsNone(env._pool_ready_replicas())

    def test_reads_ready_replicas_from_status(self):
        env = _make_env()
        with patch("urllib.request.urlopen", return_value=self._resp_mock({"status": {"readyReplicas": 3}})):
            self.assertEqual(env._pool_ready_replicas(), 3)

    def test_reads_ready_replicas_top_level(self):
        env = _make_env()
        with patch("urllib.request.urlopen", return_value=self._resp_mock({"readyReplicas": 1})):
            self.assertEqual(env._pool_ready_replicas(), 1)

    def test_reads_ready_pods_field(self):
        env = _make_env()
        with patch("urllib.request.urlopen", return_value=self._resp_mock({"status": {"readyPods": 2}})):
            self.assertEqual(env._pool_ready_replicas(), 2)

    def test_returns_none_on_unrecognised_payload(self):
        env = _make_env()
        with patch("urllib.request.urlopen", return_value=self._resp_mock({"other": 99})):
            self.assertIsNone(env._pool_ready_replicas())

    def test_url_uses_gateway_namespace_pool_ref(self):
        env = _make_env()
        env.gateway_url = "http://gw:8080"
        env.namespace = "mynamespace"
        env.pool_ref = "mypool"
        captured = []

        def fake_urlopen(req, timeout):
            captured.append(req.full_url)
            raise OSError("skip")

        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            env._pool_ready_replicas()

        self.assertEqual(
            captured[0],
            "http://gw:8080/api/v1/namespaces/mynamespace/warmpools/mypool",
        )


class TestWaitForPoolReady(unittest.TestCase):
    """Unit tests for _wait_for_pool_ready()."""

    def setUp(self):
        SessionPool._instance = None

    @patch("rllm.environments.swe.swe.time.sleep")
    def test_skips_wait_when_timeout_zero(self, mock_sleep):
        env = _make_env(pool_ready_timeout=0)
        with patch.object(env, "_pool_ready_replicas") as mock_replicas:
            env._wait_for_pool_ready()
            mock_replicas.assert_not_called()
        mock_sleep.assert_not_called()

    @patch("rllm.environments.swe.swe.time.sleep")
    def test_returns_immediately_when_pool_ready(self, mock_sleep):
        env = _make_env(pool_ready_timeout=60)
        with patch.object(env, "_pool_ready_replicas", return_value=2):
            env._wait_for_pool_ready()
        mock_sleep.assert_not_called()

    @patch("rllm.environments.swe.swe.time.sleep")
    def test_skips_wait_when_gateway_unreachable(self, mock_sleep):
        """If gateway returns None (unreachable), skip the wait entirely."""
        env = _make_env(pool_ready_timeout=60)
        with patch.object(env, "_pool_ready_replicas", return_value=None):
            env._wait_for_pool_ready()
        mock_sleep.assert_not_called()

    @patch("rllm.environments.swe.swe.time.sleep")
    @patch("rllm.environments.swe.swe.time.monotonic")
    def test_polls_until_ready(self, mock_monotonic, mock_sleep):
        """Polls the gateway until pool has >=1 ready replica."""
        # Call 1: set deadline → 0 + 60 = 60
        # Call 2: check remaining after first 0-replica result → 10, remaining = 50 > 0
        mock_monotonic.side_effect = [0, 10]
        env = _make_env(pool_ready_timeout=60)
        # First call: 0 replicas (still scaling); second call: 1 ready
        with patch.object(env, "_pool_ready_replicas", side_effect=[0, 1]):
            env._wait_for_pool_ready()
        mock_sleep.assert_called_once()

    @patch("rllm.environments.swe.swe.time.sleep")
    @patch("rllm.environments.swe.swe.time.monotonic")
    def test_raises_timeout_if_pool_never_ready(self, mock_monotonic, mock_sleep):
        """Raises TimeoutError when pool is still not ready after timeout."""
        # deadline = 0 + 60; after first 0-replica check, remaining = 0 + 60 - 61 = -1
        mock_monotonic.side_effect = [0, 61]
        env = _make_env(pool_ready_timeout=60)
        with patch.object(env, "_pool_ready_replicas", return_value=0):
            with self.assertRaises(TimeoutError) as ctx:
                env._wait_for_pool_ready()
        self.assertIn(env.pool_ref, str(ctx.exception))
        mock_sleep.assert_not_called()


class TestCreateSessionRetry(unittest.TestCase):
    """Test that _create_session retries on transient errors."""

    def setUp(self):
        SessionPool._instance = None

    def _make_mock_session(self, session_id="sid-ok"):
        mock = MagicMock()
        mock.session_id = session_id
        return mock

    @patch("rllm.environments.swe.swe.time.sleep")
    @patch("rllm.environments.swe.swe.SandboxSession")
    def test_succeeds_on_first_attempt(self, MockSession, mock_sleep):
        """No retries when create_sandbox succeeds immediately."""
        mock_inst = self._make_mock_session()
        MockSession.return_value = mock_inst

        env = _make_env(pool_scale_retries=3)
        env._create_session()

        mock_inst.create_sandbox.assert_called_once()
        mock_sleep.assert_not_called()

    @patch("rllm.environments.swe.swe.time.sleep")
    @patch("rllm.environments.swe.swe.SandboxSession")
    def test_retries_on_transient_error_then_succeeds(self, MockSession, mock_sleep):
        """Retries the configured number of times on a transient error."""
        transient_exc = RuntimeError(
            'pool became unhealthy while waiting: pool "test-pool" has failing pods '
            "and no ready replicas: container executor: ErrImagePull - pull QPS exceeded"
        )
        good_mock = self._make_mock_session()
        fail_mock1 = self._make_mock_session()
        fail_mock1.create_sandbox.side_effect = transient_exc
        fail_mock2 = self._make_mock_session()
        fail_mock2.create_sandbox.side_effect = transient_exc

        MockSession.side_effect = [fail_mock1, fail_mock2, good_mock]

        env = _make_env(pool_scale_retries=5, pool_scale_retry_delay=1.0)
        env._create_session()

        self.assertEqual(fail_mock1.create_sandbox.call_count, 1)
        self.assertEqual(fail_mock2.create_sandbox.call_count, 1)
        self.assertEqual(good_mock.create_sandbox.call_count, 1)
        self.assertEqual(mock_sleep.call_count, 2)
        mock_sleep.assert_any_call(1.0)
        mock_sleep.assert_any_call(2.0)

    @patch("rllm.environments.swe.swe.time.sleep")
    @patch("rllm.environments.swe.swe.SandboxSession")
    def test_raises_after_max_retries_exhausted(self, MockSession, mock_sleep):
        """Raises the original exception once all retries are used up."""
        transient_exc = RuntimeError("pool became unhealthy: no ready replicas")
        fail_mock = MagicMock()
        fail_mock.session_id = "x"
        fail_mock.create_sandbox.side_effect = transient_exc
        MockSession.return_value = fail_mock

        env = _make_env(pool_scale_retries=2)
        with self.assertRaises(RuntimeError) as ctx:
            env._create_session()

        self.assertIn("pool became unhealthy", str(ctx.exception))
        self.assertEqual(mock_sleep.call_count, 2)

    @patch("rllm.environments.swe.swe.time.sleep")
    @patch("rllm.environments.swe.swe.SandboxSession")
    def test_no_retry_on_non_transient_error(self, MockSession, mock_sleep):
        """Non-transient errors are raised immediately without retry."""
        fail_mock = MagicMock()
        fail_mock.session_id = "x"
        fail_mock.create_sandbox.side_effect = RuntimeError("authentication failed")
        MockSession.return_value = fail_mock

        env = _make_env(pool_scale_retries=5, pool_scale_retry_delay=1.0)
        with self.assertRaises(RuntimeError) as ctx:
            env._create_session()

        self.assertIn("authentication failed", str(ctx.exception))
        self.assertEqual(fail_mock.create_sandbox.call_count, 1)
        mock_sleep.assert_not_called()

    @patch("rllm.environments.swe.swe.time.sleep")
    @patch("rllm.environments.swe.swe.SandboxSession")
    def test_zero_retries_raises_immediately(self, MockSession, mock_sleep):
        """With pool_scale_retries=0 a transient error raises on first attempt."""
        fail_mock = MagicMock()
        fail_mock.session_id = "x"
        fail_mock.create_sandbox.side_effect = RuntimeError("ErrImagePull - pull QPS exceeded")
        MockSession.return_value = fail_mock

        env = _make_env(pool_scale_retries=0)
        with self.assertRaises(RuntimeError):
            env._create_session()

        mock_sleep.assert_not_called()

    @patch("rllm.environments.swe.swe.time.sleep")
    @patch("rllm.environments.swe.swe.SandboxSession")
    def test_session_registered_in_pool_after_success(self, MockSession, mock_sleep):
        """Session is registered in SessionPool after successful creation."""
        good_mock = self._make_mock_session(session_id="registered-sid")
        MockSession.return_value = good_mock

        env = _make_env()
        env._create_session()

        pool = SessionPool.get_instance()
        entry = pool.get("test-instance-1")
        self.assertIsNotNone(entry)
        self.assertEqual(entry.session_id, "registered-sid")

    @patch("rllm.environments.swe.swe.time.sleep")
    @patch("rllm.environments.swe.swe.SandboxSession")
    def test_imagepullbackoff_is_transient(self, MockSession, mock_sleep):
        """ImagePullBackOff variant is also treated as transient."""
        fail_mock = MagicMock()
        fail_mock.session_id = "x"
        fail_mock.create_sandbox.side_effect = RuntimeError(
            "container executor: ImagePullBackOff"
        )
        good_mock = self._make_mock_session()
        MockSession.side_effect = [fail_mock, good_mock]

        env = _make_env(pool_scale_retries=3)
        env._create_session()

        self.assertEqual(mock_sleep.call_count, 1)

    @patch("rllm.environments.swe.swe.time.sleep")
    @patch("rllm.environments.swe.swe.SandboxSession")
    def test_wait_for_ready_called_before_create_sandbox(self, MockSession, mock_sleep):
        """_wait_for_pool_ready() is invoked before the first create_sandbox attempt."""
        good_mock = self._make_mock_session()
        MockSession.return_value = good_mock

        env = _make_env(pool_ready_timeout=60)
        call_order = []

        with patch.object(
            env,
            "_wait_for_pool_ready",
            side_effect=lambda: call_order.append("wait"),
        ):
            good_mock.create_sandbox.side_effect = lambda: call_order.append("create")
            env._create_session()

        self.assertEqual(call_order[0], "wait")
        self.assertIn("create", call_order)


if __name__ == "__main__":
    unittest.main()
