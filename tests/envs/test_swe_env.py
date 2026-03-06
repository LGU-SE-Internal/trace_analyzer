"""Tests for SWEEnv fault-tolerance retry logic in _create_session()."""

import sys
import types
import unittest
from unittest.mock import MagicMock, call, patch


# ---------------------------------------------------------------------------
# Stub the `arl` module so the tests can run without the real ARL package.
# ---------------------------------------------------------------------------
def _make_arl_stub():
    arl_mod = types.ModuleType("arl")

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

    arl_mod.SandboxSession = SandboxSession
    return arl_mod


sys.modules.setdefault("arl", _make_arl_stub())

# Stub heavy optional deps that SWEEnv imports at module level.
for _mod in ("datasets",):
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()

# Now we can import the environment.
from rllm.environments.swe.swe import SWEEnv, _POOL_TRANSIENT_ERRORS  # noqa: E402
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
    }
    defaults.update(kwargs)
    return SWEEnv(**defaults)


class TestSWEEnvInit(unittest.TestCase):
    """Test that new parameters are stored correctly."""

    def test_default_retry_parameters(self):
        env = _make_env()
        self.assertEqual(env.pool_scale_retries, 5)
        self.assertEqual(env.pool_scale_retry_delay, 0.0)

    def test_custom_retry_parameters(self):
        env = _make_env(pool_scale_retries=3, pool_scale_retry_delay=10.0)
        self.assertEqual(env.pool_scale_retries, 3)
        self.assertEqual(env.pool_scale_retry_delay, 10.0)

    def test_zero_retries(self):
        env = _make_env(pool_scale_retries=0)
        self.assertEqual(env.pool_scale_retries, 0)


class TestPoolTransientErrors(unittest.TestCase):
    """Sanity-check the _POOL_TRANSIENT_ERRORS constant."""

    def test_known_patterns_present(self):
        patterns = {p.lower() for p in _POOL_TRANSIENT_ERRORS}
        self.assertIn("pool became unhealthy", patterns)
        self.assertIn("errimagepull", patterns)
        self.assertIn("pull qps exceeded", patterns)
        self.assertIn("no ready replicas", patterns)


class TestCreateSessionRetry(unittest.TestCase):
    """Test that _create_session retries on transient errors."""

    def setUp(self):
        # Reset the singleton so tests don't interfere with each other.
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
        good_mock.create_sandbox.return_value = None

        # Fail twice, then succeed.
        fail_mock1 = self._make_mock_session()
        fail_mock1.create_sandbox.side_effect = transient_exc
        fail_mock2 = self._make_mock_session()
        fail_mock2.create_sandbox.side_effect = transient_exc

        MockSession.side_effect = [fail_mock1, fail_mock2, good_mock]

        env = _make_env(pool_scale_retries=5, pool_scale_retry_delay=1.0)
        env._create_session()

        # create_sandbox called 3 times total
        self.assertEqual(fail_mock1.create_sandbox.call_count, 1)
        self.assertEqual(fail_mock2.create_sandbox.call_count, 1)
        self.assertEqual(good_mock.create_sandbox.call_count, 1)

        # sleep called twice (after attempt 0 and attempt 1)
        self.assertEqual(mock_sleep.call_count, 2)
        # Exponential backoff: delay * 2^0, delay * 2^1
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

        env = _make_env(pool_scale_retries=2, pool_scale_retry_delay=0.0)

        with self.assertRaises(RuntimeError) as ctx:
            env._create_session()

        self.assertIn("pool became unhealthy", str(ctx.exception))
        # sleep called twice (after attempt 0 and attempt 1; attempt 2 raises)
        self.assertEqual(mock_sleep.call_count, 2)

    @patch("rllm.environments.swe.swe.time.sleep")
    @patch("rllm.environments.swe.swe.SandboxSession")
    def test_no_retry_on_non_transient_error(self, MockSession, mock_sleep):
        """Non-transient errors are raised immediately without retry."""
        non_transient_exc = RuntimeError("authentication failed: invalid token")
        fail_mock = MagicMock()
        fail_mock.session_id = "x"
        fail_mock.create_sandbox.side_effect = non_transient_exc
        MockSession.return_value = fail_mock

        env = _make_env(pool_scale_retries=5, pool_scale_retry_delay=1.0)

        with self.assertRaises(RuntimeError) as ctx:
            env._create_session()

        self.assertIn("authentication failed", str(ctx.exception))
        # Only one attempt, no sleep.
        self.assertEqual(fail_mock.create_sandbox.call_count, 1)
        mock_sleep.assert_not_called()

    @patch("rllm.environments.swe.swe.time.sleep")
    @patch("rllm.environments.swe.swe.SandboxSession")
    def test_zero_retries_raises_immediately(self, MockSession, mock_sleep):
        """With pool_scale_retries=0 a transient error is raised on first attempt."""
        transient_exc = RuntimeError("ErrImagePull - pull QPS exceeded")
        fail_mock = MagicMock()
        fail_mock.session_id = "x"
        fail_mock.create_sandbox.side_effect = transient_exc
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
        transient_exc = RuntimeError(
            "container executor: ImagePullBackOff - back-off pulling image"
        )
        good_mock = self._make_mock_session()
        fail_mock = MagicMock()
        fail_mock.session_id = "x"
        fail_mock.create_sandbox.side_effect = transient_exc

        MockSession.side_effect = [fail_mock, good_mock]

        env = _make_env(pool_scale_retries=3)
        env._create_session()  # Should not raise

        self.assertEqual(mock_sleep.call_count, 1)


if __name__ == "__main__":
    unittest.main()
