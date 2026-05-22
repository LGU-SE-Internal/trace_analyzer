import base64
import json
import os
import re

import numpy as np
from datasets import Dataset, load_dataset

from arl import ManagedSession

from rllm.environments.swe.action import Action
from rllm.environments.swe.reward import calculate_reward
from rllm.environments.base.base_env import BaseEnv

TOOLS_DIR = os.path.join(os.path.dirname(__file__), "tools")
CONTINUE_MSG = """
You forgot to use a function call in your response.
YOU MUST USE A FUNCTION CALL IN EACH RESPONSE.

IMPORTANT: YOU SHOULD NEVER ASK FOR HUMAN HELP.
"""

CMD_TIMEOUT = 120  # seconds

# Matches R2E-Gym's DOCKER_PATH: ensures /root/.venv/bin (symlinked to the
# correct conda env or repo venv during setup_env) is first on PATH, so that
# ``python``, ``pip``, and all project CLI tools resolve to the right env
# without requiring ``conda activate``.
DOCKER_PATH = (
    "/root/.venv/bin:/root/.local/bin:/root/.cargo/bin"
    ":/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
)

SKIP_FILES_NEW = ["run_tests.sh", "r2e_tests"]

# Only tools that do real work inside the sandbox.
# execute_bash/finish/submit are handled directly in step().
R2EGYM_TOOL_FILES = [
    os.path.join(TOOLS_DIR, "r2egym/file_editor.py"),
    os.path.join(TOOLS_DIR, "r2egym/search.py"),
]

SWEAGENT_TOOL_FILES = [
    os.path.join(TOOLS_DIR, "sweagent/str_replace_editor.py"),
]

BLOCKED_COMMANDS = frozenset(["git", "ipython", "jupyter", "nohup"])

R2E_ENV_IDS = [
    "R2E-Gym/R2E-Gym-Subset",
    "R2E-Gym/R2E-Gym-V1",
    "R2E-Gym/R2E-Gym-Lite",
    "R2E-Gym/SWE-Bench-Verified",
    "R2E-Gym/SWE-Bench-Lite",
]
DEFAULT_R2E_ENV_ID = "R2E-Gym/R2E-Gym-Lite"
_TRUNCATION_LINES = 40


def format_observation(output: str, error_code: str, action_name: str) -> str:
    """Format command output into an observation string.

    For bash commands: includes exit code and truncates long output
    (keeps first/last 40 lines) to save LLM context.
    For tool commands (file_editor, search, etc.): light header only,
    since tool scripts handle their own truncation internally.
    """
    if action_name in ("execute_bash", "bash"):
        lines = output.splitlines() if output else []
        if len(lines) > 2 * _TRUNCATION_LINES:
            top = "\n".join(lines[:_TRUNCATION_LINES])
            bottom = "\n".join(lines[-_TRUNCATION_LINES:])
            divider = "-" * 50
            output = (
                f"{top}\n"
                f"{divider}\n"
                f"<Observation truncated in middle for saving context>\n"
                f"{divider}\n"
                f"{bottom}"
            )
        return (
            f"Exit code: {error_code}\nExecution output of [{action_name}]:\n{output}"
        )

    return f"Execution output of [{action_name}]:\n{output}"


def _mirror_image(docker_image: str) -> str:
    """Rewrite docker image to mirror registry (matches batch_prefetch.py convention).

    Set ARL_MIRROR_REGISTRY="" to disable mirroring and use original images.
    """
    registry = os.environ.get(
        "ARL_MIRROR_REGISTRY", "pair-diag-cn-guangzhou.cr.volces.com"
    )
    if not registry:
        return docker_image
    namespace = os.environ.get("ARL_MIRROR_NAMESPACE", "code")
    parts = docker_image.split("/", 1)
    image_path = parts[1] if len(parts) == 2 else docker_image
    return f"{registry}/{namespace}/{image_path}"


class SWEEnv(BaseEnv):
    """Software Engineering Environment backed by ARL sandbox sessions."""

    def __init__(
        self,
        entry: dict | None = None,
        dataset: Dataset | None = None,
        idx: int | None = None,
        step_timeout: int = 90,
        reward_timeout: int = 300,
        gateway_url: str | None = None,
        namespace: str = "default",
        experiment_id: str | None = None,
        max_replicas: int | None = None,
        verbose: bool = False,
        scaffold: str = "r2egym",
        normalize_pytest: bool = False,
    ):
        if entry is not None:
            self.entry = entry
            self.dataset = None
            self.idx = None
        else:
            if dataset is None:
                dataset = load_dataset(DEFAULT_R2E_ENV_ID, split="test")
            self.dataset = dataset
            if idx is None:
                idx = np.random.randint(0, len(self.dataset))
            assert 0 <= idx < len(self.dataset), "Selected index out of range"
            self.idx = idx
            self.entry = self.dataset[idx]

        self.step_timeout = step_timeout
        self.reward_timeout = reward_timeout
        self.gateway_url = gateway_url or os.environ.get(
            "ARL_GATEWAY_URL", "http://localhost:8080"
        )
        self.namespace = namespace
        self.experiment_id = experiment_id or os.environ.get(
            "ARL_EXPERIMENT_ID", "default"
        )
        self.max_replicas = max_replicas
        self.total_steps = 0
        self.verbose = verbose
        self.scaffold = scaffold
        self.normalize_pytest = normalize_pytest
        self._cmd_counter = 0
        assert scaffold in ["r2egym", "sweagent"], (
            f"Invalid scaffold: {scaffold}, must be one of ['r2egym', 'sweagent']"
        )

        # ARL session (created in reset)
        self.session: ManagedSession | None = None
        self._closed = False

        # Detect dataset type and store mirrored image for ManagedSession
        image = self.entry.get("docker_image", self.entry.get("image_name", ""))
        self._image = _mirror_image(image)
        self.swebench_verified = "swebench" in image
        self.repo_path = "/testbed"
        self.alt_path = "/" if self.swebench_verified else "/root"

    def _execute_raw(
        self, cmd: str, timeout: int = CMD_TIMEOUT, workdir: str | None = None
    ) -> tuple[str, str, int]:
        """Execute a command and return raw (stdout, stderr, exit_code).

        Low-level method that returns stdout/stderr separately so callers
        can format them as needed (e.g. with [STDOUT]/[STDERR] headers).

        Environment activation strategy differs by dataset type to match
        each upstream implementation:

        - **SWE-bench Verified**: ``conda activate testbed`` before every
          command, mirroring the swebench harness ``make_eval_script_list_py``
          which prefixes every eval script with
          ``source /opt/miniconda3/bin/activate && conda activate testbed``.
          ``conda activate`` is needed (not just PATH) because it also runs
          activation scripts in ``etc/conda/activate.d/`` and sets
          ``CONDA_PREFIX`` etc. that some packages depend on.

        - **R2E-Gym**: ``env={PATH: DOCKER_PATH}`` with ``/root/.venv/bin``
          first, matching R2E-Gym's ``DockerRuntime.run()`` which passes
          ``environment={"PATH": DOCKER_PATH}`` to ``container.exec_run``.
          R2E-Gym containers use a plain venv (not conda) so PATH is
          sufficient.
        """
        self._cmd_counter += 1
        workdir = workdir or self.repo_path
        assert self.session is not None, "Session not initialized"

        if self.swebench_verified:
            # SWE-bench: conda activate (matches swebench harness eval scripts)
            shell_cmd = (
                f"source /opt/miniconda3/bin/activate && "
                f"conda activate testbed && "
                f"{cmd}"
            )
            step = {
                "name": f"cmd_{self._cmd_counter}",
                "command": ["bash", "-c", shell_cmd],
                "workDir": workdir,
                "timeout": timeout,
            }
        else:
            # R2E-Gym: PATH-based activation (matches R2E-Gym DOCKER_PATH)
            step = {
                "name": f"cmd_{self._cmd_counter}",
                "command": ["bash", "-c", cmd],
                "env": {"PATH": DOCKER_PATH},
                "workDir": workdir,
                "timeout": timeout,
            }

        response = self.session.execute(steps=[step])
        result = response.results[0]
        stdout = re.sub(r"\x1b\[[0-9;]*m|\r", "", result.output.stdout or "")
        stderr = re.sub(r"\x1b\[[0-9;]*m|\r", "", result.output.stderr or "")
        return stdout, stderr, result.output.exit_code

    def _run(
        self, cmd: str, timeout: int = CMD_TIMEOUT, workdir: str | None = None
    ) -> tuple[str, str]:
        """Execute a shell command in the sandbox session.

        Returns (output, error_code_str) matching the previous DockerRuntime interface.
        """
        stdout, stderr, exit_code = self._execute_raw(cmd, timeout, workdir)
        output = stdout
        if stderr:
            output = output + "\n" + stderr if output else stderr

        if exit_code == 124:  # timeout exit code
            return f"The command took too long to execute (>{timeout}s)", "-1"
        if exit_code != 0:
            return output, f"Error: Exit code {exit_code}"
        return output, str(exit_code)

    def _copy_to_sandbox(self, src_path: str, dest_path: str):
        """Copy a local file into the sandbox via base64 encoding."""
        with open(src_path, "rb") as f:
            content = f.read()
        b64 = base64.b64encode(content).decode()
        dir_path = os.path.dirname(dest_path)
        # Split into chunks to avoid shell argument length limits
        chunk_size = 65536
        if len(b64) <= chunk_size:
            self._run(
                f"mkdir -p {dir_path} && printf '%s' '{b64}' | base64 -d > {dest_path}"
            )
        else:
            self._run(f"mkdir -p {dir_path} && : > {dest_path}")
            for i in range(0, len(b64), chunk_size):
                chunk = b64[i : i + chunk_size]
                self._run(f"printf '%s' '{chunk}' | base64 -d >> {dest_path}")

    def _setup_env(self):
        """Initialize the sandbox environment (same steps as DockerRuntime.setup_env)."""
        if self.swebench_verified:
            self._run("chmod +x /run_tests.sh")
            self._run("ln -sf /opt/miniconda3/envs/testbed /root/.venv")
            self._run("/root/.venv/bin/python -m pip install chardet")
        else:
            self._run(f"ln -sf {self.repo_path}/.venv {self.alt_path}/.venv")
            self._run(
                f"ln -sf {self.repo_path}/.venv/bin/python {self.alt_path}/.local/bin/python"
            )
            self._run(
                f"ln -sf {self.repo_path}/.venv/bin/python {self.alt_path}/.local/bin/python3"
            )
            self._run(
                f"find {self.repo_path}/.venv/bin -type f -executable "
                f"-exec ln -sf {{}} {self.alt_path}/.local/bin/ \\;"
            )
            self._run("uv pip install chardet")
            self._run("find . -name '*.pyc' -delete")
            self._run("find . -name '__pycache__' -exec rm -rf {} +")
            self._run("find /r2e_tests -name '*.pyc' -delete 2>/dev/null; true")
            self._run("find /r2e_tests -name '__pycache__' -exec rm -rf {} + 2>/dev/null; true")
            for skip_file in SKIP_FILES_NEW:
                if skip_file == "r2e_tests":
                    continue
                self._run(
                    f"mv {self.repo_path}/{skip_file} {self.alt_path}/{skip_file}"
                )
            # Move r2e_tests to alt_path and symlink back.
            # Some images have r2e_tests at /r2e_tests, others at /testbed/r2e_tests.
            self._run(
                f"if [ -d /r2e_tests ] && [ ! -d {self.alt_path}/r2e_tests ]; then "
                f"  mv /r2e_tests {self.alt_path}/r2e_tests; "
                f"fi"
            )
            self._run(
                f"if [ -d {self.repo_path}/r2e_tests ] && [ ! -L {self.repo_path}/r2e_tests ] "
                f"   && [ ! -d {self.alt_path}/r2e_tests ]; then "
                f"  mv {self.repo_path}/r2e_tests {self.alt_path}/r2e_tests; "
                f"fi"
            )
            # Ensure symlink from repo_path → alt_path (if not already there)
            self._run(
                f"if [ -d {self.alt_path}/r2e_tests ] && [ ! -e {self.repo_path}/r2e_tests ]; then "
                f"  ln -s {self.alt_path}/r2e_tests {self.repo_path}/r2e_tests; "
                f"fi"
            )

        # Per-repo dependency fixups (mirrors R2E-Gym install_utils)
        from rllm.environments.swe.install_utils import apply_repo_fixups
        docker_image = self.entry.get("docker_image", self.entry.get("image_name", ""))
        apply_repo_fixups(self, docker_image)

    def _provision_tools(self, tool_files: list[str]):
        """Copy tool scripts into sandbox and make them executable."""
        for tool_file in tool_files:
            _, ext = os.path.splitext(tool_file)
            cmd_name = os.path.basename(tool_file)
            if ext == ".py":
                container_cmd_name = cmd_name[:-3]  # strip .py
            else:
                container_cmd_name = cmd_name
            container_path = f"/usr/local/bin/{container_cmd_name}"
            self._copy_to_sandbox(tool_file, container_path)
            self._run(f"chmod +x {container_path}")

    def _patch_pytest_args(self):
        """Patch pytest arguments in the test script for standardized output.

        Ensures -rA (show all test result short info) and --tb=short
        (short traceback) are present on pytest command lines.
        Non-pytest test runners (e.g. Django runtests.py) are left untouched.
        """
        script_path = "/run_tests.sh" if self.swebench_verified else f"{self.alt_path}/run_tests.sh"
        # 1. Replace any existing --tb=<value> with --tb=short
        self._run(f"sed -i 's/--tb=[^ ]*/--tb=short/g' {script_path}")
        # 2. For pytest lines without --tb, add --tb=short after 'pytest'
        self._run(f"sed -i '/pytest/{{/--tb=/!s/pytest/pytest --tb=short/}}' {script_path}")
        # 3. For pytest lines without -rA, add -rA after 'pytest'
        self._run(f"sed -i '/pytest/{{/-rA/!s/pytest/pytest -rA/}}' {script_path}")

    def _get_task_instruction(self) -> str:
        """Extract problem statement from dataset entry."""
        try:
            content = self.entry["problem_statement"]
            match = re.search(r"\[ISSUE\](.*)\[/ISSUE\]", content, re.DOTALL)
            return match.group(1) if match else content
        except Exception:
            return self.entry.get("problem_statement", "")

    def _create_session(self):
        """Create a new ARL sandbox session."""
        kwargs = dict(
            image=self._image,
            experiment_id=self.experiment_id,
            namespace=self.namespace,
            gateway_url=self.gateway_url,
            timeout=max(self.reward_timeout, self.step_timeout) + 60,
        )
        if self.max_replicas is not None:
            kwargs["max_replicas"] = self.max_replicas
        self.session = ManagedSession(**kwargs)
        self.session.create_sandbox()

    # =====================================================
    # BaseEnv interface
    # =====================================================

    def reset(self) -> tuple[str, dict]:
        if self.session:
            self.close()
        self._closed = False
        self._create_session()
        self._setup_env()

        tool_files = (
            R2EGYM_TOOL_FILES if self.scaffold == "r2egym" else SWEAGENT_TOOL_FILES
        )
        self._provision_tools(tool_files)
        if self.normalize_pytest:
            self._patch_pytest_args()
        self.total_steps = 0
        self._cmd_counter = 0
        return self._get_task_instruction(), {}

    def step(self, action: str | Action) -> tuple[str, float, bool, dict]:
        if isinstance(action, str):
            action_obj = Action.from_string(action)
        else:
            action_obj = action

        fn = action_obj.function_name.lower() if action_obj.function_name else ""

        # No function call — remind the agent.
        if not fn:
            return CONTINUE_MSG, 0, False, {}

        done = fn in ("finish", "submit")
        if done:
            self.total_steps += 1
            return "<<< Finished >>>", 0, True, {}

        if fn in ("execute_bash", "bash"):
            # Run the command directly in the sandbox.
            # Output is formatted with [STDOUT]/[STDERR] headers to match
            # the old execute_bash tool script's output format.
            cmd = action_obj.parameters.get("command") or action_obj.parameters.get(
                "cmd", ""
            )
            first_token = cmd.strip().split()[0] if cmd.strip() else ""
            if first_token in BLOCKED_COMMANDS:
                output = (
                    f"Bash command '{first_token}' is not allowed. "
                    "Please use a different command or tool."
                )
                error_code = "Error: Exit code 1"
            else:
                stdout, stderr, exit_code = self._execute_raw(
                    cmd, timeout=self.step_timeout
                )
                if exit_code == 124:
                    output = (
                        f"The command took too long to execute (>{self.step_timeout}s)"
                    )
                    error_code = "-1"
                elif exit_code != 0:
                    output = (
                        f"Error executing command:\n\n"
                        f"[STDOUT]\n\n{stdout.strip()}\n\n"
                        f"[STDERR]\n\n{stderr.strip()}"
                    )
                    error_code = f"Error: Exit code {exit_code}"
                else:
                    output = (
                        f"[STDOUT]\n\n{stdout.strip()}\n\n[STDERR]\n\n{stderr.strip()}"
                    )
                    error_code = str(exit_code)
        else:
            # file_editor, search, str_replace_editor — run the tool binary.
            bash_cmd = action_obj.to_bashcmd()
            output, error_code = self._run(bash_cmd, timeout=self.step_timeout)

        self.total_steps += 1
        return format_observation(output, error_code, fn), 0, False, {}

    def compute_final_reward(self) -> float:
        return calculate_reward(
            session=self.session,
            ds=self.entry,
            repo_path=self.repo_path,
            alt_path=self.alt_path,
            timeout=self.reward_timeout,
        )

    def close(self):
        if self._closed:
            return
        self._closed = True
        session = self.session
        self.session = None
        if session:
            try:
                session.delete_sandbox()
            except Exception as e:
                print(f"[SWEEnv] Warning: delete_sandbox failed (session_id={session.session_id}): {e}")
            finally:
                try:
                    session.close()
                except Exception as e:
                    print(f"[SWEEnv] Warning: session.close failed (session_id={session.session_id}): {e}")

    @staticmethod
    def from_dict(extra_info: dict | str) -> "SWEEnv":
        import inspect

        if isinstance(extra_info, str):
            extra_info = json.loads(extra_info)
        sig = inspect.signature(SWEEnv.__init__)
        init_params = {}
        for param_name, param in sig.parameters.items():
            if param_name == "self":
                continue
            if param_name in extra_info:
                init_params[param_name] = extra_info[param_name]
        init_params["entry"] = extra_info
        return SWEEnv(**init_params)
