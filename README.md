# SWE-rebench-V2

[![⭐ OpenReward Environment](https://img.shields.io/badge/%E2%AD%90%20OpenReward-Environment-f7e6cc)](https://openreward.ai/nebius/SWE-rebench-V2)

## Description

SWE-rebench-V2 is an OpenReward port of the [SWE-rebench V2](https://github.com/SWE-rebench/SWE-rebench-V2) dataset by Badertdinov et al. (Nebius AI). It evaluates agents on real-world software engineering tasks across multiple programming languages. Agents are given a repository checked out to a specific commit and a problem statement, and must modify the source code so that previously-failing tests pass without breaking existing tests. The dataset covers 32K+ instances across Python, JavaScript, Go, Rust, Java, Ruby, and many more languages.

This fork can use OpenReward's hosted sandboxes or run task images locally with
Docker or Enroot.

## Local self-hosting

### 1. Install the server

Python 3.11 is recommended.

```bash
uv venv --python 3.11
uv pip install -r requirements.txt
```

### 2. Download and index the dataset

The server image intentionally does not contain the multi-gigabyte dataset.
Download the Hugging Face parquet shards to persistent storage:

```bash
export DATA_DIR="$HOME/data/SWE-rebench-V2"
uvx --from huggingface-hub hf download nebius/SWE-rebench-V2 \
  --repo-type dataset \
  --local-dir "$DATA_DIR" \
  --include 'data/*.parquet'

.venv/bin/python build_index.py --data-dir "$DATA_DIR"
```

The index excludes rows without a test command and lets the server read one
parquet row group at a time instead of loading roughly 2 GB into each server
process.

### 3a. Run with local Docker task sandboxes

```bash
export DATA_DIR="$HOME/data/SWE-rebench-V2"
export SWE_SANDBOX_RUNTIME=docker
export OPENREWARD_PORT=8080
.venv/bin/python server.py
```

You can also containerize the server itself. Mounting the Docker socket lets
the server create sibling task containers:

```bash
docker build -t openreward-swe-rebench-v2:local .
docker run --rm \
  -p 8080:8080 \
  -v "$DATA_DIR:/orwd_data:ro" \
  -v /var/run/docker.sock:/var/run/docker.sock \
  -e SWE_SANDBOX_RUNTIME=docker \
  openreward-swe-rebench-v2:local
```

Task containers use `--network none` by default. Set `SWE_DOCKER_NETWORK` only
if a task explicitly needs another Docker network.

### 3b. Run with Enroot on a Slurm node

Run the Python environment server directly on the allocated host and let it
invoke the host's Enroot installation:

```bash
export DATA_DIR=/shared/datasets/SWE-rebench-V2
export SWE_SANDBOX_RUNTIME=enroot
export OPENREWARD_PORT="${OPENREWARD_PORT:-8080}"

# Keep writable Enroot state on node-local storage.
export ENROOT_CACHE_PATH="${SLURM_TMPDIR:-/tmp}/enroot-cache"
export ENROOT_DATA_PATH="${SLURM_TMPDIR:-/tmp}/enroot-data"
export ENROOT_RUNTIME_PATH="${SLURM_TMPDIR:-/tmp}/enroot-runtime"

# Squashfs task images may instead use a persistent shared cache. It must
# support file locking and atomic rename if multiple nodes write to it.
export SWE_ENROOT_IMAGE_CACHE="${SLURM_TMPDIR:-/tmp}/swe-rebench-images"

mkdir -p \
  "$ENROOT_CACHE_PATH" \
  "$ENROOT_DATA_PATH" \
  "$ENROOT_RUNTIME_PATH" \
  "$SWE_ENROOT_IMAGE_CACHE"

.venv/bin/python server.py
```

Each task image is imported lazily from the OCI registry, cached as squashfs,
and unpacked into a unique writable Enroot container for the session. The
first use of an image is therefore slower. Do not try to pre-import all 32K+
images; cache the subset used by the current training run.

Running the server inside one Enroot container and launching more Enroot
containers from inside it requires nested user/mount namespaces and is not
portable across HPC configurations. Running the lightweight Python server on
the host avoids this nesting while task execution remains containerized.

Enroot uses host networking. Unlike the Docker backend, it cannot enforce
`--network none`; use cluster-level egress controls when task network
isolation is required.

### Client configuration

Point the OpenReward client at the node and selected port:

```bash
export OPENREWARD_API_URL="http://NODE_HOSTNAME:8080"
export OPENREWARD_SESSION_URL="http://NODE_HOSTNAME:8080"
```

The server port can be set with `OPENREWARD_PORT` (preferred) or `PORT`.

### Runtime configuration

- `SWE_SANDBOX_RUNTIME`: `hosted` (default), `docker`, or `enroot`.
- `SWE_SANDBOX_COMMAND_TIMEOUT_SECONDS`: ordinary command timeout; default 600.
- `SWE_TEST_TIMEOUT_SECONDS`: submission test timeout; default 600.
- `SWE_SANDBOX_CREATE_TIMEOUT_SECONDS`: image import/create timeout; default 1800.
- `SWE_TOOL_OUTPUT_MAX_CHARS`: maximum returned command output; default 50000.
- `SWE_ENROOT_IMAGE_CACHE`: squashfs cache location.
- `SWE_ENROOT_ROOT_REMAP`: pass `--root` to Enroot; enabled by default.
- `SWE_ENROOT_MOUNTS`: semicolon-separated Enroot mounts such as
  `/scratch:/scratch;/datasets:/datasets`.
- `SWE_DOCKER_CPUS` and `SWE_DOCKER_MEMORY`: optional task-container limits.

## Community

You can reach out with any questions in Discord: https://discord.gg/V8FqXQ4CgU

## Capabilities

- Multi-language codebase navigation and understanding
- Bug diagnosis from problem statements and test failures
- Source code editing to fix defects
- Reasoning about test expectations and code behavior

## Compute Requirements

Agents are given a sandboxed environment with a pre-built instance image for
each task. Hosted mode defaults to the `2:4` OpenReward machine size. Local
Docker limits are configurable; Enroot resources are controlled by the Slurm
allocation.

## License

[MIT](https://opensource.org/licenses/MIT). The underlying SWE-rebench V2 dataset is subject to its own license terms.

## Tasks

There is one split in this environment:

- **Train**: 32K+ software engineering tasks

Each task provides a repository, base commit, problem statement, and a set of tests that should transition from failing to passing after the agent's fix. Tasks span issue-based and PR-based scenarios across dozens of programming languages and frameworks.

## Reward Structure

This is a multi-turn environment with binary reward by default:

- **1.0** — All FAIL_TO_PASS tests now pass and all PASS_TO_PASS tests remain passing
- **0.0** — Any required test fails or regresses

On submission, the environment applies the held-out test patch, runs the task's test command, and parses the output using a language/framework-specific log parser to determine per-test pass/fail status.

Set `OPENREWARD_REWARD_MODE=partial` to return the mean of the
FAIL_TO_PASS pass rate and PASS_TO_PASS pass rate. Equal group weighting keeps
large regression suites from overwhelming the actual bug-fix signal.

## Data

Data is read lazily from one parquet file or multiple parquet shards under
`DATA_DIR`. Each row contains the instance ID, repository, base commit, test
patch, problem statement, OCI image name, language, test expectations
(FAIL_TO_PASS and PASS_TO_PASS lists), and install/test configuration. The
dataset is derived from the SWE-rebench V2 collection on Hugging Face
(`nebius/SWE-rebench-V2`).

## Tools

| Tool | Description |
|------|-------------|
| `bash` | Run bash commands in the sandbox container. |
| `str_replace` | Replace a unique string in a file with another string. |
| `view` | View file contents or directory listings. |
| `create_file` | Create a new file with specified content. |
| `submit_answer` | Submit the solution. Applies the test patch, runs the test suite, and returns reward. |

## Time Horizon

SWE-rebench-V2 is a multi-turn environment. Agents explore the repository, read code, diagnose the issue, make edits, and optionally run tests before submitting. A typical task may involve 10-50+ tool calls depending on complexity.

## Environment Difficulty

SWE-rebench V2 is a challenging benchmark spanning many languages and difficulty levels. Tasks are annotated with difficulty codes. Performance varies significantly by language, framework, and problem complexity. As of the paper's publication, frontier models solve a modest fraction of tasks, with Python tasks being the most commonly attempted.

## Safety

Agents operate within sandboxed Docker containers with no network access to external services. The environment does not involve private data or production systems. Agents can only modify files within the repository checkout; the test patch is applied automatically at submission time and cannot be tampered with.

## Citations

```bibtex
@misc{badertdinov2026swerebenchv2languageagnosticswe,
      title={SWE-rebench V2: Language-Agnostic SWE Task Collection at Scale},
      author={Ibragim Badertdinov and Maksim Nekrashevich and Anton Shevtsov and Alexander Golubev},
      year={2026},
      eprint={2602.23866},
      archivePrefix={arXiv},
      primaryClass={cs.SE},
      url={https://arxiv.org/abs/2602.23866},
}
```
