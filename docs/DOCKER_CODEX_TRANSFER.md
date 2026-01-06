# Codex Docker Transfer Design

## Goals
- Start a GPU-enabled Docker sandbox that mirrors the host Codex setup.
- Keep dataset mounts read-only.
- Preserve host user ownership for all workspace writes.
- Enter Codex inside the container automatically after confirmation.

## Non-goals
- Reconfigure host OS services inside the container.
- Provide GUI desktop access from inside the container.
- Modify host Docker daemon settings.

## Constraints
- Dataset must be mounted read-only.
- Host Codex config and skills should be used inside the container.
- Use Blackwell-compatible CUDA (sm_120) and PyTorch 2.5+.

## Options Considered
1. Install Codex inside the container
   - Pros: self-contained container.
   - Cons: version drift from host; extra install steps.

2. Bind-mount host Codex binary and config (recommended)
   - Pros: exact version and settings; no re-auth needed.
   - Cons: requires host path to codex binary and write access to ~/.codex.

3. Run Codex on host, only run tools in container
   - Pros: minimal container setup.
   - Cons: not a true sandboxed Codex session.

## Recommended Approach
Use bind mounts for:
- Host Codex config and skills: `~/.codex` -> `/home/codex/.codex`
- Host Codex binary: `$(which codex)` -> `/usr/local/bin/codex`

Set container user to host UID/GID to prevent root-owned files.

## Implementation Details
- Dockerfile:
  - Base: `nvcr.io/nvidia/pytorch:25.01-py3`
  - Install `decord`, `einops`, `timm`, `ffmpeg`
  - Create user with HOST_UID/HOST_GID build args
  - Set `TORCH_CUDA_ARCH_LIST=12.0`

- docker-compose.yaml:
  - Service: `workspace`
  - Mount project root to `/workspace`
  - Mount SSv2 dataset to `/datasets/ssv2:ro`
  - Mount host `~/.codex` and codex binary
  - Set `shm_size: '32gb'`
  - Enable GPU access

- enter_sandbox.sh:
  - Verify dataset mount is read-only.
  - Resolve host codex binary path.
  - Export HOST_UID/HOST_GID/CODEX_BIN for compose.
  - `docker compose up -d` then `docker compose exec -it workspace codex`.

## Security Notes
- Read-only dataset mount prevents accidental deletion.
- Host Codex config is shared, so treat container as trusted.

## Limitations
- Container cannot access host system services or GUI apps.
- Network and device access are limited to Docker configuration.
