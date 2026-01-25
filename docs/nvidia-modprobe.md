# nvidia-modprobe issue

Summary
- Symptom: `nvidia-smi` fails with `Failed to initialize NVML: Unknown Error` (GPU telemetry unavailable).
- In severe cases, CUDA can also fail (e.g., PyTorch `torch.cuda.is_available()` returns false with `cudaErrorOperatingSystem`).
- There are two common buckets:
  1) **Missing /dev nodes** (`/dev/nvidia*` not present) → usually fixable with `nvidia-modprobe`.
  2) **Driver/GPU in a bad state** (nodes present but NVML/CUDA still fail) → usually needs root-level service/module reset or reboot.
  3) **Sandboxed process** (NVML works in a normal shell, but fails only inside a sandbox) → check `NoNewPrivs`/`Seccomp` in `/proc/self/status` and run outside the sandbox.

Evidence
- This machine: RTX 5090 D, driver/kernel module `590.48.01` loaded (`/proc/driver/nvidia/version`).
- `/proc/driver/nvidia/gpus/0000:01:00.0/information` shows the GPU is detected and in D0 power state.
- `/dev/nvidia0`, `/dev/nvidiactl`, `/dev/nvidia-uvm` exist.
- Observed in practice: NVML/CUDA can fail only inside sandboxed agent processes (seccomp / `NoNewPrivs=1`) while working fine in normal shells → points to bucket (3), not a driver outage.

Recovery steps
1) Sanity check device nodes:
   - `ls -l /dev/nvidia* /dev/nvidia-caps/* 2>/dev/null || true`
2) If `/dev/nvidia*` are missing, try:
   - `nvidia-modprobe -u -c 0`
   - `nvidia-modprobe -m` (modeset) and `nvidia-modprobe -u` (uvm) if needed
3) If nodes exist but NVML still fails, inspect kernel logs for NVRM/Xid (no sudo required):
   - `journalctl -k -b --no-pager | rg -i "nvrm|xid|nvml|nvidia" | tail -n 200`
3b) If `nvidia-smi` works in a normal terminal but fails inside an agent/sandbox:
   - Check sandbox flags: `cat /proc/self/status | rg -n "NoNewPrivs|Seccomp"`
   - Fix: run the GPU command from a non-sandboxed shell, or launch the agent with sandbox disabled (tradeoff: less safety).
4) If you see Xid/fallen-off-bus style errors (or NVML stays broken), the practical fix is a driver reset:
   - Restart persistence daemon (requires root): `sudo systemctl restart nvidia-persistenced`
   - Reload kernel modules (requires root; run from a TTY, will kill graphics):
     - `sudo modprobe -r nvidia_uvm nvidia_drm nvidia_modeset nvidia`
     - `sudo modprobe nvidia nvidia_modeset nvidia_uvm nvidia_drm`
   - Worst case: reboot.
5) If running inside a container: ensure the container is started with proper GPU support (`--gpus all` + nvidia-container-toolkit). Bind-mounting `/dev/nvidia*` alone is often insufficient for NVML/CUDA.

Prevention
- Consider enabling `nvidia-persistenced` at boot if this recurs.
- If the system suspends/hibernates, re-run `nvidia-modprobe -u -c 0` before training.
