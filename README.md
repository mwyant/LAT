Parakeet Voice Transcriber — Docker + FastAPI (Windows 11 / Docker Desktop / WSL2)

Quick start (recommended: run from WSL distro, e.g., Ubuntu)
1. Ensure Docker Desktop for Windows is installed and WSL2 integration is enabled.
   - Docker Desktop > Settings > General: enable "Use the WSL 2 based engine".
   - Docker Desktop > Settings > Resources > WSL Integration: enable your distro (e.g., Ubuntu).
2. Enable GPU support in Docker Desktop:
   - Docker Desktop > Settings > Resources > GPU: check "Enable GPU support" and select your GPU.
   - Install NVIDIA WSL driver on Windows (see NVIDIA docs).
3. From your WSL distro shell, cd into the project folder and run:
   docker compose up --build
4. Open http://localhost:8000 in your Windows browser and upload an audio file.

Notes
- The Dockerfile points pip to PyTorch's cu124 wheel index so torch==2.1.2+cu124 resolves.
- Keep your project files inside the WSL filesystem for best performance (e.g., /home/you/projects/...).
- If model loading fails or runs out of memory, check container logs and adjust GPU memory / attention context.
- The "gpus: all" setting requires Docker Desktop GPU support enabled; there's no need to install legacy nvidia-docker on Windows.

