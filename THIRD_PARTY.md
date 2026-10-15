# Third-party sources and attribution

This repository contains project code and configuration; upstream projects are
not bundled. Their software and datasets retain their respective licenses.

- **DT-CORL:** <https://github.com/simonjzh/DT-CORL>, pinned architecture commit
  `7e44adf934b8aef91ca6fb9e057a1935c6a167c9`. The adapter loads
  `utils/network.py` from a user-provided upstream checkout and verifies its
  SHA-256 (`f1ead7f83ef231ca0b2786c24a47b3cbef1179393f6b754e84cb98a08992f81f`).
  The comparison uses a frozen-policy adaptation protocol, not the original
  joint belief/policy training procedure.
- **ManiSkill:** <https://github.com/haosulab/ManiSkill>, version 3.0.1,
  commit `a4a4f9272ad64b1564035874b605ceb687b63ed8`. Diffusion Policy interfaces
  use its official state-based baseline implementation and demonstrations.
- **Diffusion Policy:** <https://diffusion-policy.cs.columbia.edu/>. The
  CADR-DP interface preserves action chunking and replanning; it does not claim
  to introduce the upstream diffusion policy.
- **Minari:** <https://minari.farama.org/>, for offline locomotion and Adroit
  datasets. Follow the dataset-specific source, license, and version metadata.
- **Gymnasium / Gymnasium-Robotics:** <https://gymnasium.farama.org/> and
  <https://robotics.farama.org/>, for environment interfaces.
- **MuJoCo:** <https://github.com/google-deepmind/mujoco>.
- **Mobile latency data:** <https://github.com/NUWiNS/pam2025-multi-carrier-dataset>.
  The publication also uses locally measured WiFi traces, which are not included
  in this code release.

Dependencies such as PyTorch, NumPy, SciPy, SAPIEN, and diffusers are installed
separately and retain their own licenses. This notice is attribution, not a
replacement for any upstream license. No new repository-wide license is imposed
on the project code in this initial release.
