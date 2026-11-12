# Publishing on GitHub

The companion `paper-github-20261028.zip` contains a clean `repairs-for-whom/` repository directory. Extract it and use Git or GitHub Desktop to create and publish a repository.

Alternatively, use the existing local `paper` directory with Git, respecting `.gitignore`. Pre-existing private notes and the old root manuscript/PDF remain excluded. Avoid dragging the entire original working folder into the GitHub web uploader: it does not apply the local ignore rules.

Run `python tools/verify_release.py` before the first commit. `README.md` is the English landing page. The Apache-2.0 license and third-party notices are already included, so do not generate a conflicting license when creating the repository.

The 2026-11-13 update follows the same procedure: replace the working tree with the update folder (keeping `.git`), run `python tools/verify_release.py`, review `git status`, then commit and push. P2w is included since the 2026-10-28 snapshot. For future updates, follow `REPRODUCING.md` and update the result version, status and checksums together. Named authors are included in the arXiv manuscript; add its public link and formal citation after arXiv v1 is available. Use Git to retain dotfiles and the version history.
