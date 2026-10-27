# Publishing on GitHub

The companion `paper-github-20261028.zip` contains a clean `repairs-for-whom/` repository directory. Extract it and use Git or GitHub Desktop to create and publish a repository.

Alternatively, use the existing local `paper` directory with Git, respecting `.gitignore`. Pre-existing private notes and the old root manuscript/PDF remain excluded. Avoid dragging the entire original working folder into the GitHub web uploader: it does not apply the local ignore rules.

Run `python tools/verify_release.py` before the first commit. `README.md` is the English landing page. The Apache-2.0 license and third-party notices are already included, so do not generate a conflicting license when creating the repository.

Follow `REPRODUCING.md` when P2w is complete, then update the result version, status and checksums. Add the paper link, authors and formal citation after arXiv v1 is available. This local preparation does not publish to GitHub.
