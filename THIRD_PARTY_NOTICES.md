# Licensing and third-party materials

Original project code in `experiments/`, `framework/`, `paper_figs/`, and `tools/` is licensed under the Apache License 2.0 in [LICENSE](LICENSE), except material carrying its own license.

- The WebShop subset in `experiments/webshop/vendor/` and `framework/runtime/webshop/vendor/` originates from [princeton-nlp/WebShop](https://github.com/princeton-nlp/WebShop), commit prefix `64fa2a5`. Its original MIT `LICENSE.md` remains inside each archive. It is a subset, not the full upstream distribution.
- ALFWorld/TextWorld, vLLM, NumPy, model weights and external datasets retain their own licenses. We do not redistribute their installed environments or model weights.
- LaTeX style files in `manuscript/` retain their original notices and terms; the root code license does not relicense these files.
- Trajectories include observations and instructions from ALFWorld and WebShop, including product descriptions. Publishing research records does not transfer ownership of third-party text or put that text under Apache-2.0. See the upstream environment/data terms when reusing those contents. No separate blanket license for the research datasets or manuscript is asserted by this release.
- API model access is not included. Users supply their own credentials through environment variables.

Research records are preserved byte-for-byte. Source comments, docstrings and runtime display text have been localized into English. Python AST checks verify that computation, control flow and originally English string literals are unchanged. Original and localized source hashes are recorded separately in `provenance/english_localization.json`; third-party vendor archives remain unchanged.
