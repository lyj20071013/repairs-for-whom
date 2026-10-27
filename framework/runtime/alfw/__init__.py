"""ALFWorld（TextWorld text mode）Environment II。

stage A（Feasibility）：alfw_stage_a.py。Module Division：
    acfg.py     All Settings & Phases A Pre-judgment criteria
    games.py    Game file collection and sampling（Read-only json，Not dependent textworld）
    twenv.py    real ALFWorld Environmental Envelope（replay、planner、Status Keys、Accelerating patches）
    mockenv.py  Simulation of the home environment for offline testing（Same interface）
    pool.py     Multi-process Environment Pool（Main Process Run 7B，Sub-process moving the environment forward）
    agent.py    prompt、Action Parsing、vLLM / Simulate Backend
    rollout.py  Batch Interlacation rollout
"""
