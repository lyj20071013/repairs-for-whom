"""ALFWorld（TextWorld 文本版）第二环境。

阶段 A（可行性）：alfw_stage_a.py。模块划分：
    acfg.py     全部设置与阶段 A 的事先判断标准
    games.py    游戏文件收集与抽样（只读 json，不依赖 textworld）
    twenv.py    真实 ALFWorld 环境封装（重放、规划器、状态键、加速补丁）
    mockenv.py  离线测试用的模拟家务环境（同一接口）
    pool.py     多进程环境池（主进程跑 7B，子进程推进环境）
    agent.py    提示、动作解析、vLLM / 模拟后端
    rollout.py  批量交错 rollout
"""
