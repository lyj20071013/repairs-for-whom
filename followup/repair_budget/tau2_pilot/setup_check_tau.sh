#!/usr/bin/env bash
# tau 服务器清点（2026-10-29 写）。只读，不安装、不修改任何东西；无卡模式也能跑。
# 用法（在 /root/paper/repair_budget/tau2_pilot 下）：bash setup_check_tau.sh 2>&1 | tee setup_check_tau.log
echo "== 时间"; TZ=Asia/Shanghai date
echo "== 主机"; hostname; uname -r
echo "== GPU"; nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv 2>&1 | head -5
echo "== 磁盘"; df -h / /root/autodl-tmp 2>&1
echo "== 内存 / CPU"; free -g | head -2; nproc
echo "== Python 与 conda"; which python python3 conda 2>&1; python3 --version 2>&1
conda env list 2>&1 | head -20
echo "== 已有模型目录"; ls /root/autodl-tmp/models 2>&1 | head -20
echo "== vLLM / torch（当前 python）"; python3 -c "import torch, sys; print('torch', torch.__version__, 'cuda', torch.version.cuda)" 2>&1 | tail -1
python3 -c "import vllm; print('vllm', vllm.__version__)" 2>&1 | tail -1
echo "== 网络（各 10 秒超时）"
for u in https://github.com https://pypi.org/simple/ https://www.modelscope.cn https://huggingface.co https://maas.qianwenaiapi.com; do
  code=$(curl -s -o /dev/null -m 10 -w "%{http_code}" "$u"); echo "$u -> $code"
done
echo "== 学术加速脚本"; ls /etc/network_turbo 2>&1
echo "== QWEN_API_KEY 是否已设置（不打印值）"; if [ -n "$QWEN_API_KEY" ]; then echo set; else echo unset; fi
echo "== 结束"
