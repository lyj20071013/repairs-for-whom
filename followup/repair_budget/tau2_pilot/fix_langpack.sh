#!/usr/bin/env bash
# tau 服务器：给已安装的 τ²-bench 补上英文语言包（2026-10-29 写；无卡即可）。
# 原因：trim 版源码包去掉了 data/tau2/multilingual，tau2 run 报 "no published language pack for 'en'"。
# 只解压 multilingual 目录到已有的源码树；tau2 是 pip install -e 装的，不用重装。
set -u
HERE=/root/paper/repair_budget/tau2_pilot
T=$HERE/tau2-bench-122e243-trim2.tar.gz
[ -f "$T" ] || { echo "[停下] 找不到 $T（先用 sync.py push）"; exit 2; }
got=$(md5sum < "$T" | cut -d' ' -f1); [ "$got" = 97b3d4f644b101954f73a397ec7c00cf ] || { echo "[停下] md5 不对：$got"; exit 2; }
tar -xzf "$T" -C "$HERE" tau2-bench/data/tau2/multilingual || { echo "[停下] 解压失败"; exit 1; }
ls "$HERE/tau2-bench/data/tau2/multilingual/en/pack.yaml" || { echo "[停下] 解压后仍没有 en/pack.yaml"; exit 1; }
/root/autodl-tmp/envs/tau/bin/python -c "from tau2.multilingual.runtime import load_pack; print('英文语言包：', load_pack('en').language)" || exit 1
echo "== 完成"
