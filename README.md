# SGLang
<img width="1918" height="1124" alt="屏幕截图 2026-09-27 140713" src="https://github.com/user-attachments/assets/ba74f169-7fc3-4318-85f1-c556083677e2" />
<img width="1905" height="1121" alt="屏幕截图 2026-09-27 125419" src="https://github.com/user-attachments/assets/8a1242b4-7f2f-48c5-801b-fc0a926a6ae9" />
wsl --status检查是否安装了WSL，然后将其指定到相应盘
wsl --install -d Ubuntu-24.04 --location D:\WSL\Ubuntu-24.04 --web-download
wsl -l -v验证安装位置
重启电脑后wsl -l -v验证是否安装完成，wsl --install -d Ubuntu-24.04 --location D:\WSL\Ubuntu-24.04 --web-download直接下载安装，可能需要挂梯子，完成后就输入用户名和密码
sudo apt update
sudo apt install -y python3-pip python3-venv安装pip
python3 -m venv ~/sglang-env
source ~/sglang-env/bin/activate
pip install --upgrade pip用虚拟环境安装SGLang
uv pip install --prerelease=allow sglang安装核心包（避免all拉太多依赖）
pip install ray==2.56.0安装ray2.56.0
nvidia-smi验证GPU是否可行

echo $HF_ENDPOINT确认镜像变量生效
python -m sglang.launch_server \
    --model-path Qwen/Qwen3-0.6B \
    --host 0.0.0.0 \
    --port 30000启动SGLang服务

wsl -d Ubuntu-24.04新建一个Ubuntu小黑屏并执行该语句

*卡点1：CUDA_HOME未设置：导致SGLang启动时加载deep_gemm失败，而Qwen3-0.6B很小，不需要
SGLANG_ENABLE_JIT_DEEPGEMM=0 python -m sglang.launch_server \
    --model-path Qwen/Qwen3-0.6B \
    --host 0.0.0.0 \
    --port 30000
    然而强制禁用后也没有效果，仍旧显示断言错误，因此最直接的修复是安装CUDA Toolkit
    sudo apt install -y cuda-toolkit-12-8
export CUDA_HOME=/usr/local/cuda-12.8
export PATH=$CUDA_HOME/bin:$PATH
然后nvcc --version确认是否安装好，但是发现无法安装问题
卡点2：安装CUDA Toolkit
sudo apt-key del 7fa2af80清除可能的残留的错误key
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda-toolkit-12-8  添加WSL专用CUDA仓库并安装

echo 'export PATH=/usr/local/cuda-12.8/bin:$PATH' >> ~/.bashrc
echo 'export CUDA_HOME=/usr/local/cuda-12.8' >> ~/.bashrc
source ~/.bashrc配置环境变量

nvcc --version验证

echo $CUDA_HOME
which nvcc确认环境变量在当前窗口是否生效

回归虚拟环境source ~/sglang-env/bin/activate
执行
python -m sglang.launch_server \
    --model-path Qwen/Qwen3-0.6B \
    --host 0.0.0.0 \
    --port 30000
仍旧报错，报错原因：要求CUDA>=12.9，因此需要进行更新处理
sudo apt update
sudo apt install -y cuda-toolkit-12-9

更新环境变量
sed -i 's/cuda-12.8/cuda-12.9/g' ~/.bashrc
source ~/.bashrc

验证是否成功
nvcc --version

最后启动SGLang
python -m sglang.launch_server \
    --model-path Qwen/Qwen3-0.6B \
    --host 0.0.0.0 \
    --port 30000
发现仍然报错，需要禁用Xet,强制回退普通HTTP下载
HF_HUB_DISABLE_XET=1 python -m sglang.launch_server \
    --model-path Qwen/Qwen3-0.6B \
    --host 0.0.0.0 \
    --port 30000
运行成功，保持当前页面
新开一个Ubuntu窗口
# 验证 /v1/models
curl http://localhost:30000/v1/models验证，发现JSON返回模型版本————成功

curl -X POST "http://localhost:30000/v1/chat/completions" \
  -H "Content-Type: application/json" \
  --data '{
    "model": "Qwen/Qwen3-0.6B",
    "messages": [{"role": "user", "content": "What is the capital of France?"}],
    "max_tokens": 32
  }'
  发送推理，输出choices[0].message.content则成功

  https://github.com/kvcache-ai/Mooncake下载mooncake_trace.jsonl文件，去到github然后搜索框搜索

  在文件资源管理器里搜索\\wsl$\Ubuntu-24.04\home\windows
  将下载下来的mooncake_trace.jsonl文件拖进来
  ls -lh ~/mooncake_trace.jsonl验证文件位置是否正确
  head -3 ~/mooncake_trace.jsonl确认字段名，作为依据才能写采样脚本

  source ~/sglang-env/bin/activate
pip list | grep -E "requests|transformers|numpy"确认是否有这三样东西

curl http://localhost:30000/v1/models确认服务还在运行

nano ~/mooncake_bench.py在自己建立的虚拟环境中创建一个文件，然后粘贴下面的代码
#!/usr/bin/env python3
"""
Mooncake trace 采样 + Poisson 到达 + SGLang OpenAI-compatible 推理
记录 input tokens / output tokens / status / TTFT / latency
"""

import json
import time
import random
import sys
import requests
import numpy as np
from transformers import AutoTokenizer

# ============ 可调参数 ============
TRACE_PATH = "/home/windows/mooncake_trace.jsonl"
BASE_URL = "http://localhost:30000"
MODEL = "Qwen/Qwen3-0.6B"
NUM_REQUESTS = 20              # 采样条数 (10-30)
POISSON_MEAN_INTERVAL = 0.5    # Poisson 平均到达间隔(秒)
MAX_NEW_TOKENS = 64            # 每条最多生成 token 数
TOKENS_PER_HASH = 500          # 每个 hash_id 估算的 token 数
RANDOM_SEED = 42
# =================================

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# ---------- 1. 读 trace 并采样 ----------
def load_trace(path, n):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    print(f"[trace] 总条数: {len(records)}")
    if len(records) > n:
        sampled = random.sample(records, n)
    else:
        sampled = records
    print(f"[trace] 采样条数: {len(sampled)}")
    return sampled

# ---------- 2. 构造 synthetic prompt ----------
def build_prompt(record, tokenizer):
    """根据 hash_ids 和 input_length 构造 prompt，尽量让 token 数接近 input_length"""
    target_len = record["input_length"]
    hash_ids = record.get("hash_ids", [])
    # 每个 hash_id 对应一块文本，用重复的短语填充
    # 让每块大致等于 TOKENS_PER_HASH 个 token
    # 先用一个短句测出它有多少 token，再决定重复次数
    filler_sentence = "The quick brown fox jumps over the lazy dog. "
    filler_tokens = len(tokenizer.encode(filler_sentence, add_special_tokens=False))
    repeat_per_hash = max(1, TOKENS_PER_HASH // max(1, filler_tokens))

    parts = []
    for hid in hash_ids:
        # 给每个 hash_id 一个稳定的前缀，方便 RadixCache 命中
        parts.append(f"[block-{hid}] ")
        parts.append(filler_sentence * repeat_per_hash)
    prompt = "".join(parts)

    # 按目标长度截断/补齐
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    if len(ids) > target_len:
        ids = ids[:target_len]
    elif len(ids) < target_len:
        # 补齐
        pad_id = tokenizer.encode(" data", add_special_tokens=False)
        while len(ids) < target_len and pad_id:
            need = target_len - len(ids)
            ids.extend(pad_id[:need])
    prompt = tokenizer.decode(ids, skip_special_tokens=True)
    return prompt, len(ids)

# ---------- 3. 发送流式请求，测 TTFT ----------
def send_request(prompt, max_new_tokens):
    url = f"{BASE_URL}/v1/chat/completions"
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_new_tokens,
        "stream": True,
        "temperature": 0.0,
    }
    start = time.time()
    ttft = None
    completion_tokens = 0
    status = None
    try:
        with requests.post(url, json=payload, stream=True, timeout=300) as r:
            status = r.status_code
            if status != 200:
                return {
                    "status": status,
                    "ttft": None,
                    "latency": time.time() - start,
                    "output_tokens": 0,
                }
            for line in r.iter_lines(decode_unicode=True):
                if not line:
                    continue
                if line.startswith("data: "):
                    data = line[6:]
                    if data.strip() == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    choices = chunk.get("choices", [])
                    if choices:
                        delta = choices[0].get("delta", {})
                        content = delta.get("content")
                        if content:
                            if ttft is None:
                                ttft = time.time() - start
                            completion_tokens += 1
        latency = time.time() - start
        return {
            "status": status,
            "ttft": ttft,
            "latency": latency,
            "output_tokens": completion_tokens,
        }
    except Exception as e:
        return {
            "status": f"ERR:{type(e).__name__}",
            "ttft": None,
            "latency": time.time() - start,
            "output_tokens": 0,
        }

# ---------- 4. 主流程 ----------
def main():
    print(f"[init] 加载 tokenizer: {MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)

    records = load_trace(TRACE_PATH, NUM_REQUESTS)

    # 先构造好所有 prompt
    jobs = []
    for rec in records:
        prompt, n_tokens = build_prompt(rec, tokenizer)
        jobs.append({
            "prompt": prompt,
            "input_tokens": n_tokens,
            "expected_output": rec["output_length"],
        })

    print(f"\n[bench] 开始发送 {len(jobs)} 条请求, Poisson 平均间隔 {POISSON_MEAN_INTERVAL}s\n")
    print(f"{'#':>3} | {'in_tok':>7} | {'out_tok':>7} | {'status':>6} | {'TTFT(s)':>8} | {'latency(s)':>10}")
    print("-" * 62)

    results = []
    t0 = time.time()
    for i, job in enumerate(jobs):
        # Poisson 到达间隔
        if i > 0:
            interval = np.random.poisson(POISSON_MEAN_INTERVAL * 1000) / 1000.0
            time.sleep(interval)

        res = send_request(job["prompt"], MAX_NEW_TOKENS)
        row = {
            "index": i,
            "input_tokens": job["input_tokens"],
            "output_tokens": res["output_tokens"],
            "status": res["status"],
            "ttft": res["ttft"],
            "latency": res["latency"],
        }
        results.append(row)

        ttft_s = f"{res['ttft']:.3f}" if res["ttft"] is not None else "N/A"
        print(f"{i:>3} | {job['input_tokens']:>7} | {res['output_tokens']:>7} | "
              f"{str(res['status']):>6} | {ttft_s:>8} | {res['latency']:>10.3f}")

    total_time = time.time() - t0

    # ---------- 5. 汇总 ----------
    ok = [r for r in results if r["status"] == 200]
    print("\n" + "=" * 62)
    print("汇总")
    print("=" * 62)
    print(f"总请求数        : {len(results)}")
    print(f"成功请求数      : {len(ok)}")
    print(f"总耗时          : {total_time:.2f} s")
    if ok:
        avg_ttft = sum(r["ttft"] for r in ok if r["ttft"] is not None) / len(ok)
        avg_lat = sum(r["latency"] for r in ok) / len(ok)
        total_in = sum(r["input_tokens"] for r in ok)
        total_out = sum(r["output_tokens"] for r in ok)
        print(f"平均 TTFT       : {avg_ttft:.3f} s")
        print(f"平均 latency    : {avg_lat:.3f} s")
        print(f"总 input tokens : {total_in}")
        print(f"总 output tokens: {total_out}")

    # 保存结果到文件，方便复查
    out_path = "/home/windows/mooncake_bench_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存到: {out_path}")

if __name__ == "__main__":
    main()

ctrl O 保存，enter确认文件名,ctrl X退出
ls -lh ~/mooncake_bench.py
head -5 ~/mooncake_bench.py确认文件写好

source ~/sglang-env/bin/activate
python ~/mooncake_bench.py
进入虚拟环境并运行脚本，采样完成

<img width="1212" height="507" alt="image" src="https://github.com/user-attachments/assets/1764f748-325c-429d-86b7-23812fcc0a88" />

RadixAttention/RadixCache解决的问题：
多轮对话可能经常包含相同的前缀，没有缓存时每次请求都要重新计算前缀的KV,浪费算力，而RadixCache用基数树存储token序列到KV物理页映射，新请求可复用已缓存的KV，降低算力消耗

page-sized KV cache 与 prefix reuse 分别对应 pipeline 哪一部分？
前者对应KV slot的物理分配，每个token的KV写入固定大小的page,由req_to_token_pool维护逻辑到物理的映射
后者用于决定计算分配给prefill的token，跳过重复匹配的前缀

vLLM 的 PagedAttention 解决什么问题？与 SGLang 的联系与差异？
PagedAttention解决：KV cache内存碎片浪费。通过将KV切成固定大小block按需分配，block table做逻辑物理映射，内存浪费控制在4%以下
联系：二者都是分页管理，支持block粒度共享
差异：pagedattention核心是内存管理，radixattention核心是前缀发现与复用调度，vLLM前缀缓存基于block hash，相对简单；SGLang用基数树做结构化匹配，在复杂场景中命中率更高

作业感受：
1.大部分依赖DeepSeek指导实现，包括安装环境、模型推理等
2.启动SGLang部分，CUDA_HOME未设置：导致SGLang启动时加载deep_gemm失败，强制停止后也无济于事，要求CUDA>=12.9，因此需要进行更新处理，发现仍然报错，需要禁用Xet,强制回退普通HTTP下载
3.各个方面，包括但不限于接受和面对各种各样的名词，理解技术的本质，还要能够动手复刻出来并观察结果；对我的启发：必须先动手实践，满足自己的好奇心
AI的使用情况：
使用的是DeepSeek,包括的提示词有一堆报错信息。AI通过给我解释各种报错背后的原因，提供给我现成的操作代码来帮助我快速推进实践进程。本次实践在ai的帮助下异常顺利。
