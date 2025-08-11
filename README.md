[AIFM 8.2.1] | [fig9]
# Introduction
此 Benchmark 是一个基于 C++ 标准库的性能测试工具，用于评估哈希表在 memcached-style 的工作负载下的性能表现，支持多种请求分布（uniform、zipfian、latest）模式。
Intro
- Provide unordered maps
- Random accesses
- High temporal locality
- Remote ones benefit from caching of popular KV pairs in local memory
Experiment Settings
Workload：YCSB workload(a-d)，记录数和操作数均为128M（10GB data）
性能指标：[time cost/throughout]
实验流程：
1. 调整远内存与本地内存的占比
local
12.5%
25%
50%
75%
100%
remote
87.5%
75%
50%
25%
0%
2. 调整 zipf 参数
zipf_s_arr=(0 0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5 0.55 0.6 0.65 0.7 0.75 0.8 0.85 0.9 0.95 1 1.05 1.1 1.15 1.2 1.25 1.3 1.35)
YCSB
1. 键生成：
insertorder：支持"hashed"和"ordered"，FNV哈希函数插入/顺序插入
2. 值生成：
fieldcount：记录中的字段数量（默认：10）
fieldlength：每个字段的大小（默认：100字节）
fieldlengthdistribution：字段长度分布，支持 "uniform"、"zipfian"、"constant"、"histogram"
3. 分布模式：
uniform（默认）：每个记录有相等的被访问概率
zipfian：某些记录有更高的被访问概率（热点数据）
latest：倾向于访问最近插入的数据
4. 两阶段执行：Load/Run
Benchmark & Manual
ycsb_hashtable_benchmark.cpp
操作指南
1. 将上述cpp文件和所有workload文件放置在同一目录下，如：
[图片]

---
2. 直接运行
# 运行
chmod +x ./run_ycsb_compatible.sh
./run_ycsb_compatible.sh -t hashtable

# AIFM 版本
./run_ycsb_compatible.sh -t aifm

---
3. 命令行参数

./run_ycsb_compatible.sh [选项] [工作负载文件...]
暂时无法在飞书文档外展示此内容
4. 示例：
# 基础用法
./run_ycsb_compatible.sh -t hashtable workloada.txt

# 创建样本并运行
./run_ycsb_compatible.sh -t hashtable -s

# 指定输出文件
./run_ycsb_compatible.sh -t hashtable -o my_results.txt workload*.txt

# 运行AIFM版本
./run_ycsb_compatible.sh -t aifm workloada.txt workloadb.txt
Workload 配置
配置文件格式
每个工作负载配置文件使用简单的键值对格式：
# 基础配置
recordcount=50000000          # 初始记录数量
operationcount=50000000       # 执行操作数量

# 操作比例（总和应为 1.0）
readproportion=0.5         # 读操作比例
updateproportion=0.4       # 更新操作比例
insertproportion=0.1       # 插入操作比例

# 请求分布模式
requestdistribution=zipfian    # uniform, zipfian, latest

# 扫描相关配置
maxscanlength=100          # 最大扫描长度
scanlengthdistribution=uniform  # 扫描长度分布
工作负载类型
YCSB官方提供的负载，其中workload e/f 中涉及 memcached 不支持的两类操作：SCAN/RMW，故删去。
1. Workload A: 更新密集型 (50% 读, 50% 更新)
readproportion=0.5
updateproportion=0.5
requestdistribution=zipfian
2. Workload B: 读密集型 (95% 读, 5% 更新)
readproportion=0.95
updateproportion=0.05
requestdistribution=zipfian
3. Workload C: 只读 (100% 读)
readproportion=1.0
requestdistribution=zipfian
4. Workload D: 读最新数据 (95% 读, 5% 插入)
readproportion=0.95
insertproportion=0.05
requestdistribution=latest
Results & Explanation
详细结果文件包含每个工作负载的完整性能分析：
YCSB Compatible Benchmark Results
Generated at: 2025-08-08 00:48:07
Command: ./ycsb_hashtable_compatible workloada.txt 
================================================================================

============================================================
YCSB Compatible Workload: workloada.txt
============================================================

=== YCSB Compatible Benchmark Results ===
Overall Performance:
  Total Operations: 10000000
  Successful Operations: 10000000
  Failed Operations: 0
  Success Rate: 100.00%
  Total Time: 136837.56 ms
  Throughput: 73079.35 ops/sec
  Average Latency: 0.01 ms/op

Operation Breakdown:
  READ Operations: 4998282
    Average Latency: 0.00 ms/op
  UPDATE Operations: 5001718
    Average Latency: 0.03 ms/op

Final HashTable Size: 10000000

================================================================================
YCSB COMPATIBLE BENCHMARK SUMMARY
================================================================================
Overall Session Statistics:
  Total Workloads Tested: 1
  Total Session Time: 415.24 seconds
  Peak Memory Usage: 18934.67 MB (19389100 KB)
  Session Completed At: 2025-08-08 00:55:02

Tested Workloads:
  workloada.txt (records: 10000000, ops: 10000000, insertorder: hashed, fields: 10)

================================================================================
AIFM
Problems
1. log文件出现FAILURE
[图片]
- 线程在调用调度器之前没有正确地处理抢占状态（可能是忘记 preempt_enable()），导致进入调度逻辑时抢占状态不合法。
- FAILURE也会输出mops且与论文数据对得上。
main_fixed.cpp
[图片]
2. 上传修改后的 run.sh 报错&解决方法
sed -i 's/\r$//' run.sh
chmod +x run.sh
[图片]
3. 通过 SFTP 上传文件权限开放太大&解决方法
chmod 600 /root/.ssh/id_rsa
[图片]
Result
待更新
