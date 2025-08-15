25/08/13 更新benchmark：改为400线程+记录运行前200s的MOPS(每10s记录一次)

```Bash
g++ -O2 -std=c++17 -pthread -o cpp_ycsb_benchmark mops_benchmark.cpp
./cpp_ycsb_benchmark workloadb.txt
```

===========================================

此 Benchmark 是一个基于 C++ 标准库的性能测试工具，用于评估哈希表在 memcached-style 的工作负载下的性能表现，支持多种请求分布（uniform、zipfian、latest）模式。

# Intro

- Provide unordered maps
- Random accesses
- High temporal locality
- Remote ones benefit from caching of popular KV pairs in local memory

# Experiment Settings

Workload：YCSB workload(a-d)，记录数和操作数均为128M（10GB data）

性能指标：[**time cost**/throughout]

实验流程：

1. 调整远内存与本地内存的占比

| local  | 12.5% | 25%  | 50%  | 75%  | 100% |
| ------ | ----- | ---- | ---- | ---- | ---- |
| remote | 87.5% | 75%  | 50%  | 25%  | 0%   |

2. 调整 zipf 参数
固定远内存与本地内存占比 [暂定为25%local,75%remote]，在 workloada.txt 进行测试。
zipf_s_arr=(0 0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5 0.55 0.6 0.65 0.7 0.75 0.8 0.85 0.9 0.95 1 1.05 1.1 1.15 1.2 1.25 1.3 1.35)

# YCSB

1. 键生成：

`insertorder`：支持"hashed"和"ordered"，FNV哈希函数插入/顺序插入

2. 值生成：

`fieldcount`：记录中的字段数量（默认：2）

`fieldlength`：每个字段的大小（默认：15字节）

`fieldlengthdistribution`：字段长度分布，支持 "uniform"、"zipfian"、"constant"、"histogram"

3. 分布模式：

`uniform`（默认）：每个记录有相等的被访问概率

`zipfian`：某些记录有更高的被访问概率（热点数据）

`latest`：倾向于访问最近插入的数据

4. 两阶段执行：Load/Run

# Benchmark & Manual

[ycsb_hashtable_benchmark.cpp](https://jianmucloud.feishu.cn/wiki/M0nVwQuioiTBBRkTMmEccWWVnwg)

## 操作指南

1. 将上述cpp文件和所有workload文件放置在同一目录下，如：

![img](https://jianmucloud.feishu.cn/space/api/box/stream/download/asynccode/?code=NDBkMWZiNjI4NDk4OWMzYjhhYWYzYzQzYjI5MmI1MmRfdnpXQ0ZtY2wwUHQxbVlDVmwwb280R3VLYnBNVEhQQ0JfVG9rZW46UzdVTmJwcmNRb1JIbzB4V3NRZ2NFdmhTbk5nXzE3NTQ4ODg1OTE6MTc1NDg5MjE5MV9WNA)

2. 直接运行

```Bash
# 运行
chmod +x ./run_ycsb_compatible.sh
./run_ycsb_compatible.sh -t hashtable

# AIFM 版本
./run_ycsb_compatible.sh -t aifm
```

3. 命令行参数

```Bash
./run_ycsb_compatible.sh [选项] [工作负载文件...]
```

| 参数      | 短参数 | 描述                             | 必需 | 默认值                      |
| --------- | ------ | -------------------------------- | ---- | --------------------------- |
| --type    | -t     | benchmark类型：hashtable 或 aifm | 是   | 无                          |
| --output  | -o     | 结果输出文件路径                 | 否   | ycsb_compatible_results.txt |
| --samples | -s     | 创建样本YCSB工作负载文件         | 否   | FALSE                       |
| --help    | -h     | 显示帮助信息                     | 否   | -                           |

4. 示例：

```Bash
# 基础用法
./run_ycsb_compatible.sh -t hashtable workloada.txt

# 创建样本并运行
./run_ycsb_compatible.sh -t hashtable -s

# 指定输出文件
./run_ycsb_compatible.sh -t hashtable -o my_results.txt workload*.txt

# 运行AIFM版本
./run_ycsb_compatible.sh -t aifm workloada.txt workloadb.txt
```

## Workload 配置

### 配置文件格式

每个工作负载配置文件使用简单的键值对格式：

```Plain
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
```

### 工作负载类型

YCSB官方提供的负载，其中workload e/f 中涉及 memcached 不支持的两类操作：SCAN/RMW，故删去。

1. Workload A: 更新密集型 (50% 读, 50% 更新)

```Plain
readproportion=0.5
updateproportion=0.5
requestdistribution=zipfian
```

2. Workload B: 读密集型 (95% 读, 5% 更新)

```Plain
readproportion=0.95
updateproportion=0.05
requestdistribution=zipfian
```

3. Workload C: 只读 (100% 读)

```Plain
readproportion=1.0
requestdistribution=zipfian
```

4. Workload D: 读最新数据 (95% 读, 5% 插入)

```Plain
readproportion=0.95
insertproportion=0.05
requestdistribution=latest
```

## Results & Explanation

详细结果文件包含每个工作负载的完整性能分析：

```Plain
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
```

# AIFM

## Problems

1. log文件出现FAILURE

![img](https://jianmucloud.feishu.cn/space/api/box/stream/download/asynccode/?code=ZmVjNjc0ODgwZDU5N2RhOGM4OTZmNzlhNTAwNTQyNGRfNzZja0hVS0xxNzJEZHh6ZGNsOUtGcDRSTW1yUTZNYm1fVG9rZW46V25LemIyTDRWb3pIZWd4MkthSGNkblNobkRkXzE3NTQ4ODg1OTE6MTc1NDg5MjE5MV9WNA)

- 线程在调用调度器之前没有正确地处理抢占状态（可能是忘记 `preempt_enable()`），导致进入调度逻辑时抢占状态不合法。
- FAILURE也会输出mops且与论文数据对得上。

main_fixed.cpp：

![img](https://jianmucloud.feishu.cn/space/api/box/stream/download/asynccode/?code=YWM0OGRmZDE3YzMzZTFiMWYzZmFkYjg1NDAxMjIxOWFfeXlybUhzU0gxc255dmJuNk91a2FCeDlCQzRjaFNHNUZfVG9rZW46UTYxWmJMdEY5b2M4TGt4eFk3RGNNNEpzbkFnXzE3NTQ4ODg1OTE6MTc1NDg5MjE5MV9WNA)

2. 上传修改后的 run.sh 报错&解决方法

```Shell
sed -i 's/\r$//' run.sh
chmod +x run.sh
```

![img](https://jianmucloud.feishu.cn/space/api/box/stream/download/asynccode/?code=NjY2YjFjOTYxYTI4NDNhZDNiY2FkMGY4NDU1NDZkOWVfU0lkYWlQT3ZMcExYNExXRHNlRGZvdDR6b3ljS3d0UW1fVG9rZW46UmpzTGIyZWVNb0cxM2F4T29QZ2NrNGJwbllkXzE3NTQ4ODg1OTE6MTc1NDg5MjE5MV9WNA)

3. 通过 SFTP 上传文件权限开放太大&解决方法

```Shell
chmod 600 /root/.ssh/id_rsa
```

![img](https://jianmucloud.feishu.cn/space/api/box/stream/download/asynccode/?code=YTczNWM3ZmQ5OGQwZjg0ODlkNTNmZTNhZDVlMWYwY2ZfQmtvTVNQQ2p2OFN1Y0FISlNpTXFLV3M1aGRqWWxVMVhfVG9rZW46SURFWmJxZ0VVb2lHV0t4ZWNBZ2NpaUl2bk15XzE3NTQ4ODg1OTE6MTc1NDg5MjE5MV9WNA)

## Result


待更新




