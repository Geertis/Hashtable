#!/bin/bash

# YCSB Compatible Benchmark Runner
# 支持运行YCSB兼容的benchmark和AIFM版本

# 导入AIFM环境
if [ -f "../../../shared.sh" ]; then
    source ../../../shared.sh
else
    echo "Warning: shared.sh not found"
fi

# 命令行参数解析
BENCHMARK_TYPE=""
OUTPUT_FILE=""
WORKLOAD_FILES=()
CREATE_SAMPLES=false
AIFM_MODE=false

show_help() {
    echo "YCSB Compatible Benchmark Runner"
    echo "Usage: $0 [options] [workload_files...]"
    echo ""
    echo "Options:"
    echo "  -t, --type <type>      Benchmark type: 'hashtable' or 'aifm'"
    echo "  -o, --output <file>    Output file for results"
    echo "  -s, --samples          Create sample YCSB workload files"
    echo "  -h, --help            Show this help"
    echo ""
    echo "Examples:"
    echo "  $0 -t hashtable -s                           # Run hashtable benchmark with sample workloads"
    echo "  $0 -t aifm workloada_ycsb.txt               # Run AIFM with specific workload"
    echo "  $0 -t hashtable -o results.txt *.txt        # Run hashtable with all .txt workloads"
    echo ""
    echo "YCSB Compatible Features:"
    echo "  ✓ FNV hash key generation (insertorder=hashed/ordered)"
    echo "  ✓ Multi-field records with configurable count/length"
    echo "  ✓ Field length distributions: constant, uniform, zipfian"
    echo "  ✓ Request distributions: uniform, zipfian, latest, sequential"
    echo "  ✓ Zero padding support for ordered keys"
    echo "  ✓ Read/Write all fields vs single field options"
    echo "  ✓ Scan operations with length distributions"
    echo "  ✓ Read-Modify-Write operations"
}

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        -t|--type)
            BENCHMARK_TYPE="$2"
            shift 2
            ;;
        -o|--output)
            OUTPUT_FILE="$2"
            shift 2
            ;;
        -s|--samples)
            CREATE_SAMPLES=true
            shift
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            WORKLOAD_FILES+=("$1")
            shift
            ;;
    esac
done

# 检查benchmark类型
if [ -z "$BENCHMARK_TYPE" ]; then
    echo "Please specify benchmark type with -t/--type (hashtable or aifm)"
    show_help
    exit 1
fi

if [ "$BENCHMARK_TYPE" != "hashtable" ] && [ "$BENCHMARK_TYPE" != "aifm" ]; then
    echo "Invalid benchmark type: $BENCHMARK_TYPE"
    echo "Must be 'hashtable' or 'aifm'"
    exit 1
fi

if [ "$BENCHMARK_TYPE" == "aifm" ]; then
    AIFM_MODE=true
fi

echo "YCSB Compatible Benchmark Runner"
echo "Benchmark type: $BENCHMARK_TYPE"

# 创建样本workload文件
create_sample_workloads() {
    echo "Creating sample YCSB compatible workload files..."
    
    # Workload A: Heavy read/update workload (50%/50%)
    cat > workloada.txt << 'EOF'
# YCSB Workload A: Heavy read/update workload
# Application example: Session store recording recent actions
recordcount=10000
operationcount=10000
fieldcount=10
fieldlength=100
fieldlengthdistribution=constant
fieldnameprefix=field
insertorder=hashed
zeropadding=1
readallfields=true
writeallfields=true
readproportion=0.5
updateproportion=0.5
insertproportion=0.0
scanproportion=0.0
readmodifywriteproportion=0.0
requestdistribution=zipfian
maxscanlength=100
minscanlength=1
scanlengthdistribution=uniform
insertstart=0
EOF

    # Workload B: Read mostly workload (95%/5%)
    cat > workloadb.txt << 'EOF'
# YCSB Workload B: Read mostly workload
# Application example: Photo tagging; add a tag is an update, but most operations are to read tags
recordcount=10000
operationcount=10000
fieldcount=10
fieldlength=100
fieldlengthdistribution=constant
fieldnameprefix=field
insertorder=hashed
zeropadding=1
readallfields=true
writeallfields=true
readproportion=0.95
updateproportion=0.05
insertproportion=0.0
scanproportion=0.0
readmodifywriteproportion=0.0
requestdistribution=zipfian
maxscanlength=100
minscanlength=1
scanlengthdistribution=uniform
insertstart=0
EOF

    # Workload C: Read only workload
    cat > workloadc.txt << 'EOF'
# YCSB Workload C: Read only
# Application example: User profile cache, where profiles are constructed elsewhere
recordcount=10000
operationcount=10000
fieldcount=10
fieldlength=100
fieldlengthdistribution=constant
fieldnameprefix=field
insertorder=hashed
zeropadding=1
readallfields=true
writeallfields=true
readproportion=1.0
updateproportion=0.0
insertproportion=0.0
scanproportion=0.0
readmodifywriteproportion=0.0
requestdistribution=zipfian
maxscanlength=100
minscanlength=1
scanlengthdistribution=uniform
insertstart=0
EOF

    # Workload D: Read latest workload
    cat > workloadd.txt << 'EOF'
# YCSB Workload D: Read latest workload
# Application example: User status updates; people want to read the latest
recordcount=10000
operationcount=10000
fieldcount=10
fieldlength=100
fieldlengthdistribution=constant
fieldnameprefix=field
insertorder=hashed
zeropadding=1
readallfields=true
writeallfields=true
readproportion=0.95
updateproportion=0.0
insertproportion=0.05
scanproportion=0.0
readmodifywriteproportion=0.0
requestdistribution=latest
maxscanlength=100
minscanlength=1
scanlengthdistribution=uniform
insertstart=0
EOF

    # Workload E: Short ranges (scan workload)
    cat > workloade.txt << 'EOF'
# YCSB Workload E: Short ranges
# Application example: Threaded conversations, where each scan is for posts in a thread
recordcount=10000
operationcount=10000
fieldcount=10
fieldlength=100
fieldlengthdistribution=constant
fieldnameprefix=field
insertorder=hashed
zeropadding=1
readallfields=true
writeallfields=true
readproportion=0.0
updateproportion=0.0
insertproportion=0.05
scanproportion=0.95
readmodifywriteproportion=0.0
requestdistribution=zipfian
maxscanlength=100
minscanlength=1
scanlengthdistribution=uniform
insertstart=0
EOF

    # Workload F: Read-modify-write workload
    cat > workloadf.txt << 'EOF'
# YCSB Workload F: Read-modify-write workload
# Application example: User database, where user records are read and modified
recordcount=10000
operationcount=10000
fieldcount=10
fieldlength=100
fieldlengthdistribution=constant
fieldnameprefix=field
insertorder=hashed
zeropadding=1
readallfields=true
writeallfields=true
readproportion=0.5
updateproportion=0.0
insertproportion=0.0
scanproportion=0.0
readmodifywriteproportion=0.5
requestdistribution=zipfian
maxscanlength=100
minscanlength=1
scanlengthdistribution=uniform
insertstart=0
EOF

    echo "Created sample workload files:"
    echo "  workloada.txt - Heavy read/update (50%/50%)"
    echo "  workloadb.txt - Read mostly (95%/5%)"
    echo "  workloadc.txt - Read only (100%)"
    echo "  workloadd.txt - Read latest (95% read, latest distribution)"
    echo "  workloade.txt - Short ranges (95% scan)"
    echo "  workloadf.txt - Read-modify-write (50%/50%)"
}

# 如果需要创建样本文件
if [ "$CREATE_SAMPLES" = true ]; then
    create_sample_workloads
fi

# 确定workload文件
if [ ${#WORKLOAD_FILES[@]} -eq 0 ]; then
    # 如果没有指定文件，查找现有的YCSB workload文件
    YCSB_FILES=(workload*.txt)
    if [ -f "${YCSB_FILES[0]}" ]; then
        WORKLOAD_FILES=("${YCSB_FILES[@]}")
        echo "Auto-detected YCSB workload files: ${WORKLOAD_FILES[*]}"
    else
        echo "No workload files found. Use -s to create samples or specify files."
        exit 1
    fi
fi

# 验证workload文件存在
for file in "${WORKLOAD_FILES[@]}"; do
    if [ ! -f "$file" ]; then
        echo "Workload file not found: $file"
        exit 1
    fi
done

echo "Using workload files: ${WORKLOAD_FILES[*]}"

# 选择合适的源文件和可执行文件名
if [ "$AIFM_MODE" = true ]; then
    SOURCE_FILE="ycsb_aifm_compatible.cpp"
    EXECUTABLE="ycsb_aifm_compatible"
    TEMP_MAIN="main.cpp"
else
    SOURCE_FILE="ycsb_hashtable_compatible.cpp"
    EXECUTABLE="ycsb_hashtable_compatible"
fi

# 编译
echo "Building $BENCHMARK_TYPE benchmark..."

if [ "$AIFM_MODE" = true ]; then
    # AIFM版本需要重命名为main.cpp
    if [ ! -f "$SOURCE_FILE" ]; then
        echo "Source file $SOURCE_FILE not found!"
        echo "Please create the YCSB compatible AIFM implementation first."
        exit 1
    fi
    
    cp "$SOURCE_FILE" "$TEMP_MAIN"
    make clean > /dev/null 2>&1
    make -j
    COMPILE_SUCCESS=$?
    rm -f "$TEMP_MAIN"
    EXECUTABLE="main"
else
    # HashTable版本
    if [ ! -f "$SOURCE_FILE" ]; then
        echo "Source file $SOURCE_FILE not found!"
        echo "Please create the YCSB compatible hashtable implementation first."
        exit 1
    fi
    
    g++ -std=c++17 -O3 -o "$EXECUTABLE" "$SOURCE_FILE"
    COMPILE_SUCCESS=$?
fi

if [ $COMPILE_SUCCESS -ne 0 ]; then
    echo "Build failed!"
    exit 1
fi

echo "Build successful!"

# 运行benchmark
if [ "$AIFM_MODE" = true ]; then
    # 启动AIFM服务器
    if command -v rerun_local_iokerneld &> /dev/null; then
        echo "Starting AIFM servers..."
        kill_local_iokerneld 2>/dev/null
        kill_mem_server 2>/dev/null
        rerun_local_iokerneld
        rerun_mem_server
        sleep 3
    fi

    # 检查可执行文件
    if [ ! -f "$EXECUTABLE" ]; then
        echo "Executable not found: $EXECUTABLE"
        ls -la
        exit 1
    fi

    # 从shared.sh读取服务器配置
    SERVER_IP=${MEM_SERVER_DPDK_IP:-18.18.1.3}
    SERVER_PORT=${MEM_SERVER_PORT:-8000}
    SERVER_ADDR="$SERVER_IP:$SERVER_PORT"

    echo "Running YCSB compatible AIFM benchmark..."
    echo "Connecting to: $SERVER_ADDR"
    
    if [ -n "$OUTPUT_FILE" ]; then
        echo "Results will be saved to: $OUTPUT_FILE"
    fi
    
    sudo ./"$EXECUTABLE" ../../../configs/client.config "$SERVER_ADDR" "${WORKLOAD_FILES[@]}"

    # 清理服务器
    if command -v kill_local_iokerneld &> /dev/null; then
        kill_local_iokerneld 2>/dev/null
        kill_mem_server 2>/dev/null
    fi

else
    # HashTable版本
    echo "Running YCSB compatible hashtable benchmark..."
    
    if [ -n "$OUTPUT_FILE" ]; then
        ./"$EXECUTABLE" --output "$OUTPUT_FILE" "${WORKLOAD_FILES[@]}"
    else
        ./"$EXECUTABLE" "${WORKLOAD_FILES[@]}"
    fi
fi

echo ""
echo "YCSB Compatible Benchmark completed!"
echo ""
echo "Key improvements over original benchmark:"
echo "✓ FNV hash key generation (same as YCSB)"
echo "✓ Support for insertorder=hashed (default) and ordered"
echo "✓ Multi-field records with configurable field count/length"
echo "✓ Field length distributions (constant, uniform, zipfian)"
echo "✓ Zero padding support for ordered keys"
echo "✓ Read/Write all fields vs single field options"
echo "✓ Request distributions: uniform, zipfian, latest, sequential"
if [ "$BENCHMARK_TYPE" != "aifm" ]; then
    echo "✓ Scan operations with configurable length distributions"
    echo "✓ Read-Modify-Write operations"
fi
echo ""
echo "Results demonstrate YCSB-compatible behavior and can be compared"
echo "with other YCSB benchmark results from different systems."