#include <iostream>
#include <unordered_map>
#include <fstream>
#include <string>
#include <vector>
#include <chrono>
#include <random>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <cstring>
#include <cmath>
#include <sys/resource.h>
#include <unistd.h>
#include <cstdint>
#include <thread>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <memory>

// 简单的barrier实现，用于替代C++20的std::barrier
class SimpleBarrier {
private:
    std::mutex mutex_;
    std::condition_variable cv_;
    std::size_t count_;
    std::size_t current_;
    std::size_t generation_;

public:
    explicit SimpleBarrier(std::size_t count)
        : count_(count), current_(0), generation_(0) {}

    void arrive_and_wait() {
        std::unique_lock<std::mutex> lock(mutex_);
        auto gen = generation_;
        if (++current_ == count_) {
            generation_++;
            current_ = 0;
            cv_.notify_all();
        }
        else {
            cv_.wait(lock, [this, gen] { return gen != generation_; });
        }
    }
};

// FNV Hash实现
class FNVHash {
private:
    static constexpr uint64_t FNV_OFFSET_BASIS_64 = 0xCBF29CE484222325ULL;
    static constexpr uint64_t FNV_PRIME_64 = 1099511628211ULL;

public:
    static uint64_t hash(uint64_t val) {
        uint64_t hashval = FNV_OFFSET_BASIS_64;
        for (int i = 0; i < 8; i++) {
            uint8_t octet = val & 0x00ff;
            val = val >> 8;
            hashval = hashval ^ octet;
            hashval = hashval * FNV_PRIME_64;
        }
        return hashval > 0 ? hashval : -hashval;
    }
};

// 全局统计结构
struct GlobalStats {
    std::chrono::high_resolution_clock::time_point start_time;
    std::chrono::high_resolution_clock::time_point end_time;
    size_t total_workloads = 0;
    double total_benchmark_time_ms = 0.0;
    size_t peak_memory_kb = 0;

    void startTiming() {
        start_time = std::chrono::high_resolution_clock::now();
    }

    void endTiming() {
        end_time = std::chrono::high_resolution_clock::now();
        total_benchmark_time_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    }

    void updatePeakMemory() {
        struct rusage usage;
        if (getrusage(RUSAGE_SELF, &usage) == 0) {
#ifdef __linux__
            peak_memory_kb = std::max(peak_memory_kb, static_cast<size_t>(usage.ru_maxrss));
#else
            peak_memory_kb = std::max(peak_memory_kb, static_cast<size_t>(usage.ru_maxrss / 1024));
#endif
        }
    }
};

// YCSB工作负载配置结构
struct WorkloadConfig {
    int64_t recordcount = 1000;
    std::string workload = "site.ycsb.workloads.CoreWorkload";

    // 字段配置
    int fieldcount = 2;
    int fieldlength = 15;
    int minfieldlength = 1;
    std::string fieldlengthdistribution = "constant";
    std::string fieldnameprefix = "field";

    // 操作比例
    bool readallfields = true;
    double readproportion = 0.0;
    double updateproportion = 0.0;
    double scanproportion = 0.0;
    double insertproportion = 0.0;
    double readmodifywriteproportion = 0.0;

    // 分布配置
    std::string requestdistribution = "uniform";
    std::string insertorder = "hashed";
    int maxscanlength = 1000;
    int minscanlength = 1;
    std::string scanlengthdistribution = "uniform";
    int zeropadding = 1;

    // 插入配置
    int64_t insertstart = 0;
    int64_t insertcount = -1;

    // 多线程配置 - 匹配原始AIFM的400线程
    int threadcount = 400;  // 默认400线程，匹配原始AIFM

    // 固定性能测试参数
    static constexpr int64_t kFixedRuntimeSeconds = 240; // 4分钟
    static constexpr int64_t kMonitorIntervalSeconds = 10; // 每10秒监控一次
};

// 记录结构
struct Record {
    std::string key;
    std::vector<std::pair<std::string, std::string>> fields;

    Record() = default;
    Record(const std::string& k) : key(k) {}
};

// 线程安全的HashTable类
class ThreadSafeHashTable {
private:
    std::unordered_map<std::string, Record> table;
    mutable std::mutex table_mutex;

public:
    bool insert(const std::string& key, const Record& record) {
        std::lock_guard<std::mutex> lock(table_mutex);
        table[key] = record;
        return true;
    }

    bool read(const std::string& key, Record& record, bool readAllFields = true) {
        std::lock_guard<std::mutex> lock(table_mutex);
        auto it = table.find(key);
        if (it != table.end()) {
            record = it->second;
            if (!readAllFields && !record.fields.empty()) {
                thread_local std::mt19937 gen(std::chrono::system_clock::now().time_since_epoch().count());
                std::uniform_int_distribution<size_t> dist(0, record.fields.size() - 1);
                size_t fieldIndex = dist(gen);

                Record singleFieldRecord(key);
                singleFieldRecord.fields.push_back(record.fields[fieldIndex]);
                record = singleFieldRecord;
            }
            return true;
        }
        return false;
    }

    bool update(const std::string& key, const Record& record, bool writeAllFields = true) {
        std::lock_guard<std::mutex> lock(table_mutex);
        auto it = table.find(key);
        if (it != table.end()) {
            if (writeAllFields) {
                it->second = record;
            }
            else {
                if (!record.fields.empty() && !it->second.fields.empty()) {
                    thread_local std::mt19937 gen(std::chrono::system_clock::now().time_since_epoch().count());
                    std::uniform_int_distribution<size_t> dist(0, it->second.fields.size() - 1);
                    size_t fieldIndex = dist(gen);
                    it->second.fields[fieldIndex] = record.fields[0];
                }
            }
            return true;
        }
        return false;
    }

    bool scan(const std::string& start_key, int count, std::vector<Record>& results) {
        std::lock_guard<std::mutex> lock(table_mutex);
        results.clear();
        int scanned = 0;

        for (auto it = table.begin(); it != table.end() && scanned < count; ++it) {
            if (it->first >= start_key) {
                results.push_back(it->second);
                ++scanned;
            }
        }
        return !results.empty();
    }

    size_t size() const {
        std::lock_guard<std::mutex> lock(table_mutex);
        return table.size();
    }

    void clear() {
        std::lock_guard<std::mutex> lock(table_mutex);
        table.clear();
    }
};

// 分布式随机数生成器
class DistributionGenerator {
private:
    std::mt19937 gen;
    std::uniform_real_distribution<double> uniform_dist{ 0.0, 1.0 };

    class zipf_distribution {
    private:
        std::vector<double> probabilities;
        std::discrete_distribution<int> discrete_dist;

    public:
        zipf_distribution(int n = 1000, double s = 0.99) {
            probabilities.resize(n);
            double sum = 0.0;

            for (int i = 1; i <= n; ++i) {
                probabilities[i - 1] = 1.0 / std::pow(i, s);
                sum += probabilities[i - 1];
            }

            for (auto& p : probabilities) {
                p /= sum;
            }

            discrete_dist = std::discrete_distribution<int>(probabilities.begin(), probabilities.end());
        }

        template<class Generator>
        int operator()(Generator& g) {
            return discrete_dist(g);
        }
    };

    zipf_distribution zipf_gen{ 1000, 0.99 };

public:
    DistributionGenerator() : gen(std::random_device{}()) {}

    int64_t generateKey(const std::string& distribution, int64_t range) {
        if (distribution == "zipfian") {
            return zipf_gen(gen) % range;
        }
        else if (distribution == "latest") {
            std::exponential_distribution<double> exp_dist(2.0);
            int64_t offset = static_cast<int64_t>(exp_dist(gen));
            return std::max(static_cast<int64_t>(0), range - 1 - offset % range);
        }
        else if (distribution == "sequential") {
            static std::atomic<int64_t> seq_counter{ 0 };
            return (seq_counter++) % range;
        }
        else {
            std::uniform_int_distribution<int64_t> dist(0, range - 1);
            return dist(gen);
        }
    }

    double generateDouble() {
        return uniform_dist(gen);
    }

    int generateScanLength(const std::string& distribution, int minLength, int maxLength) {
        if (distribution == "uniform") {
            return std::uniform_int_distribution<int>(minLength, maxLength)(gen);
        }
        else if (distribution == "zipfian") {
            zipf_distribution zipf_scan(maxLength - minLength + 1, 0.99);
            return minLength + zipf_scan(gen);
        }
        else {
            return minLength;
        }
    }

    int generateFieldLength(const std::string& distribution, int minLength, int maxLength) {
        if (distribution == "uniform") {
            return std::uniform_int_distribution<int>(minLength, maxLength)(gen);
        }
        else if (distribution == "zipfian") {
            zipf_distribution zipf_field(maxLength - minLength + 1, 0.99);
            return minLength + zipf_field(gen);
        }
        else {
            return maxLength;
        }
    }
};

// 键生成器
class KeyGenerator {
private:
    std::string insertOrder;
    int zeroPadding;

public:
    KeyGenerator(const std::string& order = "hashed", int padding = 1)
        : insertOrder(order), zeroPadding(padding) {}

    std::string generateKey(int64_t keynum) {
        std::string keyStr;

        if (insertOrder == "hashed") {
            uint64_t hashedKey = FNVHash::hash(static_cast<uint64_t>(keynum));
            keyStr = "user" + std::to_string(hashedKey);
        }
        else {
            if (zeroPadding > 1) {
                std::ostringstream oss;
                oss << "user" << std::setfill('0') << std::setw(zeroPadding) << keynum;
                keyStr = oss.str();
            }
            else {
                keyStr = "user" + std::to_string(keynum);
            }
        }

        return keyStr;
    }
};

// 工作负载配置解析器
class WorkloadParser {
public:
    static WorkloadConfig parseConfig(const std::string& filename) {
        WorkloadConfig config;
        std::ifstream file(filename);

        if (!file.is_open()) {
            std::cerr << "Warning: Cannot open " << filename << ", using default config" << std::endl;
            return config;
        }

        std::string line;
        while (std::getline(file, line)) {
            line.erase(std::remove_if(line.begin(), line.end(), ::isspace), line.end());

            if (line.empty() || line[0] == '#') continue;

            auto pos = line.find('=');
            if (pos == std::string::npos) continue;

            std::string key = line.substr(0, pos);
            std::string value = line.substr(pos + 1);

            if (key == "recordcount") config.recordcount = std::stoll(value);
            else if (key == "workload") config.workload = value;
            else if (key == "fieldcount") config.fieldcount = std::stoi(value);
            else if (key == "fieldlength") config.fieldlength = std::stoi(value);
            else if (key == "minfieldlength") config.minfieldlength = std::stoi(value);
            else if (key == "fieldlengthdistribution") config.fieldlengthdistribution = value;
            else if (key == "fieldnameprefix") config.fieldnameprefix = value;
            else if (key == "readallfields") config.readallfields = (value == "true");
            else if (key == "readproportion") config.readproportion = std::stod(value);
            else if (key == "updateproportion") config.updateproportion = std::stod(value);
            else if (key == "scanproportion") config.scanproportion = std::stod(value);
            else if (key == "insertproportion") config.insertproportion = std::stod(value);
            else if (key == "readmodifywriteproportion") config.readmodifywriteproportion = std::stod(value);
            else if (key == "requestdistribution") config.requestdistribution = value;
            else if (key == "insertorder") config.insertorder = value;
            else if (key == "maxscanlength") config.maxscanlength = std::stoi(value);
            else if (key == "minscanlength") config.minscanlength = std::stoi(value);
            else if (key == "scanlengthdistribution") config.scanlengthdistribution = value;
            else if (key == "zeropadding") config.zeropadding = std::stoi(value);
            else if (key == "insertstart") config.insertstart = std::stoll(value);
            else if (key == "insertcount") config.insertcount = std::stoll(value);
            else if (key == "threadcount") config.threadcount = std::stoi(value);
        }

        if (config.insertcount == -1) {
            config.insertcount = config.recordcount;
        }

        return config;
    }
};

// 多线程YCSB兼容基准测试器 - 专注于MOPS吞吐量测量
class MultiThreadedYCSBBenchmark {
private:
    ThreadSafeHashTable hashtable;
    std::string output_filename;
    std::atomic<bool> should_stop{ false };
    std::atomic<int64_t> global_insert_counter;

    // 性能监控相关
    std::atomic<int64_t> total_operations{ 0 };
    std::vector<double> mops_measurements;
    std::mutex mops_mutex;

    std::string generateValue(const WorkloadConfig& config, const std::string& fieldName, DistributionGenerator& gen) {
        int length = gen.generateFieldLength(
            config.fieldlengthdistribution,
            config.minfieldlength,
            config.fieldlength
        );

        const std::string chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
        std::string result;
        result.reserve(length);

        for (int i = 0; i < length; ++i) {
            result += chars[static_cast<size_t>(gen.generateDouble() * chars.length())];
        }

        return result;
    }

    Record generateRecord(const std::string& key, const WorkloadConfig& config, DistributionGenerator& gen) {
        Record record(key);

        for (int i = 0; i < config.fieldcount; ++i) {
            std::string fieldName = config.fieldnameprefix + std::to_string(i);
            std::string fieldValue = generateValue(config, fieldName, gen);
            record.fields.push_back({ fieldName, fieldValue });
        }

        return record;
    }

    void loadInitialDataWorker(const WorkloadConfig& config, int thread_id, int total_threads, SimpleBarrier& barrier) {
        KeyGenerator keyGen(config.insertorder, config.zeropadding);
        DistributionGenerator gen;

        int64_t records_per_thread = config.insertcount / total_threads;
        int64_t start_idx = config.insertstart + thread_id * records_per_thread;
        int64_t end_idx = start_idx + records_per_thread;

        if (thread_id == total_threads - 1) {
            end_idx = config.insertstart + config.insertcount;
        }

        for (int64_t i = start_idx; i < end_idx; ++i) {
            std::string key = keyGen.generateKey(i);
            Record record = generateRecord(key, config, gen);
            hashtable.insert(key, record);
        }

        barrier.arrive_and_wait();
    }

    void loadInitialData(const WorkloadConfig& config) {
        std::cout << "Loading " << config.recordcount << " initial records using "
            << config.threadcount << " threads..." << std::endl;

        SimpleBarrier barrier(config.threadcount);
        std::vector<std::thread> threads;

        auto start_time = std::chrono::high_resolution_clock::now();

        for (int i = 0; i < config.threadcount; ++i) {
            threads.emplace_back(&MultiThreadedYCSBBenchmark::loadInitialDataWorker,
                this, std::ref(config), i, config.threadcount, std::ref(barrier));
        }

        for (auto& thread : threads) {
            thread.join();
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        auto load_time = std::chrono::duration<double>(end_time - start_time).count();

        std::cout << "Initial data loaded in " << std::fixed << std::setprecision(2)
            << load_time << " seconds" << std::endl;
        std::cout << "HashTable size: " << hashtable.size() << std::endl;
        std::cout << "Load throughput: " << std::fixed << std::setprecision(0)
            << (config.insertcount / load_time) << " ops/sec" << std::endl;
    }

    void benchmarkWorker(const WorkloadConfig& config, int thread_id) {
        DistributionGenerator gen;
        KeyGenerator keyGen(config.insertorder, config.zeropadding);

        while (!should_stop.load()) {
            double op_selector = gen.generateDouble();
            double cumulative = 0.0;

            if ((cumulative += config.readproportion) >= op_selector) {
                // READ操作
                int64_t keynum = config.insertstart + gen.generateKey(config.requestdistribution, config.insertcount);
                std::string key = keyGen.generateKey(keynum);
                Record record;
                hashtable.read(key, record, config.readallfields);

            }
            else if ((cumulative += config.updateproportion) >= op_selector) {
                // UPDATE操作
                int64_t keynum = config.insertstart + gen.generateKey(config.requestdistribution, config.insertcount);
                std::string key = keyGen.generateKey(keynum);
                Record record = generateRecord(key, config, gen);
                hashtable.update(key, record, true);

            }
            else if ((cumulative += config.insertproportion) >= op_selector) {
                // INSERT操作
                int64_t insert_keynum = config.insertstart + config.insertcount + global_insert_counter.fetch_add(1);
                std::string key = keyGen.generateKey(insert_keynum);
                Record record = generateRecord(key, config, gen);
                hashtable.insert(key, record);

            }
            else if ((cumulative += config.scanproportion) >= op_selector) {
                // SCAN操作
                int64_t keynum = config.insertstart + gen.generateKey(config.requestdistribution, config.insertcount);
                std::string start_key = keyGen.generateKey(keynum);
                int scan_length = gen.generateScanLength(
                    config.scanlengthdistribution,
                    config.minscanlength,
                    config.maxscanlength
                );
                std::vector<Record> results;
                hashtable.scan(start_key, scan_length, results);

            }
            else if ((cumulative += config.readmodifywriteproportion) >= op_selector) {
                // READ-MODIFY-WRITE操作
                int64_t keynum = config.insertstart + gen.generateKey(config.requestdistribution, config.insertcount);
                std::string key = keyGen.generateKey(keynum);
                Record old_record;
                if (hashtable.read(key, old_record, config.readallfields)) {
                    Record new_record = generateRecord(key, config, gen);
                    hashtable.update(key, new_record, true);
                }
            }

            total_operations.fetch_add(1);
        }
    }

    void performanceMonitor() {
        auto start_time = std::chrono::steady_clock::now();
        int64_t prev_operations = 0;
        int measurement_count = 0;

        std::ofstream outfile(output_filename, std::ios::app);
        if (outfile.is_open()) {
            outfile << "\n=== MOPS Performance Measurements ===\n";
            outfile << "Time(s)\tMOPS\n";
        }

        while (!should_stop.load()) {
            std::this_thread::sleep_for(std::chrono::seconds(WorkloadConfig::kMonitorIntervalSeconds));

            auto current_time = std::chrono::steady_clock::now();
            auto elapsed_seconds = std::chrono::duration<double>(current_time - start_time).count();

            int64_t current_operations = total_operations.load();
            int64_t ops_in_interval = current_operations - prev_operations;

            // 计算MOPS (Million Operations Per Second)
            double mops = (double)ops_in_interval / (WorkloadConfig::kMonitorIntervalSeconds * 1000000.0);

            {
                std::lock_guard<std::mutex> lock(mops_mutex);
                mops_measurements.push_back(mops);
            }

            measurement_count++;
            std::cout << std::fixed << std::setprecision(3);
            std::cout << "[" << std::setw(3) << (int)elapsed_seconds << "s] MOPS: " << mops << std::endl;

            if (outfile.is_open()) {
                outfile << std::fixed << std::setprecision(1) << elapsed_seconds << "\t"
                    << std::setprecision(3) << mops << "\n";
                outfile.flush();
            }

            prev_operations = current_operations;

            // 检查是否达到4分钟
            if (elapsed_seconds >= WorkloadConfig::kFixedRuntimeSeconds) {
                should_stop.store(true);
                break;
            }
        }

        if (outfile.is_open()) {
            outfile.close();
        }
    }

public:
    MultiThreadedYCSBBenchmark(const std::string& output_file = "multithreaded_ycsb_mops_results.txt")
        : output_filename(output_file), global_insert_counter(0) {}

    void runBenchmark(const WorkloadConfig& config) {
        std::cout << "\n=== Running YCSB MOPS Benchmark ===" << std::endl;

        // 打印核心配置信息
        std::cout << "Records: " << config.recordcount
            << ", Threads: " << config.threadcount
            << ", Distribution: " << config.requestdistribution << std::endl;

        // 初始化
        hashtable.clear();
        global_insert_counter.store(0);
        should_stop.store(false);
        total_operations.store(0);
        mops_measurements.clear();

        // 加载初始数据
        loadInitialData(config);

        // 清空并准备输出文件
        std::ofstream outfile(output_filename);
        if (outfile.is_open()) {
            auto now = std::chrono::system_clock::now();
            auto time_t = std::chrono::system_clock::to_time_t(now);
            outfile << "Multithreaded YCSB MOPS Benchmark Results\n";
            outfile << "Generated at: " << std::put_time(std::localtime(&time_t), "%Y-%m-%d %H:%M:%S") << "\n";
            outfile << "Threads: " << config.threadcount << "\n";
            outfile << "Records: " << config.recordcount << "\n";
            outfile << "Distribution: " << config.requestdistribution << "\n";
            outfile << "Runtime: " << WorkloadConfig::kFixedRuntimeSeconds << " seconds\n";
            outfile << std::string(50, '=') << "\n";
            outfile.close();
        }

        // 执行基准测试
        std::cout << "Starting " << WorkloadConfig::kFixedRuntimeSeconds << "s benchmark..." << std::endl;

        std::vector<std::thread> worker_threads;

        // 启动性能监控线程
        std::thread monitor_thread(&MultiThreadedYCSBBenchmark::performanceMonitor, this);

        // 启动工作线程
        for (int i = 0; i < config.threadcount; ++i) {
            worker_threads.emplace_back(&MultiThreadedYCSBBenchmark::benchmarkWorker, this, std::ref(config), i);
        }

        // 等待监控线程完成（它会设置should_stop）
        monitor_thread.join();

        // 等待所有工作线程完成
        for (auto& thread : worker_threads) {
            thread.join();
        }

        std::cout << "Benchmark completed!" << std::endl;
        std::cout << "Total operations: " << total_operations.load() << std::endl;

        // 计算平均MOPS
        double average_mops = 0.0;
        {
            std::lock_guard<std::mutex> lock(mops_mutex);
            if (!mops_measurements.empty()) {
                // 排除前几次测量（预热期）
                size_t start_idx = std::min(static_cast<size_t>(3), mops_measurements.size());
                double sum = 0.0;
                for (size_t i = start_idx; i < mops_measurements.size(); ++i) {
                    sum += mops_measurements[i];
                }
                average_mops = sum / (mops_measurements.size() - start_idx);
            }
        }

        std::cout << std::fixed << std::setprecision(3);
        std::cout << "Average MOPS (excluding warmup): " << average_mops << std::endl;

        // 写入最终结果
        std::ofstream final_outfile(output_filename, std::ios::app);
        if (final_outfile.is_open()) {
            final_outfile << "\n=== Final Results ===\n";
            final_outfile << "Total Operations: " << total_operations.load() << "\n";
            final_outfile << std::fixed << std::setprecision(3);
            final_outfile << "Average MOPS (excluding warmup): " << average_mops << "\n";
            final_outfile << "Final HashTable Size: " << hashtable.size() << "\n";
            final_outfile.close();
        }
    }

    void printResults(const std::string& workload_name = "") {
        std::lock_guard<std::mutex> lock(mops_mutex);
        if (!mops_measurements.empty()) {
            auto max_mops = *std::max_element(mops_measurements.begin(), mops_measurements.end());
            auto min_mops = *std::min_element(mops_measurements.begin(), mops_measurements.end());

            std::cout << std::fixed << std::setprecision(3);
            std::cout << "Max MOPS: " << max_mops << ", Min MOPS: " << min_mops << std::endl;
        }
    }

    void generateGlobalSummary(const GlobalStats& global_stats, const std::string& workload_summary = "") {
        // 简化的全局总结
    }
};

int main(int argc, char* argv[]) {
    std::vector<std::string> workload_files;
    std::string output_file = "multithreaded_ycsb_mops_results.txt";

    GlobalStats global_stats;
    std::stringstream workload_summary;

    // 解析命令行参数
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--output" || arg == "-o") {
            if (i + 1 < argc) {
                output_file = argv[++i];
            }
        }
        else if (arg == "--help" || arg == "-h") {
            std::cout << "Multithreaded YCSB MOPS Benchmark\n"
                << "Usage: " << argv[0] << " [options] [workload_files...]\n"
                << "Options:\n"
                << "  -o, --output <file>    Output results to file\n"
                << "  -h, --help            Show this help message\n";
            return 0;
        }
        else {
            workload_files.push_back(arg);
        }
    }

    if (workload_files.empty()) {
        std::cout << "Error: No workload files specified." << std::endl;
        std::cout << "Usage: " << argv[0] << " [options] workload_file1 [workload_file2 ...]" << std::endl;
        std::cout << "Use --help for more information." << std::endl;
        return 1;
    }

    global_stats.total_workloads = workload_files.size();
    global_stats.startTiming();

    std::cout << "Multithreaded YCSB MOPS Benchmark" << std::endl;
    std::cout << std::string(80, '=') << std::endl;

    for (const auto& filename : workload_files) {
        std::cout << "Testing workload: " << filename << std::endl;

        global_stats.updatePeakMemory();

        WorkloadConfig config = WorkloadParser::parseConfig(filename);

        // 为每个工作负载创建独立的输出文件
        std::string workload_output_file = filename + "_mops_results.txt";
        MultiThreadedYCSBBenchmark benchmark(workload_output_file);

        // 记录工作负载信息
        workload_summary << "  " << filename << " (records: " << config.recordcount
            << ", threads: " << config.threadcount
            << ", distribution: " << config.requestdistribution << ")\n";

        benchmark.runBenchmark(config);
        benchmark.printResults(filename);

        global_stats.updatePeakMemory();
    }

    global_stats.endTiming();
    global_stats.updatePeakMemory();

    MultiThreadedYCSBBenchmark summary_benchmark(output_file);
    summary_benchmark.generateGlobalSummary(global_stats, workload_summary.str());

    std::cout << "\n" << std::string(80, '=') << std::endl;
    std::cout << "All benchmarks completed!" << std::endl;
    std::cout << "Total time: " << std::fixed << std::setprecision(2)
        << global_stats.total_benchmark_time_ms / 1000.0 << " seconds" << std::endl;

    return 0;
}