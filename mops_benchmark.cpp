#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <random>
#include <thread>
#include <atomic>
#include <memory>
#include <unordered_map>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <iomanip>
#include <mutex>
#include <shared_mutex>
#include <string_view>
#include <numeric>
#include <cmath>
#include <cstring>

// Global parameters - will be overwritten by the workload file
static double kReadProportion = 0.5;
static double kUpdateProportion = 0.5;
static double kInsertProportion = 0.0;
static double kScanProportion = 0.0;
static double kRMWProportion = 0.0;
static std::string kRequestDistribution = "zipfian";

// Fixed parameters
constexpr static int64_t kRecordCount = 128000000;
constexpr static uint32_t kKeyLen = 32;
constexpr static uint32_t kValueLen = 64;
constexpr static uint32_t kNumMutatorThreads = 20;
constexpr static uint32_t kReqSeqLenPerCore = 1 << 20; // 1M per core
constexpr static uint32_t kMonitorPerIter = 1024;
constexpr static uint32_t kMinMonitorIntervalUs = 10 * 1000 * 1000; // 10 seconds
constexpr static uint32_t kMaxRunningUs = 200 * 1000 * 1000; // 200 seconds
constexpr static double kZipfParamS = 0.8;
constexpr static size_t kNumBuckets = 1 << 12; // 262144 buckets

// FNV Hash
class FNVHash {
public:
    static uint64_t hash(uint64_t val) {
        const uint64_t FNV_OFFSET_BASIS = 0xCBF29CE484222325ULL;
        const uint64_t FNV_PRIME = 1099511628211ULL;
        uint64_t hashval = FNV_OFFSET_BASIS;
        for (int i = 0; i < 8; i++) {
            uint8_t octet = val & 0x00ff;
            val = val >> 8;
            hashval = hashval ^ octet;
            hashval = hashval * FNV_PRIME;
        }
        return hashval > 0 ? hashval : -hashval;
    }
};

class ZipfDistribution {
private:
    std::vector<double> prob_;
    std::discrete_distribution<int64_t> dist_;
public:
    ZipfDistribution(int64_t n, double s) {
        if (n <= 0) return;
        prob_.resize(n);
        double sum = 0.0;
        for (int64_t i = 1; i <= n; ++i) {
            prob_[i - 1] = 1.0 / std::pow(i, s);
            sum += prob_[i - 1];
        }
        for (auto& p : prob_) p /= sum;
        dist_ = std::discrete_distribution<int64_t>(prob_.begin(), prob_.end());
    }
    template<class Gen>
    int64_t operator()(Gen& gen) { return dist_(gen); }
};

class RandomGenerator {
private:
    std::mt19937 gen;
    std::uniform_real_distribution<double> uniform_dist{ 0.0, 1.0 };
public:
    RandomGenerator() : gen(std::random_device{}()) {}
    double generateDouble() { return uniform_dist(gen); }
};

class KeyGenerator {
public:
    static void generateKey(int64_t keynum, char* key_data, uint32_t key_len) {
        uint64_t hashedKey = FNVHash::hash(static_cast<uint64_t>(keynum));
        std::string keyStr = "user" + std::to_string(hashedKey);
        strncpy(key_data, keyStr.c_str(), key_len);
        key_data[key_len - 1] = '\0';
    }
};

struct Key { char data[kKeyLen]; };
struct Value { char data[kValueLen]; };

struct KeyHasher {
    std::size_t operator()(const Key* k) const {
        return std::hash<std::string_view>{}(std::string_view(k->data, strnlen(k->data, kKeyLen)));
    }
};

struct KeyEqualTo {
    bool operator()(const Key* lhs, const Key* rhs) const {
        return std::string_view(lhs->data, strnlen(lhs->data, kKeyLen)) ==
            std::string_view(rhs->data, strnlen(rhs->data, kKeyLen));
    }
};

struct alignas(64) Cnt { uint64_t c; };

class ConcurrentHashTable {
private:
    struct Bucket {
        std::unordered_map<const Key*, Value*, KeyHasher, KeyEqualTo> data;
        mutable std::shared_mutex mtx;
        char padding[64];
    };
    std::vector<std::unique_ptr<Bucket>> buckets_;
    size_t mask_;
    size_t getBucket(const Key* key) const { return KeyHasher{}(key)&mask_; }
public:
    ConcurrentHashTable(size_t num_buckets) {
        size_t actual = 1;
        while (actual < num_buckets) actual <<= 1;
        buckets_.reserve(actual);
        for (size_t i = 0; i < actual; ++i) {
            buckets_.push_back(std::make_unique<Bucket>());
        }
        mask_ = actual - 1;
    }
    void put(const Key* key, Value* value) {
        auto& bucket = *buckets_[getBucket(key)];
        std::unique_lock<std::shared_mutex> lock(bucket.mtx);
        bucket.data[key] = value;
    }
    void update(const Key* key, const char* new_value_data) {
        auto& bucket = *buckets_[getBucket(key)];
        std::unique_lock<std::shared_mutex> lock(bucket.mtx);
        auto it = bucket.data.find(key);
        if (it != bucket.data.end()) {
            memcpy(it->second->data, new_value_data, kValueLen);
        }
    }
    bool get(const Key* key, Value*& value) {
        auto& bucket = *buckets_[getBucket(key)];
        std::shared_lock<std::shared_mutex> lock(bucket.mtx);
        auto it = bucket.data.find(key);
        if (it != bucket.data.end()) { value = it->second; return true; }
        return false;
    }
    void clear() {
        for (auto& bucket_ptr : buckets_) {
            std::unique_lock<std::shared_mutex> lock(bucket_ptr->mtx);
            for (auto const& [key, val] : bucket_ptr->data) {
                delete key;
                delete val;
            }
            bucket_ptr->data.clear();
        }
    }
};

// Global variables
std::atomic_flag flag;
std::vector<const Key*> key_pointers_for_test;
// MODIFICATION: A single, small, shared vector for key access indices. ~4MB.
std::vector<uint32_t> shared_key_indices;
Cnt cnts[kNumMutatorThreads];
std::vector<double> mops_vec;
uint64_t prev_sum_cnts = 0;
uint64_t prev_us = 0;
uint64_t running_us = 0;
std::atomic<int64_t> global_insert_counter{ kRecordCount };
ConcurrentHashTable* table_ptr;
std::ofstream log_file;
thread_local uint32_t per_core_req_idx = 0; // Each thread has its own index into the shared array


uint64_t microtime() {
    auto now = std::chrono::high_resolution_clock::now();
    return std::chrono::duration_cast<std::chrono::microseconds>(now.time_since_epoch()).count();
}

void generateValue(char* data, uint32_t len, RandomGenerator& gen) {
    const std::string chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
    for (uint32_t i = 0; i < len; ++i) {
        data[i] = chars[static_cast<size_t>(gen.generateDouble() * chars.length())];
    }
    data[len - 1] = '\0';
}

void parseWorkloadFile(const std::string& filename); // Definition at the end

void prepare() {
    std::cout << "=== C++ YCSB Baseline Benchmark (Low-Memory Mode) ===" << std::endl;
    std::cout << "Records: " << kRecordCount << std::endl;
    std::cout << "Threads: " << kNumMutatorThreads << std::endl;
    std::cout << "Request Distribution: " << kRequestDistribution << std::endl;
    std::cout << "=====================================================" << std::endl;

    std::cout << "Loading " << kRecordCount << " records (lock-free parallel load)..." << std::endl;
    auto load_start = microtime();

    std::vector<std::thread> threads;
    std::vector<std::vector<const Key*>> local_key_pointers(kNumMutatorThreads);

    for (uint32_t tid = 0; tid < kNumMutatorThreads; tid++) {
        threads.emplace_back([&, tid]() {
            auto records_per_thread = kRecordCount / kNumMutatorThreads;
            auto start_idx = tid * records_per_thread;
            auto end_idx = (tid == kNumMutatorThreads - 1) ? kRecordCount : start_idx + records_per_thread;

            RandomGenerator gen;
            local_key_pointers[tid].reserve(end_idx - start_idx);

            for (int64_t i = start_idx; i < end_idx; i++) {
                Key* new_key = new Key();
                Value* new_val = new Value();
                KeyGenerator::generateKey(i, new_key->data, kKeyLen);
                generateValue(new_val->data, kValueLen, gen);
                table_ptr->put(new_key, new_val);
                local_key_pointers[tid].push_back(new_key);
            }
            });
    }
    for (auto& thread : threads) { thread.join(); }

    std::cout << "Merging key pointers..." << std::endl;
    key_pointers_for_test.reserve(kRecordCount);
    for (uint32_t tid = 0; tid < kNumMutatorThreads; ++tid) {
        key_pointers_for_test.insert(key_pointers_for_test.end(),
            local_key_pointers[tid].begin(),
            local_key_pointers[tid].end());
    }

    auto load_end = microtime();
    std::cout << "Loaded in " << (load_end - load_start) / 1000000.0 << " seconds" << std::endl;

    // MODIFICATION: Generate a single, shared, small access pattern array.
    std::cout << "Generating " << kReqSeqLenPerCore / (1024.0 * 1024.0) << "M shared access indices..." << std::endl;
    shared_key_indices.resize(kReqSeqLenPerCore);
    std::mt19937 gen(std::random_device{}());
    if (kRequestDistribution == "zipfian") {
        ZipfDistribution zipf(kRecordCount, kZipfParamS);
        for (uint32_t i = 0; i < kReqSeqLenPerCore; ++i) shared_key_indices[i] = zipf(gen);
    }
    else if (kRequestDistribution == "latest") {
        std::exponential_distribution<double> exp_dist(2.0);
        for (uint32_t i = 0; i < kReqSeqLenPerCore; ++i) {
            int64_t offset = static_cast<int64_t>(exp_dist(gen));
            shared_key_indices[i] = std::max(static_cast<int64_t>(0), kRecordCount - 1 - offset % kRecordCount);
        }
    }
    else if (kRequestDistribution == "sequential") {
        for (uint32_t i = 0; i < kReqSeqLenPerCore; ++i) shared_key_indices[i] = i % kRecordCount;
    }
    else { // uniform
        std::uniform_int_distribution<uint32_t> dist(0, kRecordCount - 1);
        for (uint32_t i = 0; i < kReqSeqLenPerCore; ++i) shared_key_indices[i] = dist(gen);
    }
    std::cout << "Access pattern generated." << std::endl;
}

void monitor_perf(); // Definition at the end

void bench_ycsb_operations() {
    prev_us = microtime();
    std::vector<std::thread> threads;

    for (uint32_t tid = 0; tid < kNumMutatorThreads; tid++) {
        threads.emplace_back([&, tid]() {
            uint32_t cnt = 0;
            RandomGenerator op_gen; // Thread-local generator for operations

            while (true) {
                if (cnt++ % kMonitorPerIter == 0) monitor_perf();

                // MODIFICATION: Read from the shared index array, using a thread-local counter to progress.
                uint32_t key_idx = shared_key_indices[per_core_req_idx++];
                if (per_core_req_idx == kReqSeqLenPerCore) per_core_req_idx = 0;
                const Key* key_ptr = key_pointers_for_test[key_idx];

                double op_selector = op_gen.generateDouble();
                double cumulative = 0.0;

                if ((cumulative += kReadProportion) >= op_selector) {
                    Value* val_ptr = nullptr;
                    table_ptr->get(key_ptr, val_ptr);
                }
                else if ((cumulative += kUpdateProportion) >= op_selector) {
                    Value new_val_data;
                    generateValue(new_val_data.data, kValueLen, op_gen);
                    table_ptr->update(key_ptr, new_val_data.data);
                }
                else if ((cumulative += kInsertProportion) >= op_selector) {
                    int64_t keynum = global_insert_counter.fetch_add(1);
                    Key* new_key = new Key();
                    Value* new_val = new Value();
                    KeyGenerator::generateKey(keynum, new_key->data, kKeyLen);
                    generateValue(new_val->data, kValueLen, op_gen);
                    table_ptr->put(new_key, new_val);
                }
                else if ((cumulative += kRMWProportion) >= op_selector) {
                    Value* val_ptr = nullptr;
                    if (table_ptr->get(key_ptr, val_ptr)) {
                        Value new_val_data;
                        generateValue(new_val_data.data, kValueLen, op_gen);
                        table_ptr->update(key_ptr, new_val_data.data);
                    }
                }
                cnts[tid].c++;
            }
            });
    }
    for (auto& thread : threads) { thread.join(); }
}

int main(int argc, char* argv[]) {
    if (argc >= 2) {
        parseWorkloadFile(argv[1]);
        std::string workload_file = argv[1];
        std::string log_filename = "cpp_" + workload_file.substr(0, workload_file.find('.')) + ".log";
        log_file.open(log_filename);
    }

    ConcurrentHashTable table(kNumBuckets);
    table_ptr = &table;

    std::cout << "Prepare..." << std::endl;
    prepare();

    std::cout << "YCSB Operations..." << std::endl;
    bench_ycsb_operations();

    std::cout << "Cleaning up memory..." << std::endl;
    table_ptr->clear();
    std::cout << "Cleanup complete." << std::endl;

    return 0;
}

// Function definitions moved to the end for clarity
void parseWorkloadFile(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cout << "Warning: Could not open " << filename << ", using defaults" << std::endl;
        return;
    }
    std::cout << "Parsing " << filename << std::endl;
    std::string line;
    while (std::getline(file, line)) {
        line.erase(std::remove_if(line.begin(), line.end(), ::isspace), line.end());
        if (line.empty() || line[0] == '#') continue;
        auto eq_pos = line.find('=');
        if (eq_pos == std::string::npos) continue;
        auto key = line.substr(0, eq_pos);
        auto value = line.substr(eq_pos + 1);
        if (key == "readproportion") kReadProportion = std::stod(value);
        else if (key == "updateproportion") kUpdateProportion = std::stod(value);
        else if (key == "insertproportion") kInsertProportion = std::stod(value);
        else if (key == "scanproportion") kScanProportion = std::stod(value);
        else if (key == "readmodifywriteproportion") kRMWProportion = std::stod(value);
        else if (key == "requestdistribution") kRequestDistribution = value;
    }
    std::cout << "Loaded: Read=" << kReadProportion << " Update=" << kUpdateProportion
        << " Insert=" << kInsertProportion << " RMW=" << kRMWProportion
        << " Dist=" << kRequestDistribution << std::endl;
}

void monitor_perf() {
    if (!flag.test_and_set()) {
        auto us = microtime();
        if (us - prev_us > kMinMonitorIntervalUs) {
            uint64_t sum_cnts = 0;
            for (uint32_t i = 0; i < kNumMutatorThreads; i++) {
                sum_cnts += cnts[i].c;
            }
            us = microtime();
            auto mops = (double)(sum_cnts - prev_sum_cnts) / (us - prev_us);
            mops_vec.push_back(mops);
            running_us += (us - prev_us);
            auto output = "[" + std::to_string(running_us / 1000000) + "s] MOPS: " +
                std::to_string(mops) + " | Total Ops: " +
                std::to_string(sum_cnts / 1000000) + "M";
            std::cout << output << std::endl;
            if (log_file.is_open()) {
                log_file << output << std::endl;
                log_file.flush();
            }
            if (running_us >= kMaxRunningUs) {
                std::vector<double> last_5_mops(mops_vec.end() - std::min(static_cast<int>(mops_vec.size()), 5), mops_vec.end());
                double final_mops = 0.0;
                if (!last_5_mops.empty()) {
                    final_mops = std::accumulate(last_5_mops.begin(), last_5_mops.end(), 0.0) / last_5_mops.size();
                }
                std::cout << "mops = " << final_mops << std::endl;
                if (log_file.is_open()) {
                    log_file << "mops = " << final_mops << std::endl;
                    log_file.close();
                }
                exit(0);
            }
            prev_us = us;
            prev_sum_cnts = sum_cnts;
        }
        flag.clear();
    }
}