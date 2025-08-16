#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>
#include <iomanip>
#include <mutex>
#include <shared_mutex>
#include <string_view> // Include for string_view
#include <numeric>     // Include for std::accumulate

// Global parameters - will be overwritten by the workload file
static double kReadProportion = 0.5;
static double kUpdateProportion = 0.5;
static double kInsertProportion = 0.0;
static double kScanProportion = 0.0;
static double kRMWProportion = 0.0;
static std::string kRequestDistribution = "zipfian";

// Fixed parameters
constexpr static int64_t kRecordCount = 128000000;
// Pre-allocate 10% extra space for INSERT operations
constexpr static int64_t kPaddedRecordCount = static_cast<int64_t>(kRecordCount * 1.1);
constexpr static uint32_t kKeyLen = 32;
constexpr static uint32_t kValueLen = 64;
constexpr static uint32_t kNumMutatorThreads = 400;
constexpr static uint32_t kReqSeqLenPerCore = 1 << 20; // 1M per core
constexpr static uint32_t kMonitorPerIter = 1024;
constexpr static uint32_t kMinMonitorIntervalUs = 10 * 1000 * 1000; // 10 seconds
constexpr static uint32_t kMaxRunningUs = 200 * 1000 * 1000; // 200 seconds
constexpr static double kZipfParamS = 0.8;

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

// Zipf Distribution
class ZipfDistribution {
private:
    std::vector<double> prob_;
    std::discrete_distribution<int64_t> dist_;
public:
    ZipfDistribution(int64_t n, double s) {
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

// Distribution Generator
class DistributionGenerator {
private:
    std::mt19937 gen;
    std::uniform_real_distribution<double> uniform_dist{ 0.0, 1.0 };
public:
    DistributionGenerator() : gen(std::random_device{}()) {}
    int64_t generateKey(const std::string& distribution, int64_t range) {
        if (distribution == "zipfian") {
            ZipfDistribution zipf(range, kZipfParamS);
            return zipf(gen);
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
        else { // uniform
            std::uniform_int_distribution<int64_t> dist(0, range - 1);
            return dist(gen);
        }
    }
    double generateDouble() { return uniform_dist(gen); }
};

// Key Generator
class KeyGenerator {
public:
    static void generateKey(int64_t keynum, char* key_data, uint32_t key_len) {
        uint64_t hashedKey = FNVHash::hash(static_cast<uint64_t>(keynum));
        std::string keyStr = "user" + std::to_string(hashedKey);
        strncpy(key_data, keyStr.c_str(), key_len);
        key_data[key_len - 1] = '\0';
    }
};

// Data Structures
struct Key {
    char data[kKeyLen];
};

struct Value {
    char data[kValueLen];
};

struct alignas(64) Cnt {
    uint64_t c;
};

// **Modified** Concurrent Hash Table
class ConcurrentHashTable {
private:
    struct Bucket {
        // Use string_view to avoid data copying
        std::unordered_map<std::string_view, std::string_view> data;
        mutable std::shared_mutex mtx;
        char padding[64];
    };

    std::vector<std::unique_ptr<Bucket>> buckets_;
    size_t mask_;

    size_t getBucket(std::string_view key) const {
        return std::hash<std::string_view>{}(key)&mask_;
    }

public:
    ConcurrentHashTable(size_t num_buckets = 2048) {
        size_t actual = 1;
        while (actual < num_buckets) actual <<= 1;

        buckets_.reserve(actual);
        for (size_t i = 0; i < actual; ++i) {
            buckets_.push_back(std::make_unique<Bucket>());
        }
        mask_ = actual - 1;
    }

    void put(std::string_view key, std::string_view value) {
        auto& bucket = *buckets_[getBucket(key)];
        std::unique_lock<std::shared_mutex> lock(bucket.mtx);
        bucket.data[key] = value;
    }

    // **Added**: For Update/RMW operations, directly modify the Value's memory
    void update(std::string_view key, const char* new_value_data, size_t new_value_len) {
        auto& bucket = *buckets_[getBucket(key)];
        std::unique_lock<std::shared_mutex> lock(bucket.mtx);
        auto it = bucket.data.find(key);
        if (it != bucket.data.end()) {
            // it->second is a string_view, its data() returns a const char*
            // We cast away constness to modify the memory it points to
            char* dest = const_cast<char*>(it->second.data());
            memcpy(dest, new_value_data, std::min((size_t)kValueLen, new_value_len));
        }
    }

    bool get(std::string_view key, std::string_view& value) {
        auto& bucket = *buckets_[getBucket(key)];
        std::shared_lock<std::shared_mutex> lock(bucket.mtx);
        auto it = bucket.data.find(key);
        if (it != bucket.data.end()) {
            value = it->second;
            return true;
        }
        return false;
    }

    size_t size() const {
        size_t total = 0;
        for (auto& bucket : buckets_) {
            std::shared_lock<std::shared_mutex> lock(bucket->mtx);
            total += bucket->data.size();
        }
        return total;
    }
};

// Global variables
std::atomic_flag flag;
std::unique_ptr<std::mt19937> generators[kNumMutatorThreads];
thread_local uint32_t per_core_req_idx = 0;
Key* all_gen_keys;
Value* all_gen_values; // **Added**: Unified storage for Values
uint32_t all_zipf_key_indices[kNumMutatorThreads][kReqSeqLenPerCore];
Cnt cnts[kNumMutatorThreads];
std::vector<double> mops_vec;
uint64_t prev_sum_cnts = 0;
uint64_t prev_us = 0;
uint64_t running_us = 0;
// Atomic counter starts from kRecordCount to assign indices for new inserts
std::atomic<int64_t> global_insert_idx{ kRecordCount };
ConcurrentHashTable* table_ptr;
std::ofstream log_file;

uint64_t microtime() {
    auto now = std::chrono::high_resolution_clock::now();
    return std::chrono::duration_cast<std::chrono::microseconds>(now.time_since_epoch()).count();
}

void generateValue(char* data, uint32_t len, DistributionGenerator& gen) {
    const std::string chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
    for (uint32_t i = 0; i < len; ++i) {
        data[i] = chars[static_cast<size_t>(gen.generateDouble() * chars.length())];
    }
    data[len - 1] = '\0';
}

// Parse workload file
void parseWorkloadFile(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cout << "Warning: Could not open " << filename << ", using defaults" << std::endl;
        return;
    }

    std::cout << "Parsing " << filename << std::endl;
    std::string line;
    while (std::getline(file, line)) {
        // Remove whitespace
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

void prepare() {
    std::cout << "=== C++ YCSB Baseline Benchmark ===" << std::endl;
    std::cout << "Records: " << kRecordCount << " (Padded: " << kPaddedRecordCount << ")" << std::endl;
    std::cout << "Threads: " << kNumMutatorThreads << std::endl;
    std::cout << "Operations: Read(" << kReadProportion << ") Update(" << kUpdateProportion
        << ") Insert(" << kInsertProportion << ") RMW(" << kRMWProportion << ")" << std::endl;
    std::cout << "Request Distribution: " << kRequestDistribution << std::endl;
    std::cout << "===================================" << std::endl;

    // Allocate arrays for keys and values
    all_gen_keys = new Key[kPaddedRecordCount];
    all_gen_values = new Value[kPaddedRecordCount];

    // Initialize random number generators
    for (uint32_t i = 0; i < kNumMutatorThreads; i++) {
        std::random_device rd;
        generators[i].reset(new std::mt19937(rd()));
    }

    // Data loading
    std::cout << "Loading " << kRecordCount << " records..." << std::endl;
    auto load_start = microtime();

    std::vector<std::thread> threads;
    for (uint32_t tid = 0; tid < kNumMutatorThreads; tid++) {
        threads.emplace_back([&, tid]() {
            auto records_per_thread = kRecordCount / kNumMutatorThreads;
            auto start_idx = tid * records_per_thread;
            auto end_idx = start_idx + records_per_thread;
            if (tid == kNumMutatorThreads - 1) {
                end_idx = kRecordCount;
            }

            DistributionGenerator gen;
            for (int64_t i = start_idx; i < end_idx; i++) {
                // Generate data directly into the pre-allocated arrays
                KeyGenerator::generateKey(i, all_gen_keys[i].data, kKeyLen);
                generateValue(all_gen_values[i].data, kValueLen, gen);

                // Insert using string_view, no data copy involved
                std::string_view key_sv(all_gen_keys[i].data, strnlen(all_gen_keys[i].data, kKeyLen));
                std::string_view val_sv(all_gen_values[i].data, strnlen(all_gen_values[i].data, kValueLen));
                table_ptr->put(key_sv, val_sv);
            }
            });
    }
    for (auto& thread : threads) {
        thread.join();
    }

    auto load_end = microtime();
    auto load_time = (load_end - load_start) / 1000000.0;
    std::cout << "Loaded in " << load_time << " seconds" << std::endl;

    // Generate access patterns
    DistributionGenerator gen;
    if (kRequestDistribution == "zipfian") {
        ZipfDistribution zipf(kRecordCount, kZipfParamS);
        for (uint32_t i = 0; i < kReqSeqLenPerCore; i++) {
            auto key_idx = zipf(*generators[0]);
            all_zipf_key_indices[0][i] = key_idx;
        }
    }
    else if (kRequestDistribution == "uniform") {
        std::uniform_int_distribution<uint32_t> uniform_dist(0, kRecordCount - 1);
        for (uint32_t i = 0; i < kReqSeqLenPerCore; i++) {
            auto key_idx = uniform_dist(*generators[0]);
            all_zipf_key_indices[0][i] = key_idx;
        }
    }
    else if (kRequestDistribution == "latest") {
        std::exponential_distribution<double> exp_dist(2.0);
        for (uint32_t i = 0; i < kReqSeqLenPerCore; i++) {
            int64_t offset = static_cast<int64_t>(exp_dist(*generators[0]));
            auto key_idx = std::max(static_cast<int64_t>(0),
                static_cast<int64_t>(kRecordCount) - 1 - offset % kRecordCount);
            all_zipf_key_indices[0][i] = static_cast<uint32_t>(key_idx);
        }
    }

    // Copy access patterns to all threads
    for (uint32_t k = 1; k < kNumMutatorThreads; k++) {
        memcpy(all_zipf_key_indices[k], all_zipf_key_indices[0],
            sizeof(uint32_t) * kReqSeqLenPerCore);
    }
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
                std::vector<double> last_5_mops(
                    mops_vec.end() - std::min(static_cast<int>(mops_vec.size()), 5),
                    mops_vec.end());
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

void bench_ycsb_operations() {
    prev_us = microtime();
    std::vector<std::thread> threads;

    for (uint32_t tid = 0; tid < kNumMutatorThreads; tid++) {
        threads.emplace_back([&, tid]() {
            uint32_t cnt = 0;
            DistributionGenerator gen;

            while (1) {
                if (cnt++ % kMonitorPerIter == 0) monitor_perf();

                auto key_idx = all_zipf_key_indices[tid][per_core_req_idx++];
                if (per_core_req_idx == kReqSeqLenPerCore) per_core_req_idx = 0;

                double op_selector = gen.generateDouble();
                double cumulative = 0.0;

                Key& key_ref = all_gen_keys[key_idx];
                std::string_view key_sv(key_ref.data, strnlen(key_ref.data, kKeyLen));

                if ((cumulative += kReadProportion) >= op_selector) {
                    // READ
                    std::string_view val_sv;
                    table_ptr->get(key_sv, val_sv);
                }
                else if ((cumulative += kUpdateProportion) >= op_selector) {
                    // UPDATE
                    Value new_val;
                    generateValue(new_val.data, kValueLen, gen);
                    table_ptr->update(key_sv, new_val.data, kValueLen);
                }
                else if ((cumulative += kInsertProportion) >= op_selector) {
                    // INSERT
                    int64_t insert_idx = global_insert_idx.fetch_add(1);
                    if (insert_idx < kPaddedRecordCount) {
                        // Generate new data in the pre-allocated space
                        KeyGenerator::generateKey(insert_idx, all_gen_keys[insert_idx].data, kKeyLen);
                        generateValue(all_gen_values[insert_idx].data, kValueLen, gen);

                        std::string_view new_key_sv(all_gen_keys[insert_idx].data, strnlen(all_gen_keys[insert_idx].data, kKeyLen));
                        std::string_view new_val_sv(all_gen_values[insert_idx].data, strnlen(all_gen_values[insert_idx].data, kValueLen));
                        table_ptr->put(new_key_sv, new_val_sv);
                    }
                }
                else if ((cumulative += kRMWProportion) >= op_selector) {
                    // READ-MODIFY-WRITE
                    std::string_view val_sv;
                    if (table_ptr->get(key_sv, val_sv)) {
                        Value new_val;
                        generateValue(new_val.data, kValueLen, gen);
                        table_ptr->update(key_sv, new_val.data, kValueLen);
                    }
                }

                cnts[tid].c++;
            }
            });
    }
    for (auto& thread : threads) {
        thread.join();
    }
}

int main(int argc, char* argv[]) {
    if (argc >= 2) {
        parseWorkloadFile(argv[1]);
        std::string workload_file = argv[1];
        std::string log_filename = "cpp_" + workload_file.substr(0, workload_file.find('.')) + ".log";
        log_file.open(log_filename);
    }

    ConcurrentHashTable table(2048);
    table_ptr = &table;

    std::cout << "Prepare..." << std::endl;
    prepare();
    std::cout << "YCSB Operations..." << std::endl;
    bench_ycsb_operations();

    // Clean up globally allocated memory
    delete[] all_gen_keys;
    delete[] all_gen_values;

    return 0;
}