#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <numeric>
#include <random>
#include <shared_mutex>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

// =========================
// Utilities: barrier (C++17)
// =========================
class SimpleBarrier {
    std::mutex mtx_;
    std::condition_variable cv_;
    size_t count_;
    size_t arrived_ = 0;
    size_t gen_ = 0;

public:
    explicit SimpleBarrier(size_t count) : count_(count) {}
    void arrive_and_wait() {
        std::unique_lock<std::mutex> lk(mtx_);
        size_t g = gen_;
        if (++arrived_ == count_) {
            arrived_ = 0;
            ++gen_;
            cv_.notify_all();
        }
        else {
            cv_.wait(lk, [&] { return g != gen_; });
        }
    }
};

// =========================
// FNV-1a 64-bit hash (same as AIFM main)
// =========================
struct FNVHash {
    static constexpr uint64_t kOffset = 0xCBF29CE484222325ULL;
    static constexpr uint64_t kPrime = 1099511628211ULL;
    static uint64_t hash(uint64_t v) {
        uint64_t h = kOffset;
        for (int i = 0; i < 8; ++i) {
            uint8_t o = v & 0xFF;
            v >>= 8;
            h ^= o;
            h *= kPrime;
        }
        return h ? h : -h;
    }
};

// =========================
// Workload configuration (kept compatible with your AIFM main)
// =========================
struct WorkloadConfig {
    int64_t recordcount = 1000;
    std::string workload = "site.ycsb.workloads.CoreWorkload";
    int fieldcount = 2;
    int fieldlength = 15;
    int minfieldlength = 1;
    std::string fieldlengthdistribution = "constant";
    std::string fieldnameprefix = "field";
    bool readallfields = true;
    double readproportion = 0.5;     // default to 50/50 read/update
    double updateproportion = 0.5;
    double scanproportion = 0.0;
    double insertproportion = 0.0;
    double readmodifywriteproportion = 0.0;
    std::string requestdistribution = "zipfian"; // match AIFM default
    std::string insertorder = "hashed";          // match AIFM default
    int maxscanlength = 1000;
    int minscanlength = 1;
    std::string scanlengthdistribution = "uniform";
    int zeropadding = 1;
    int64_t insertstart = 0;
    int64_t insertcount = -1;
    int threadcount = 400; // hard high-concurrency
    static constexpr int64_t kFixedRuntimeSeconds = 240; // 4 minutes
    static constexpr int64_t kMonitorIntervalSeconds = 10; // every 10s
};

// =========================
// Record representation
// =========================
struct Record {
    std::string key;
    std::vector<std::pair<std::string, std::string>> fields;
};

// =========================
// Striped concurrent hash table (true multi-threading)
// =========================
class ConcurrentHashTable {
    struct Bucket {
        std::unordered_map<std::string, Record> map;
        mutable std::shared_mutex mtx;
    };
    std::vector<Bucket> stripes_;

    static size_t str_hash(const std::string& s) {
        // Use std::hash<string>; good enough for striping
        return std::hash<std::string>{}(s);
    }

    Bucket& bucket_for(const std::string& key) {
        return stripes_[str_hash(key) % stripes_.size()];
    }
    const Bucket& bucket_for(const std::string& key) const {
        return stripes_[str_hash(key) % stripes_.size()];
    }

public:
    explicit ConcurrentHashTable(size_t stripes = 1024) : stripes_(stripes) {}

    void reserve(size_t n) {
        size_t per = std::max<size_t>(1, n / stripes_.size());
        for (auto& b : stripes_) b.map.reserve(per);
    }

    bool insert(const std::string& key, const Record& rec) {
        auto& b = bucket_for(key);
        std::unique_lock<std::shared_mutex> lk(b.mtx);
        b.map[key] = rec;
        return true;
    }

    bool read(const std::string& key, Record& out, bool all_fields = true) const {
        const auto& b = bucket_for(key);
        std::shared_lock<std::shared_mutex> lk(b.mtx);
        auto it = b.map.find(key);
        if (it == b.map.end()) return false;
        out = it->second;
        if (!all_fields && !out.fields.empty()) {
            thread_local std::mt19937 gen(std::random_device{}());
            std::uniform_int_distribution<size_t> dist(0, out.fields.size() - 1);
            size_t idx = dist(gen);
            Record r{ out.key, {out.fields[idx]} };
            out = std::move(r);
        }
        return true;
    }

    bool update(const std::string& key, const Record& rec, bool write_all = true) {
        auto& b = bucket_for(key);
        std::unique_lock<std::shared_mutex> lk(b.mtx);
        auto it = b.map.find(key);
        if (it == b.map.end()) return false;
        if (write_all) {
            it->second = rec;
        }
        else {
            if (!rec.fields.empty() && !it->second.fields.empty()) {
                thread_local std::mt19937 gen(std::random_device{}());
                std::uniform_int_distribution<size_t> dist(0, it->second.fields.size() - 1);
                size_t idx = dist(gen);
                it->second.fields[idx] = rec.fields[0];
            }
        }
        return true;
    }

    bool scan(const std::string& start_key, int count, std::vector<Record>& out) const {
        out.clear();
        // Lock all stripes with shared lock in a fixed order to avoid deadlock.
        std::vector<std::shared_lock<std::shared_mutex>> locks;
        locks.reserve(stripes_.size());
        for (auto& b : stripes_) locks.emplace_back(b.mtx);

        int scanned = 0;
        for (const auto& b : stripes_) {
            for (const auto& kv : b.map) {
                if (kv.first >= start_key) {
                    out.push_back(kv.second);
                    if (++scanned >= count) return true;
                }
            }
        }
        return !out.empty();
    }

    size_t size() const {
        size_t total = 0;
        for (auto& b : stripes_) {
            std::shared_lock<std::shared_mutex> lk(b.mtx);
            total += b.map.size();
        }
        return total;
    }

    void clear() {
        for (auto& b : stripes_) {
            std::unique_lock<std::shared_mutex> lk(b.mtx);
            b.map.clear();
        }
    }
};

// =========================
// Distributions (zipf/uniform/latest/sequential)
// =========================
class DistributionGenerator {
    std::mt19937 gen{ std::random_device{}() };
    std::uniform_real_distribution<double> uni{ 0.0, 1.0 };

    // Simple discrete Zipf sampler to mirror AIFM main's s=0.8
    class Zipf {
        std::vector<double> prob_;
        std::discrete_distribution<int> dist_;
    public:
        Zipf(int n = 1000, double s = 0.8) {
            prob_.resize(n);
            double sum = 0.0;
            for (int i = 1; i <= n; ++i) { prob_[i - 1] = 1.0 / std::pow(i, s); sum += prob_[i - 1]; }
            for (auto& p : prob_) p /= sum;
            dist_ = std::discrete_distribution<int>(prob_.begin(), prob_.end());
        }
        template<class G> int operator()(G& g) { return dist_(g); }
    };

    Zipf zipf_{ 100000, 0.8 }; // large-ish support for hotspot behavior

public:
    int64_t generateKey(const std::string& distribution, int64_t range) {
        if (distribution == "zipfian") {
            return zipf_(gen) % range;
        }
        else if (distribution == "latest") {
            std::exponential_distribution<double> e(2.0);
            int64_t off = static_cast<int64_t>(e(gen));
            return std::max<int64_t>(0, range - 1 - (off % range));
        }
        else if (distribution == "sequential") {
            static std::atomic<int64_t> c{ 0 };
            return (c++) % range;
        }
        else {
            std::uniform_int_distribution<int64_t> d(0, range - 1);
            return d(gen);
        }
    }
    double generateDouble() { return uni(gen); }
    int generateScanLength(const std::string& dist, int minL, int maxL) {
        if (dist == "uniform") {
            std::uniform_int_distribution<int> d(minL, maxL); return d(gen);
        }
        else if (dist == "zipfian") {
            Zipf z(maxL - minL + 1, 0.99); return minL + z(gen);
        }
        else { return minL; }
    }
    int generateFieldLength(const std::string& dist, int minL, int maxL) {
        if (dist == "uniform") {
            std::uniform_int_distribution<int> d(minL, maxL); return d(gen);
        }
        else if (dist == "zipfian") {
            Zipf z(maxL - minL + 1, 0.99); return minL + z(gen);
        }
        else { return maxL; }
    }
};

// =========================
// Key generator (identical behavior to AIFM main)
// =========================
class KeyGenerator {
    std::string insert_order_;
    int zero_padding_;
public:
    KeyGenerator(std::string order = "hashed", int padding = 1)
        : insert_order_(std::move(order)), zero_padding_(padding) {}

    std::string generateKey(int64_t keynum) const {
        if (insert_order_ == "hashed") {
            uint64_t h = FNVHash::hash(static_cast<uint64_t>(keynum));
            return std::string("user") + std::to_string(h);
        }
        std::ostringstream oss;
        if (zero_padding_ > 1) {
            oss << "user" << std::setfill('0') << std::setw(zero_padding_) << keynum;
        }
        else {
            oss << "user" << keynum;
        }
        return oss.str();
    }
};

// =========================
// Benchmark driver
// =========================
class MultiThreadedYCSBBenchmark {
    ConcurrentHashTable table_;
    std::string outfile_;
    std::atomic<bool> stop_{ false };
    std::atomic<int64_t> global_insert_counter_{ 0 };
    std::atomic<int64_t> total_ops_{ 0 };
    std::mutex mops_mu_;
    std::vector<double> mops_; // per-interval

    static std::string randValue(int len, DistributionGenerator& g) {
        static const std::string chars =
            "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
        std::string s; s.resize(len);
        for (int i = 0; i < len; ++i) s[i] = chars[static_cast<size_t>(g.generateDouble() * chars.size())];
        return s;
    }

    Record makeRecord(const std::string& key, const WorkloadConfig& cfg, DistributionGenerator& g) {
        Record r; r.key = key; r.fields.reserve(cfg.fieldcount);
        for (int i = 0; i < cfg.fieldcount; ++i) {
            int len = g.generateFieldLength(cfg.fieldlengthdistribution, cfg.minfieldlength, cfg.fieldlength);
            r.fields.emplace_back(cfg.fieldnameprefix + std::to_string(i), randValue(len, g));
        }
        return r;
    }

    void loadWorker(const WorkloadConfig& cfg, int tid, int nthreads, SimpleBarrier& bar) {
        KeyGenerator kg(cfg.insertorder, cfg.zeropadding);
        DistributionGenerator g;
        int64_t per = cfg.insertcount / nthreads;
        int64_t start = cfg.insertstart + tid * per;
        int64_t end = (tid == nthreads - 1) ? (cfg.insertstart + cfg.insertcount) : (start + per);

        for (int64_t i = start; i < end; ++i) {
            std::string k = kg.generateKey(i);
            auto rec = makeRecord(k, cfg, g);
            table_.insert(k, rec);
        }
        bar.arrive_and_wait();
    }

    void perfMonitor(const WorkloadConfig& cfg) {
        auto t0 = std::chrono::steady_clock::now();
        int64_t prev = 0;
        std::ofstream out(outfile_, std::ios::app);
        if (out) {
            out << "\n=== MOPS Performance Measurements ===\nTime(s)\tMOPS\n";
        }

        while (!stop_.load()) {
            std::this_thread::sleep_for(std::chrono::seconds(WorkloadConfig::kMonitorIntervalSeconds));
            auto now = std::chrono::steady_clock::now();
            double elapsed = std::chrono::duration<double>(now - t0).count();
            int64_t cur = total_ops_.load(std::memory_order_relaxed);
            int64_t delta = cur - prev;
            double mops = delta / (WorkloadConfig::kMonitorIntervalSeconds * 1e6);
            {
                std::lock_guard<std::mutex> lk(mops_mu_);
                mops_.push_back(mops);
            }
            std::cout << std::fixed << std::setprecision(3)
                << "[" << std::setw(3) << int(elapsed) << "s] MOPS: " << mops << std::endl;
            if (out) {
                out << std::fixed << std::setprecision(1) << elapsed << "\t"
                    << std::setprecision(3) << mops << "\n";
                out.flush();
            }
            prev = cur;
            if (elapsed >= WorkloadConfig::kFixedRuntimeSeconds) {
                stop_.store(true);
                break;
            }
        }
    }

    void benchWorker(const WorkloadConfig& cfg) {
        DistributionGenerator g;
        KeyGenerator kg(cfg.insertorder, cfg.zeropadding);
        int64_t local_ops = 0; // flush batched

        while (!stop_.load(std::memory_order_relaxed)) {
            double p = g.generateDouble();
            double acc = 0.0;

            if ((acc += cfg.readproportion) >= p) {
                int64_t keynum = cfg.insertstart + g.generateKey(cfg.requestdistribution, cfg.insertcount);
                std::string k = kg.generateKey(keynum);
                Record r; table_.read(k, r, cfg.readallfields);
            }
            else if ((acc += cfg.updateproportion) >= p) {
                int64_t keynum = cfg.insertstart + g.generateKey(cfg.requestdistribution, cfg.insertcount);
                std::string k = kg.generateKey(keynum);
                auto rec = makeRecord(k, cfg, g);
                table_.update(k, rec, true);
            }
            else if ((acc += cfg.insertproportion) >= p) {
                int64_t kn = cfg.insertstart + cfg.insertcount + global_insert_counter_.fetch_add(1);
                std::string k = kg.generateKey(kn);
                auto rec = makeRecord(k, cfg, g);
                table_.insert(k, rec);
            }
            else if ((acc += cfg.scanproportion) >= p) {
                int64_t keynum = cfg.insertstart + g.generateKey(cfg.requestdistribution, cfg.insertcount);
                std::string startk = kg.generateKey(keynum);
                int scan_len = g.generateScanLength(cfg.scanlengthdistribution, cfg.minscanlength, cfg.maxscanlength);
                std::vector<Record> res; table_.scan(startk, scan_len, res);
            }
            else if ((acc += cfg.readmodifywriteproportion) >= p) {
                int64_t keynum = cfg.insertstart + g.generateKey(cfg.requestdistribution, cfg.insertcount);
                std::string k = kg.generateKey(keynum);
                Record r; if (table_.read(k, r, true)) {
                    auto rec = makeRecord(k, cfg, g);
                    table_.update(k, rec, true);
                }
            }

            if (++local_ops == 1024) {
                total_ops_.fetch_add(local_ops, std::memory_order_relaxed);
                local_ops = 0;
            }
        }
        if (local_ops) total_ops_.fetch_add(local_ops, std::memory_order_relaxed);
    }

public:
    explicit MultiThreadedYCSBBenchmark(std::string out = "multithreaded_ycsb_mops_results.txt",
        size_t stripes = 1024)
        : table_(stripes), outfile_(std::move(out)) {}

    static WorkloadConfig parseConfig(const std::string& file) {
        WorkloadConfig c;
        std::ifstream f(file);
        if (!f.is_open()) {
            std::cerr << "Warning: cannot open " << file << ", using defaults\n";
            c.insertcount = (c.insertcount == -1) ? c.recordcount : c.insertcount;
            return c;
        }
        std::string line;
        while (std::getline(f, line)) {
            line.erase(std::remove_if(line.begin(), line.end(), ::isspace), line.end());
            if (line.empty() || line[0] == '#') continue;
            auto pos = line.find('='); if (pos == std::string::npos) continue;
            auto key = line.substr(0, pos); auto val = line.substr(pos + 1);
            if (key == "recordcount") c.recordcount = std::stoll(val);
            else if (key == "fieldcount") c.fieldcount = std::stoi(val);
            else if (key == "fieldlength") c.fieldlength = std::stoi(val);
            else if (key == "minfieldlength") c.minfieldlength = std::stoi(val);
            else if (key == "fieldlengthdistribution") c.fieldlengthdistribution = val;
            else if (key == "fieldnameprefix") c.fieldnameprefix = val;
            else if (key == "readallfields") c.readallfields = (val == "true");
            else if (key == "readproportion") c.readproportion = std::stod(val);
            else if (key == "updateproportion") c.updateproportion = std::stod(val);
            else if (key == "scanproportion") c.scanproportion = std::stod(val);
            else if (key == "insertproportion") c.insertproportion = std::stod(val);
            else if (key == "readmodifywriteproportion") c.readmodifywriteproportion = std::stod(val);
            else if (key == "requestdistribution") c.requestdistribution = val;
            else if (key == "insertorder") c.insertorder = val;
            else if (key == "maxscanlength") c.maxscanlength = std::stoi(val);
            else if (key == "minscanlength") c.minscanlength = std::stoi(val);
            else if (key == "scanlengthdistribution") c.scanlengthdistribution = val;
            else if (key == "zeropadding") c.zeropadding = std::stoi(val);
            else if (key == "insertstart") c.insertstart = std::stoll(val);
            else if (key == "insertcount") c.insertcount = std::stoll(val);
            else if (key == "threadcount") c.threadcount = std::stoi(val);
        }
        if (c.insertcount == -1) c.insertcount = c.recordcount;
        return c;
    }

    void run(const WorkloadConfig& cfg) {
        // Reset state
        table_.clear();
        total_ops_.store(0);
        global_insert_counter_.store(0);
        stop_.store(false);
        mops_.clear();

        // Pre-reserve capacity across stripes to reduce rehash under lock.
        table_.reserve(cfg.insertcount);

        // Load phase (true parallel)
        {
            SimpleBarrier bar(cfg.threadcount);
            std::vector<std::thread> ths;
            auto start = std::chrono::high_resolution_clock::now();
            for (int i = 0; i < cfg.threadcount; ++i) ths.emplace_back(&MultiThreadedYCSBBenchmark::loadWorker, this, std::cref(cfg), i, cfg.threadcount, std::ref(bar));
            for (auto& t : ths) t.join();
            auto end = std::chrono::high_resolution_clock::now();
            double sec = std::chrono::duration<double>(end - start).count();
            std::cout << "Loaded " << table_.size() << " records in " << std::fixed << std::setprecision(2)
                << sec << " s (" << std::setprecision(0) << (cfg.insertcount / sec) << " ops/s)\n";
        }

        // Open results file header
        {
            std::ofstream out(outfile_);
            if (out) {
                auto now = std::chrono::system_clock::now(); auto tt = std::chrono::system_clock::to_time_t(now);
                out << "Multithreaded YCSB MOPS Benchmark Results\n";
                out << "Generated at: " << std::put_time(std::localtime(&tt), "%Y-%m-%d %H:%M:%S") << "\n";
                out << "Threads: " << cfg.threadcount << "\n";
                out << "Records: " << cfg.recordcount << "\n";
                out << "Distribution: " << cfg.requestdistribution << "\n";
                out << "Runtime: " << WorkloadConfig::kFixedRuntimeSeconds << " seconds\n";
                out << std::string(50, '=') << "\n";
            }
        }

        // Run phase (true 400-thread concurrency)
        std::cout << "Starting " << WorkloadConfig::kFixedRuntimeSeconds << "s benchmark with "
            << cfg.threadcount << " threads...\n";
        std::vector<std::thread> workers; workers.reserve(cfg.threadcount);
        std::thread monitor(&MultiThreadedYCSBBenchmark::perfMonitor, this, std::cref(cfg));
        for (int i = 0; i < cfg.threadcount; ++i) workers.emplace_back(&MultiThreadedYCSBBenchmark::benchWorker, this, std::cref(cfg));

        monitor.join(); // sets stop_ when time is up
        for (auto& t : workers) t.join();

        // Summaries
        int64_t total_ops = total_ops_.load();
        double avg_mops = 0.0;
        {
            std::lock_guard<std::mutex> lk(mops_mu_);
            if (!mops_.empty()) {
                size_t warm = std::min<size_t>(3, mops_.size());
                double sum = std::accumulate(mops_.begin() + warm, mops_.end(), 0.0);
                size_t denom = (mops_.size() - warm);
                avg_mops = denom ? (sum / denom) : 0.0;
            }
        }

        std::cout << std::fixed << std::setprecision(3)
            << "Benchmark done. Total ops: " << total_ops
            << ", Avg MOPS (excl warmup): " << avg_mops << "\n";

        std::ofstream out(outfile_, std::ios::app);
        if (out) {
            out << "\n=== Final Results ===\n";
            out << "Total Operations: " << total_ops << "\n";
            out << std::fixed << std::setprecision(3)
                << "Average MOPS (excluding warmup): " << avg_mops << "\n";
            out << "Final HashTable Size: " << table_.size() << "\n";
        }
    }
};

// =========================
// CLI
// =========================
int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <workload_file> [--out file] [--stripes N]\n";
        return 1;
    }
    std::string workload = argv[1];
    std::string out = workload + std::string("_mops_results.txt");
    size_t stripes = 1024;
    for (int i = 2; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--out" && i + 1 < argc) out = argv[++i];
        else if (a == "--stripes" && i + 1 < argc) stripes = std::stoul(argv[++i]);
    }

    auto cfg = MultiThreadedYCSBBenchmark::parseConfig(workload);
    // force 400 threads unless workload overrides (keeps your original intent)
    if (cfg.threadcount <= 0) cfg.threadcount = 400;

    MultiThreadedYCSBBenchmark bench(out, stripes);
    bench.run(cfg);
    return 0;
}
