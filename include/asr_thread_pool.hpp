#ifndef ASR_THREAD_POOL_HPP
#define ASR_THREAD_POOL_HPP

#include "asr_model.hpp"
#include "streaming_audio_recorder.hpp"
#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <functional>
#include <memory>

struct ASRResult {
    std::string text;
    size_t segment_id;
    double timestamp_start;
    double timestamp_end;
    double processing_time;
};

class ASRThreadPool {
public:
    using ResultCallback = std::function<void(const ASRResult&)>;
    
    ASRThreadPool(size_t num_threads = 2);
    ~ASRThreadPool();
    
    // Initialize with ASR model
    void initialize(std::shared_ptr<ASRModel> model);
    
    // Set callback for results
    void setResultCallback(ResultCallback callback);
    
    // Process an audio segment
    void processSegment(const AudioSegment& segment);
    
    // Start/stop the thread pool
    void start();
    void stop();
    
    // Get queue size
    size_t getQueueSize() const;

private:
    // Worker thread function
    void workerThread();
    
    // Process a single segment
    void processSegmentInternal(const AudioSegment& segment);
    
    // Thread pool
    std::vector<std::thread> workers;
    size_t numThreads;
    
    // Task queue
    std::queue<AudioSegment> taskQueue;
    mutable std::mutex queueMutex;
    std::condition_variable queueCV;
    
    // Control
    std::atomic<bool> running{false};
    std::atomic<bool> stopRequested{false};
    
    // ASR model (shared among threads)
    std::shared_ptr<ASRModel> asrModel;
    
    // Result callback
    ResultCallback resultCallback;
    std::mutex callbackMutex;
};

#endif // ASR_THREAD_POOL_HPP