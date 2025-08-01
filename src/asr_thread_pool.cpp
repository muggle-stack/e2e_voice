#include "asr_thread_pool.hpp"
#include <iostream>
#include <chrono>
#include <sstream>

ASRThreadPool::ASRThreadPool(size_t num_threads) : numThreads(num_threads) {
    workers.reserve(numThreads);
}

ASRThreadPool::~ASRThreadPool() {
    stop();
}

void ASRThreadPool::initialize(std::shared_ptr<ASRModel> model) {
    asrModel = model;
}

void ASRThreadPool::setResultCallback(ResultCallback callback) {
    std::lock_guard<std::mutex> lock(callbackMutex);
    resultCallback = callback;
}

void ASRThreadPool::start() {
    if (running) {
        return;
    }
    
    running = true;
    stopRequested = false;
    
    // Create worker threads
    for (size_t i = 0; i < numThreads; ++i) {
        workers.emplace_back(&ASRThreadPool::workerThread, this);
    }
    
    std::cout << "ASR thread pool started with " << numThreads << " threads" << std::endl;
}

void ASRThreadPool::stop() {
    if (!running) {
        return;
    }
    
    // Signal stop
    {
        std::lock_guard<std::mutex> lock(queueMutex);
        stopRequested = true;
    }
    queueCV.notify_all();
    
    // Wait for all threads to finish
    for (auto& thread : workers) {
        if (thread.joinable()) {
            thread.join();
        }
    }
    
    workers.clear();
    running = false;
    
    // Clear any remaining tasks
    std::queue<AudioSegment> empty;
    std::swap(taskQueue, empty);
}

void ASRThreadPool::processSegment(const AudioSegment& segment) {
    if (!running || !asrModel) {
        std::cerr << "ASR thread pool not initialized or not running" << std::endl;
        return;
    }
    
    {
        std::lock_guard<std::mutex> lock(queueMutex);
        taskQueue.push(segment);
    }
    queueCV.notify_one();
}

size_t ASRThreadPool::getQueueSize() const {
    std::lock_guard<std::mutex> lock(queueMutex);
    return taskQueue.size();
}

void ASRThreadPool::workerThread() {
    while (true) {
        AudioSegment segment;
        
        // Wait for a task
        {
            std::unique_lock<std::mutex> lock(queueMutex);
            queueCV.wait(lock, [this] {
                return !taskQueue.empty() || stopRequested;
            });
            
            if (stopRequested && taskQueue.empty()) {
                break;
            }
            
            if (!taskQueue.empty()) {
                segment = std::move(taskQueue.front());
                taskQueue.pop();
            }
        }
        
        // Process the segment
        if (!segment.samples.empty()) {
            processSegmentInternal(segment);
        }
    }
}

void ASRThreadPool::processSegmentInternal(const AudioSegment& segment) {
    auto start_time = std::chrono::steady_clock::now();
    
    try {
        // Redirect cout to suppress performance output
        std::stringstream buffer;
        std::streambuf* old = std::cout.rdbuf(buffer.rdbuf());
        
        // Perform ASR recognition
        std::string text = asrModel->recognize(segment.samples.data(), segment.samples.size());
        
        // Restore cout
        std::cout.rdbuf(old);
        
        auto end_time = std::chrono::steady_clock::now();
        auto processing_time = std::chrono::duration_cast<std::chrono::milliseconds>(
            end_time - start_time).count() / 1000.0;
        
        // Create result
        ASRResult result;
        result.text = text;
        result.segment_id = segment.segment_id;
        result.timestamp_start = segment.timestamp_start;
        result.timestamp_end = segment.timestamp_end;
        result.processing_time = processing_time;
        
        // Call the callback
        {
            std::lock_guard<std::mutex> lock(callbackMutex);
            if (resultCallback) {
                resultCallback(result);
            }
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Error processing segment " << segment.segment_id 
                  << ": " << e.what() << std::endl;
    }
}