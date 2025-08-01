#include "streaming_audio_recorder.hpp"
#include <algorithm>
#include <cstring>
#include <iostream>
#include <cmath>
#include <thread>
#include <iomanip>

StreamingAudioRecorder::StreamingAudioRecorder(int device_index, int sample_rate)
    : device_index(device_index), sample_rate(sample_rate) {
    // Initialize ring buffer for 10 seconds of audio at target sample rate
    ringBufferSize = target_sample_rate * 10;
    ringBuffer.resize(ringBufferSize, 0.0f);
}

StreamingAudioRecorder::~StreamingAudioRecorder() {
    stopStreamingSession();
}

void StreamingAudioRecorder::setVADDetector(std::shared_ptr<VADDetector> detector) {
    sileroVAD = detector;
}


void StreamingAudioRecorder::startStreamingSession(
    SegmentCallback callback,
    double max_duration,
    double silence_threshold,
    double pre_speech_buffer) {
    
    if (isStreaming) {
        std::cerr << "Streaming session already active" << std::endl;
        return;
    }
    
    segmentCallback = callback;
    maxSessionDuration = max_duration;
    silenceThresholdSec = silence_threshold;
    preSpeechBufferSec = pre_speech_buffer;
    
    // Reset state
    vadState = VADState::WAITING_FOR_SPEECH;
    writePos = 0;
    readPos = 0;
    silenceFrames = 0;
    currentSegmentId = 0;
    sessionStartTime = std::chrono::steady_clock::now();
    isStreaming = true;
    
    // Start processing thread
    processingThread = std::thread(&StreamingAudioRecorder::processStreamingAudio, this);
    
    // Configure and start audio stream
    PaStreamParameters inputParams;
    inputParams.device = device_index;
    inputParams.channelCount = 1;
    inputParams.sampleFormat = paFloat32;
    inputParams.suggestedLatency = Pa_GetDeviceInfo(device_index)->defaultLowInputLatency;
    inputParams.hostApiSpecificStreamInfo = nullptr;
    
    PaError err = Pa_OpenStream(
        &stream,
        &inputParams,
        nullptr,  // No output
        sample_rate,
        256,  // Frames per buffer (16ms at 16kHz)
        paClipOff,
        &StreamingAudioRecorder::streamingAudioCallback,
        this
    );
    
    if (err != paNoError) {
        std::cerr << "Failed to open stream: " << Pa_GetErrorText(err) << std::endl;
        isStreaming = false;
        return;
    }
    
    err = Pa_StartStream(stream);
    if (err != paNoError) {
        std::cerr << "Failed to start stream: " << Pa_GetErrorText(err) << std::endl;
        Pa_CloseStream(stream);
        stream = nullptr;
        isStreaming = false;
        return;
    }
    
    std::cout << "Streaming ASR session started (max duration: " << max_duration << "s)" << std::endl;
    std::cout << "Listening for speech..." << std::endl;
}

void StreamingAudioRecorder::stopStreamingSession() {
    if (!isStreaming) {
        return;
    }
    
    isStreaming = false;
    
    // Stop audio stream
    if (stream) {
        Pa_StopStream(stream);
        Pa_CloseStream(stream);
        stream = nullptr;
    }
    
    // Wait for processing thread
    if (processingThread.joinable()) {
        processingThread.join();
    }
    
    // Process any remaining audio in buffer
    if (vadState == VADState::RECORDING_SPEECH || vadState == VADState::SILENCE_DETECTED) {
        createAndQueueSegment();
    }
    
    std::cout << "\nStreaming session ended." << std::endl;
}

void StreamingAudioRecorder::waitForCompletion() {
    while (isStreaming) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        
        if (checkSessionTimeout()) {
            std::cout << "\nSession timeout reached (" << maxSessionDuration << "s)" << std::endl;
            stopStreamingSession();
            break;
        }
    }
}

void StreamingAudioRecorder::processStreamingAudio() {
    // This thread monitors the session and processes VAD state
    while (isStreaming) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        
        // Check for session timeout
        if (checkSessionTimeout()) {
            isStreaming = false;
            break;
        }
    }
}

void StreamingAudioRecorder::processAudioFrame(const float* samples, size_t numSamples) {
    // Resample if needed
    std::vector<float> resampled;
    const float* processingSamples = samples;
    size_t processingNumSamples = numSamples;
    
    if (sample_rate != target_sample_rate) {
        resampleAudio(samples, numSamples, resampled);
        processingSamples = resampled.data();
        processingNumSamples = resampled.size();
    }
    
    // Add samples to ring buffer
    addToRingBuffer(processingSamples, processingNumSamples);
    
    // Detect speech using energy-based VAD
    bool isSpeech = detectSpeech(processingSamples, processingNumSamples);
    
    // Update VAD state machine
    switch (vadState) {
        case VADState::WAITING_FOR_SPEECH:
            if (isSpeech) {
                vadState = VADState::RECORDING_SPEECH;
                // Include pre-speech buffer
                size_t preSpeechSamples = static_cast<size_t>(preSpeechBufferSec * target_sample_rate);
                speechStartPos = getBufferPosition(preSpeechSamples);
                silenceFrames = 0;
                // Clear audio level display and show speech detection
                std::cout << "\r" << std::string(60, ' ') << "\r";
                std::cout << "[Speech detected]" << std::flush;
            }
            break;
            
        case VADState::RECORDING_SPEECH:
            if (!isSpeech) {
                vadState = VADState::SILENCE_DETECTED;
                silenceFrames = 1;
            }
            break;
            
        case VADState::SILENCE_DETECTED:
            if (!isSpeech) {
                silenceFrames++;
                size_t silenceThresholdFrames = static_cast<size_t>(
                    silenceThresholdSec * target_sample_rate / processingNumSamples
                );
                
                if (silenceFrames >= silenceThresholdFrames) {
                    std::cout << " [Silence detected, processing segment]" << std::flush;
                    createAndQueueSegment();
                    vadState = VADState::WAITING_FOR_SPEECH;
                }
            } else {
                // Speech resumed, go back to recording
                vadState = VADState::RECORDING_SPEECH;
                silenceFrames = 0;
            }
            break;
    }
}

void StreamingAudioRecorder::addToRingBuffer(const float* samples, size_t numSamples) {
    std::lock_guard<std::mutex> lock(bufferMutex);
    
    size_t currentWritePos = writePos.load();
    
    for (size_t i = 0; i < numSamples; ++i) {
        ringBuffer[currentWritePos] = samples[i];
        currentWritePos = (currentWritePos + 1) % ringBufferSize;
    }
    
    writePos.store(currentWritePos);
}

std::vector<float> StreamingAudioRecorder::extractFromRingBuffer(size_t startPos, size_t numSamples) {
    std::lock_guard<std::mutex> lock(bufferMutex);
    
    std::vector<float> segment;
    segment.reserve(numSamples);
    
    size_t pos = startPos;
    for (size_t i = 0; i < numSamples; ++i) {
        segment.push_back(ringBuffer[pos]);
        pos = (pos + 1) % ringBufferSize;
    }
    
    return segment;
}

size_t StreamingAudioRecorder::getBufferPosition(size_t samplesBack) const {
    size_t currentPos = writePos.load();
    if (samplesBack >= ringBufferSize) {
        samplesBack = ringBufferSize - 1;
    }
    
    if (currentPos >= samplesBack) {
        return currentPos - samplesBack;
    } else {
        return ringBufferSize - (samplesBack - currentPos);
    }
}

size_t StreamingAudioRecorder::calculateBufferDistance(size_t from, size_t to) const {
    if (to >= from) {
        return to - from;
    } else {
        return ringBufferSize - from + to;
    }
}

void StreamingAudioRecorder::createAndQueueSegment() {
    size_t currentPos = writePos.load();
    size_t segmentLength = calculateBufferDistance(speechStartPos, currentPos);
    
    if (segmentLength == 0) {
        return;
    }
    
    // Extract audio segment
    std::vector<float> samples = extractFromRingBuffer(speechStartPos, segmentLength);
    
    // Create segment
    AudioSegment segment;
    segment.samples = std::move(samples);
    segment.segment_id = currentSegmentId++;
    segment.is_final = !isStreaming;
    
    // Calculate timestamps
    auto now = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - sessionStartTime).count();
    segment.timestamp_end = elapsed / 1000.0;
    segment.timestamp_start = segment.timestamp_end - (segmentLength / static_cast<double>(sample_rate));
    
    // Call the callback
    if (segmentCallback) {
        segmentCallback(segment);
    }
}

bool StreamingAudioRecorder::checkSessionTimeout() const {
    auto now = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - sessionStartTime).count();
    return elapsed >= maxSessionDuration;
}

bool StreamingAudioRecorder::detectSpeech(const float* samples, size_t numSamples) {
    if (!useEnergyVAD && sileroVAD) {
        return detectSpeechSilero(samples, numSamples);
    } else {
        return detectSpeechEnergy(samples, numSamples);
    }
}

bool StreamingAudioRecorder::detectSpeechEnergy(const float* samples, size_t numSamples) {
    // Calculate RMS energy
    float sum = 0.0f;
    float max_val = 0.0f;
    
    for (size_t i = 0; i < numSamples; ++i) {
        float abs_val = std::abs(samples[i]);
        max_val = std::max(max_val, abs_val);
        sum += samples[i] * samples[i];
    }
    
    float rms = std::sqrt(sum / numSamples);
    
    // Debug output every second
    static int frame_count = 0;
    static float max_rms = 0.0f;
    frame_count++;
    max_rms = std::max(max_rms, rms);
    
    if (frame_count % (target_sample_rate / 256) == 0) {  // Approximately once per second
        std::cout << "\r[Audio Level: RMS=" << std::fixed << std::setprecision(4) << rms 
                  << " Peak=" << max_val << " MaxRMS=" << max_rms << "]" << std::flush;
        max_rms = 0.0f;
    }
    
    // Use RMS for VAD decision
    return rms > vadThreshold;
}

int StreamingAudioRecorder::streamingAudioCallback(
    const void* inputBuffer,
    void* outputBuffer,
    unsigned long framesPerBuffer,
    const PaStreamCallbackTimeInfo* timeInfo,
    PaStreamCallbackFlags statusFlags,
    void* userData) {
    
    StreamingAudioRecorder* recorder = static_cast<StreamingAudioRecorder*>(userData);
    const float* input = static_cast<const float*>(inputBuffer);
    
    if (!recorder->isStreaming) {
        return paComplete;
    }
    
    if (input) {
        recorder->processAudioFrame(input, framesPerBuffer);
    }
    
    return paContinue;
}

void StreamingAudioRecorder::resampleAudio(const float* input, size_t inputSamples, std::vector<float>& output) {
    // Simple linear interpolation resampling
    double ratio = static_cast<double>(target_sample_rate) / sample_rate;
    size_t outputSamples = static_cast<size_t>(inputSamples * ratio);
    output.resize(outputSamples);
    
    for (size_t i = 0; i < outputSamples; ++i) {
        double srcIndex = i / ratio;
        size_t srcIndexInt = static_cast<size_t>(srcIndex);
        double fraction = srcIndex - srcIndexInt;
        
        if (srcIndexInt + 1 < inputSamples) {
            // Linear interpolation between two samples
            output[i] = input[srcIndexInt] * (1.0 - fraction) + input[srcIndexInt + 1] * fraction;
        } else {
            // Use last sample
            output[i] = input[inputSamples - 1];
        }
    }
}

bool StreamingAudioRecorder::detectSpeechSilero(const float* samples, size_t numSamples) {
    if (!sileroVAD) {
        return false;
    }
    
    // Silero VAD expects 512 samples at 16kHz (32ms window)
    const size_t windowSize = 512;
    static std::vector<float> accumBuffer;
    static float lastProb = 0.0f;
    
    // Accumulate samples
    accumBuffer.insert(accumBuffer.end(), samples, samples + numSamples);
    
    bool isSpeech = false;
    
    // Process complete windows
    while (accumBuffer.size() >= windowSize) {
        std::vector<float> window(accumBuffer.begin(), accumBuffer.begin() + windowSize);
        float prob = sileroVAD->detectVAD(window);
        
        // Apply trigger/stop thresholds
        if (vadState == VADState::WAITING_FOR_SPEECH) {
            isSpeech = prob > vadTriggerThreshold;
        } else {
            isSpeech = prob > vadStopThreshold;
        }
        
        lastProb = prob;
        
        // Remove processed window
        accumBuffer.erase(accumBuffer.begin(), accumBuffer.begin() + windowSize);
    }
    
    // Debug output
    static int debugCounter = 0;
    if (++debugCounter % 30 == 0) {  // Every ~1 second
        std::cout << "\r[Silero VAD: Prob=" << std::fixed << std::setprecision(3) << lastProb 
                  << " State=" << (isSpeech ? "SPEECH" : "SILENCE") << "]" << std::flush;
    }
    
    return isSpeech;
}