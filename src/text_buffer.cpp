#include "text_buffer.hpp"
#include <algorithm>

const std::string TextBuffer::CHINESE_PUNCTUATION = "。！？；.!?;";

TextBuffer::TextBuffer() : stop_flag_(false) {
}

TextBuffer::~TextBuffer() {
    stop();
}

void TextBuffer::addText(const std::string& text) {
    if (stop_flag_) return;
    
    std::lock_guard<std::mutex> lock(mutex_);
    buffer_ += text;
    processBuffer();
}

std::string TextBuffer::getNextSentence() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (sentences_.empty()) {
        return "";
    }
    
    std::string sentence = sentences_.front();
    sentences_.pop();
    return sentence;
}

bool TextBuffer::hasSentence() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return !sentences_.empty();
}

void TextBuffer::clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    buffer_.clear();
    while (!sentences_.empty()) {
        sentences_.pop();
    }
}

void TextBuffer::stop() {
    std::lock_guard<std::mutex> lock(mutex_);
    stop_flag_ = true;

    // 将残留的 buffer_ 作为最后一个句子
    if (!buffer_.empty()) {
        std::string trimmed = buffer_;
        trimmed.erase(0, trimmed.find_first_not_of(" \t\n\r"));
        trimmed.erase(trimmed.find_last_not_of(" \t\n\r") + 1);
        if (!trimmed.empty()) {
            sentences_.push(trimmed);
        }
        buffer_.clear();
    }

    cv_.notify_all();
}

void TextBuffer::processBuffer() {
    // Find complete sentences in buffer
    std::string remaining_buffer;
    std::string current_sentence;
    
    for (size_t i = 0; i < buffer_.size(); ) {
        // Handle UTF-8 characters
        int char_len = 1;
        unsigned char ch = buffer_[i];
        if ((ch & 0x80) == 0) char_len = 1;
        else if ((ch & 0xE0) == 0xC0) char_len = 2;
        else if ((ch & 0xF0) == 0xE0) char_len = 3;
        else if ((ch & 0xF8) == 0xF0) char_len = 4;
        
        if (i + char_len <= buffer_.size()) {
            std::string utf8_char = buffer_.substr(i, char_len);
            current_sentence += utf8_char;
            
            // Check if this character ends a sentence
            if (char_len == 1) {
                // ASCII punctuation
                char c = utf8_char[0];
                if (isEndOfSentence(c)) {
                    // Trim whitespace and add to sentences queue
                    std::string trimmed = current_sentence;
                    trimmed.erase(0, trimmed.find_first_not_of(" \t\n\r"));
                    trimmed.erase(trimmed.find_last_not_of(" \t\n\r") + 1);
                    
                    if (!trimmed.empty()) {
                        sentences_.push(trimmed);
                    }
                    current_sentence.clear();
                }
            } else {
                // Check for Chinese punctuation
                if (CHINESE_PUNCTUATION.find(utf8_char) != std::string::npos) {
                    // Trim whitespace and add to sentences queue
                    std::string trimmed = current_sentence;
                    trimmed.erase(0, trimmed.find_first_not_of(" \t\n\r"));
                    trimmed.erase(trimmed.find_last_not_of(" \t\n\r") + 1);
                    
                    if (!trimmed.empty()) {
                        sentences_.push(trimmed);
                    }
                    current_sentence.clear();
                }
            }
        }
        i += char_len;
    }
    
    // Keep any remaining incomplete sentence in buffer
    buffer_ = current_sentence;
}

bool TextBuffer::isEndOfSentence(char c) const {
    return c == '.' || c == '!' || c == '?' || c == ';';
}
