#pragma once

#include <string>
#include <vector>
#include <memory>
#include <chrono>

/**
 * 阿里云实时ASR客户端 - 一句话识别API
 *
 * 适用于短语音（<60秒）的实时识别
 * 延迟约1-2秒，适合实时对话场景
 * 提供与ASRModel相同的recognize()接口
 */
class ASRRealtimeClient {
public:
    struct Config {
        // 必需参数（二选一）
        // 方式1：使用AccessKey（自动获取Token，推荐生产环境）
        std::string access_key_id;        // 阿里云AccessKey ID
        std::string access_key_secret;    // 阿里云AccessKey Secret

        // 方式2：直接使用Token（适合快速测试）
        std::string token;                // 直接提供的Token（24小时有效）

        // 必需参数
        std::string appkey;               // 语音服务AppKey

        // 可选参数
        std::string region = "cn-shanghai";  // 服务区域
        int sample_rate = 16000;             // 采样率 (8000/16000)
        bool enable_punctuation = true;      // 启用标点符号
        bool enable_itn = true;              // 启用数字转换(ITN)
        bool enable_intermediate_result = false;  // 启用中间结果（一句话识别不支持）

        // Token缓存配置
        int token_expire_seconds = 86400;    // Token有效期（24小时）

        // 请求配置
        int timeout_seconds = 10;            // 请求超时（秒）
        int max_retries = 2;                 // 最大重试次数

        // 环境变量加载
        bool load_from_env = true;           // 从环境变量加载配置
    };

    ASRRealtimeClient(const Config& config);
    ASRRealtimeClient();  // 使用默认配置
    ~ASRRealtimeClient();

    /**
     * 初始化（获取Token）
     * @return true if successful
     */
    bool initialize();

    /**
     * 核心识别接口 - 与ASRModel::recognize()接口一致
     * @param audio 16kHz单声道音频数据
     * @return 识别结果文本
     */
    std::string recognize(const std::vector<float>& audio);

    /**
     * 从音频数据识别（支持指定采样率）
     * @param audio 音频数据
     * @param sample_rate 采样率
     * @return 识别结果文本
     */
    std::string recognize(const std::vector<float>& audio, int sample_rate);

    /**
     * 检查配置是否有效
     */
    bool isConfigured() const;

    /**
     * 获取最后一次错误信息
     */
    std::string getLastError() const { return last_error_; }

    /**
     * 手动刷新Token（通常不需要，会自动刷新）
     */
    bool refreshToken();

private:
    class Impl;
    std::unique_ptr<Impl> impl_;

    Config config_;
    std::string last_error_;

    // Token管理
    std::string cached_token_;
    std::chrono::system_clock::time_point token_expire_time_;

    // 加载环境变量配置
    void loadConfigFromEnv();

    // Token管理
    std::string getToken();
    bool isTokenExpired() const;
    std::string requestNewToken();

    // 音频处理
    std::vector<uint8_t> convertToWav(const std::vector<float>& audio, int sample_rate);

    // HTTP请求
    std::string httpPost(const std::string& url, const std::string& body,
                        const std::vector<std::pair<std::string, std::string>>& headers);
    std::string httpPostBinary(const std::string& url, const std::vector<uint8_t>& data,
                              const std::vector<std::pair<std::string, std::string>>& headers);
};
