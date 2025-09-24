#ifndef MEL_FILTERBANK_HPP_
#define MEL_FILTERBANK_HPP_

#include <vector>
#include <cstdint>
#include <complex>

namespace speaker_recognition {

class MelFilterbank {
public:
    // Compute Mel-frequency spectrogram
    static std::vector<float> ComputeMelSpectrogram(
        const std::vector<float>& samples,
        int32_t sample_rate,
        int32_t& num_frames,
        int32_t& num_mel_bins);

    // Wrapper function for compatibility
    static std::vector<float> ComputeFbank(
        const std::vector<float>& samples,
        int32_t sample_rate,
        std::vector<int64_t>& shape);

private:
    // Internal helper functions
    static void FFT(std::vector<std::complex<float>>& data);
    static void FFTRecursive(std::vector<std::complex<float>>& buf,
                            std::vector<std::complex<float>>& out,
                            int n, int step);
    static int NextPow2(int n);
    static float HzToMel(float hz);
    static float MelToHz(float mel);
    static std::vector<std::vector<float>> CreateMelFilterbank(
        int num_filters, int fft_size, float sample_rate);
    static void PreEmphasis(std::vector<float>& signal, float coeff);
    static void ApplyHammingWindow(std::vector<float>& frame);
    static std::vector<float> ComputePowerSpectrum(
        const std::vector<std::complex<float>>& fft_result);
    static std::vector<float> ApplyMelFilterbank(
        const std::vector<float>& power_spec,
        const std::vector<std::vector<float>>& filterbank);
};

} // namespace speaker_recognition

#endif // MEL_FILTERBANK_HPP_