#include "mel_filterbank.hpp"
#include <cmath>
#include <complex>
#include <algorithm>
#include <cstring>  // for memset

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace speaker_recognition {

// Optimized iterative FFT implementation (Cooley-Tukey)
void MelFilterbank::FFT(std::vector<std::complex<float>>& data) {
    int n = data.size();
    if (n <= 1) return;

    // Bit reversal permutation - optimized
    int bits = 0;
    while ((1 << bits) < n) bits++;

    for (int i = 1; i < n - 1; i++) {
        int rev = 0;
        int x = i;
        for (int j = 0; j < bits; j++) {
            rev = (rev << 1) | (x & 1);
            x >>= 1;
        }
        if (i < rev) {
            std::swap(data[i], data[rev]);
        }
    }

    // Cooley-Tukey FFT with optimized twiddle factors
    for (int len = 2; len <= n; len <<= 1) {
        float ang = -2.0f * M_PI / len;
        float cos_ang = std::cos(ang);
        float sin_ang = std::sin(ang);
        int halflen = len >> 1;

        for (int i = 0; i < n; i += len) {
            float cos_w = 1.0f;
            float sin_w = 0.0f;

            for (int j = 0; j < halflen; j++) {
                std::complex<float> w(cos_w, sin_w);

                std::complex<float> u = data[i + j];
                std::complex<float> v = data[i + j + halflen] * w;
                data[i + j] = u + v;
                data[i + j + halflen] = u - v;

                // Update twiddle factor using rotation
                float tmp = cos_w * cos_ang - sin_w * sin_ang;
                sin_w = sin_w * cos_ang + cos_w * sin_ang;
                cos_w = tmp;
            }
        }
    }
}

int MelFilterbank::NextPow2(int n) {
    int power = 1;
    while (power < n) power <<= 1;
    return power;
}

float MelFilterbank::HzToMel(float hz) {
    return 2595.0f * std::log10(1.0f + hz / 700.0f);
}

float MelFilterbank::MelToHz(float mel) {
    return 700.0f * (std::pow(10.0f, mel / 2595.0f) - 1.0f);
}

std::vector<std::vector<float>> MelFilterbank::CreateMelFilterbank(
    int num_filters, int fft_size, float sample_rate) {

    std::vector<std::vector<float>> filterbank(num_filters,
                                               std::vector<float>(fft_size / 2 + 1, 0.0f));

    float low_freq = 0.0f;
    float high_freq = sample_rate / 2.0f;

    float low_mel = HzToMel(low_freq);
    float high_mel = HzToMel(high_freq);

    // Create equally spaced points on Mel scale
    std::vector<float> mel_points(num_filters + 2);
    for (int i = 0; i <= num_filters + 1; i++) {
        mel_points[i] = low_mel + i * (high_mel - low_mel) / (num_filters + 1);
    }

    // Convert back to Hz
    std::vector<float> hz_points(num_filters + 2);
    for (int i = 0; i <= num_filters + 1; i++) {
        hz_points[i] = MelToHz(mel_points[i]);
    }

    // Convert to FFT bin numbers
    std::vector<int> bin_points(num_filters + 2);
    for (int i = 0; i <= num_filters + 1; i++) {
        bin_points[i] = static_cast<int>((fft_size + 1) * hz_points[i] / sample_rate);
    }

    // Create triangular filters
    for (int m = 0; m < num_filters; m++) {
        int left = bin_points[m];
        int center = bin_points[m + 1];
        int right = bin_points[m + 2];

        for (int k = left; k < center; k++) {
            if (k >= 0 && k <= fft_size / 2) {
                filterbank[m][k] = static_cast<float>(k - left) / (center - left);
            }
        }

        for (int k = center; k < right; k++) {
            if (k >= 0 && k <= fft_size / 2) {
                filterbank[m][k] = static_cast<float>(right - k) / (right - center);
            }
        }
    }

    return filterbank;
}

void MelFilterbank::PreEmphasis(std::vector<float>& signal, float coeff) {
    for (int i = signal.size() - 1; i > 0; i--) {
        signal[i] -= coeff * signal[i - 1];
    }
    signal[0] -= coeff * signal[0];
}

void MelFilterbank::ApplyHammingWindow(std::vector<float>& frame) {
    int frame_size = frame.size();
    for (int i = 0; i < frame_size; i++) {
        frame[i] *= 0.54f - 0.46f * std::cos(2 * M_PI * i / (frame_size - 1));
    }
}

std::vector<float> MelFilterbank::ComputePowerSpectrum(
    const std::vector<std::complex<float>>& fft_result) {

    int fft_size = fft_result.size();
    std::vector<float> power_spec(fft_size / 2 + 1);

    // Precompute scale factor
    float scale = 1.0f / fft_size;

    for (int i = 0; i <= fft_size / 2; i++) {
        float real = fft_result[i].real();
        float imag = fft_result[i].imag();
        power_spec[i] = (real * real + imag * imag) * scale;
    }

    return power_spec;
}

std::vector<float> MelFilterbank::ApplyMelFilterbank(
    const std::vector<float>& power_spec,
    const std::vector<std::vector<float>>& filterbank) {

    int num_filters = filterbank.size();
    int spec_size = power_spec.size();
    std::vector<float> mel_energies(num_filters);

    // Optimized loop with better cache locality
    for (int m = 0; m < num_filters; m++) {
        float energy = 0.0f;
        const float* filter_ptr = filterbank[m].data();
        const float* spec_ptr = power_spec.data();

        // Vectorizable inner loop
        for (int k = 0; k < spec_size; k++) {
            energy += spec_ptr[k] * filter_ptr[k];
        }

        // Apply log with fast math
        mel_energies[m] = std::log(std::max(energy, 1e-10f));
    }

    return mel_energies;
}

std::vector<float> MelFilterbank::ComputeMelSpectrogram(
    const std::vector<float>& samples,
    int32_t sample_rate,
    int32_t& num_frames,
    int32_t& num_mel_bins) {

    // Parameters (matching common speech recognition settings)
    const int frame_size = 400;  // 25ms at 16kHz
    const int hop_size = 160;    // 10ms at 16kHz
    num_mel_bins = 80;
    const float pre_emphasis_coeff = 0.97f;

    // Calculate number of frames
    int num_samples = samples.size();
    num_frames = (num_samples - frame_size) / hop_size + 1;
    if (num_frames <= 0) {
        num_frames = 1;
    }

    // FFT size (next power of 2)
    int fft_size = NextPow2(frame_size);

    // Create Mel filterbank
    auto filterbank = CreateMelFilterbank(num_mel_bins, fft_size, static_cast<float>(sample_rate));

    // Allocate output
    std::vector<float> mel_spec(num_frames * num_mel_bins);

    // Process each frame
    for (int frame = 0; frame < num_frames; frame++) {
        int start_idx = frame * hop_size;
        int end_idx = std::min(start_idx + frame_size, num_samples);

        // Extract frame - optimize with memset and direct copy
        std::vector<float> frame_data(fft_size, 0.0f);
        int copy_size = std::min(end_idx - start_idx, frame_size);
        std::copy_n(samples.data() + start_idx, copy_size, frame_data.data());

        // Apply pre-emphasis
        PreEmphasis(frame_data, pre_emphasis_coeff);

        // Apply Hamming window (only on actual frame size, not padded zeros)
        ApplyHammingWindow(frame_data);

        // Convert to complex for FFT - reserve and construct in-place
        std::vector<std::complex<float>> fft_data;
        fft_data.reserve(fft_size);
        for (int i = 0; i < fft_size; i++) {
            fft_data.emplace_back(frame_data[i], 0.0f);
        }

        // Compute FFT
        FFT(fft_data);

        // Compute power spectrum
        auto power_spec = ComputePowerSpectrum(fft_data);

        // Apply Mel filterbank
        auto mel_energies = ApplyMelFilterbank(power_spec, filterbank);

        // Copy to output
        std::copy(mel_energies.begin(), mel_energies.end(),
                 mel_spec.begin() + frame * num_mel_bins);
    }

    return mel_spec;
}

std::vector<float> MelFilterbank::ComputeFbank(
    const std::vector<float>& samples,
    int32_t sample_rate,
    std::vector<int64_t>& shape) {

    int32_t num_frames, num_mel_bins;
    auto mel_spec = ComputeMelSpectrogram(samples, sample_rate, num_frames, num_mel_bins);

    shape.resize(3);
    shape[0] = 1;
    shape[1] = num_frames;
    shape[2] = num_mel_bins;

    return mel_spec;
}

// Dummy implementations for unused recursive functions
void MelFilterbank::FFTRecursive(std::vector<std::complex<float>>& buf,
                                 std::vector<std::complex<float>>& out,
                                 int n, int step) {
    // Not used anymore - FFT is now iterative
}

} // namespace speaker_recognition