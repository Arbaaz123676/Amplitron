#include "audio/engine/analyzer_capture.h"

#include <algorithm>
#include <cstring>

namespace Amplitron {

AnalyzerCapture::AnalyzerCapture() {
    capture_input_.fill(0.0f);
    capture_output_.fill(0.0f);
    snapshot_input_.fill(0.0f);
    snapshot_output_.fill(0.0f);
}

void AnalyzerCapture::set_analyzer_enabled(bool enabled) {
    enabled_.store(enabled, std::memory_order_release);
}

bool AnalyzerCapture::is_analyzer_enabled() const {
    return enabled_.load(std::memory_order_acquire);
}

uint64_t AnalyzerCapture::get_analyzer_sequence() const {
    return sequence_.load(std::memory_order_acquire);
}

bool AnalyzerCapture::copy_analyzer_snapshot(float* input_dest, float* output_dest,
                                             int sample_count) const {
    if (!input_dest || !output_dest || sample_count <= 0) {
        return false;
    }

    const int count = std::min(sample_count, ANALYZER_FFT_SIZE);
    std::lock_guard<std::mutex> lock(mutex_);
    const uint64_t seq = sequence_.load(std::memory_order_relaxed);
    if (seq == 0) {
        return false;
    }

    std::memcpy(input_dest, snapshot_input_.data(), static_cast<size_t>(count) * sizeof(float));
    std::memcpy(output_dest, snapshot_output_.data(), static_cast<size_t>(count) * sizeof(float));
    return true;
}

void AnalyzerCapture::capture_input(const float* input, int count) {
    if (!enabled_.load(std::memory_order_relaxed)) return;

    int cap = capture_index_;
    for (int i = 0; i < count; ++i) {
        capture_input_[cap] = input[i];
        cap = (cap + 1) & ANALYZER_FFT_MASK;
    }
    capture_index_ = cap;
}

void AnalyzerCapture::capture_output(const float* output, int count) {
    if (!enabled_.load(std::memory_order_relaxed)) return;

    int cap = (capture_index_ - count) & ANALYZER_FFT_MASK;
    for (int i = 0; i < count; ++i) {
        capture_output_[cap] = output[i];
        cap = (cap + 1) & ANALYZER_FFT_MASK;
    }

    samples_since_publish_ += count;
    if (samples_since_publish_ >= ANALYZER_HOP_SIZE) {
        if (mutex_.try_lock()) {
            const int start = capture_index_;
            const int first_chunk = ANALYZER_FFT_SIZE - start;
            std::memcpy(snapshot_input_.data(), capture_input_.data() + start,
                        static_cast<size_t>(first_chunk) * sizeof(float));
            std::memcpy(snapshot_input_.data() + first_chunk, capture_input_.data(),
                        static_cast<size_t>(start) * sizeof(float));
            std::memcpy(snapshot_output_.data(), capture_output_.data() + start,
                        static_cast<size_t>(first_chunk) * sizeof(float));
            std::memcpy(snapshot_output_.data() + first_chunk, capture_output_.data(),
                        static_cast<size_t>(start) * sizeof(float));
            sequence_.fetch_add(1, std::memory_order_release);
            samples_since_publish_ = 0;
            mutex_.unlock();
        }
    }
}

bool AnalyzerCapture::register_pedal_analyzer(int node_id) {
    if (node_id < 0) return false;

    // Check if already registered
    for (int i = 0; i < MAX_PEDAL_ANALYZERS; ++i) {
        if (pedal_captures_[i].node_id_.load(std::memory_order_acquire) == node_id) {
            return true;
        }
    }

    // Find an empty slot
    for (int i = 0; i < MAX_PEDAL_ANALYZERS; ++i) {
        if (pedal_captures_[i].node_id_.load(std::memory_order_relaxed) == -1) {
            int expected = -1;
            if (pedal_captures_[i].node_id_.compare_exchange_strong(expected, node_id,
                                                                    std::memory_order_acq_rel)) {
                pedal_captures_[i].reset();
                return true;
            }
        }
    }

    return false;
}

void AnalyzerCapture::unregister_pedal_analyzer(int node_id) {
    if (node_id < 0) return;
    for (int i = 0; i < MAX_PEDAL_ANALYZERS; ++i) {
        int expected = node_id;
        if (pedal_captures_[i].node_id_.compare_exchange_strong(expected, -1,
                                                                std::memory_order_acq_rel)) {
            pedal_captures_[i].reset();
            return;
        }
    }
}

uint64_t AnalyzerCapture::get_pedal_analyzer_sequence(int node_id) const {
    for (int i = 0; i < MAX_PEDAL_ANALYZERS; ++i) {
        if (pedal_captures_[i].node_id_.load(std::memory_order_acquire) == node_id) {
            return pedal_captures_[i].sequence_.load(std::memory_order_acquire);
        }
    }
    return 0;
}

bool AnalyzerCapture::copy_pedal_analyzer_snapshot(int node_id, float* input_dest,
                                                   float* output_dest, int sample_count) const {
    if (!input_dest || !output_dest || sample_count <= 0) {
        return false;
    }
    for (int i = 0; i < MAX_PEDAL_ANALYZERS; ++i) {
        if (pedal_captures_[i].node_id_.load(std::memory_order_acquire) == node_id) {
            const auto& pc = pedal_captures_[i];
            const int count = std::min(sample_count, ANALYZER_FFT_SIZE);
            std::lock_guard<std::mutex> lock(pc.mutex_);
            const uint64_t seq = pc.sequence_.load(std::memory_order_relaxed);
            if (seq == 0) {
                return false;
            }
            std::memcpy(input_dest, pc.snapshot_input_.data(),
                        static_cast<size_t>(count) * sizeof(float));
            std::memcpy(output_dest, pc.snapshot_output_.data(),
                        static_cast<size_t>(count) * sizeof(float));
            return true;
        }
    }
    return false;
}

void AnalyzerCapture::capture_pedal(int node_id, const float* input, const float* output,
                                    int count) {
    for (int i = 0; i < MAX_PEDAL_ANALYZERS; ++i) {
        if (pedal_captures_[i].node_id_.load(std::memory_order_relaxed) == node_id) {
            auto& pc = pedal_captures_[i];
            int cap = pc.capture_index_.load(std::memory_order_relaxed);
            for (int s = 0; s < count; ++s) {
                pc.capture_input_[cap] = input[s];
                pc.capture_output_[cap] = output[s];
                cap = (cap + 1) & ANALYZER_FFT_MASK;
            }
            pc.capture_index_.store(cap, std::memory_order_relaxed);

            int current_samples = pc.samples_since_publish_.load(std::memory_order_relaxed) + count;
            pc.samples_since_publish_.store(current_samples, std::memory_order_relaxed);
            if (current_samples >= ANALYZER_HOP_SIZE) {
                if (pc.mutex_.try_lock()) {
                    const int start = pc.capture_index_.load(std::memory_order_relaxed);
                    const int first_chunk = ANALYZER_FFT_SIZE - start;
                    std::memcpy(pc.snapshot_input_.data(), pc.capture_input_.data() + start,
                                static_cast<size_t>(first_chunk) * sizeof(float));
                    std::memcpy(pc.snapshot_input_.data() + first_chunk, pc.capture_input_.data(),
                                static_cast<size_t>(start) * sizeof(float));
                    std::memcpy(pc.snapshot_output_.data(), pc.capture_output_.data() + start,
                                static_cast<size_t>(first_chunk) * sizeof(float));
                    std::memcpy(pc.snapshot_output_.data() + first_chunk, pc.capture_output_.data(),
                                static_cast<size_t>(start) * sizeof(float));
                    pc.sequence_.fetch_add(1, std::memory_order_release);
                    pc.samples_since_publish_.store(0, std::memory_order_relaxed);
                    pc.mutex_.unlock();
                }
            }
            break;
        }
    }
}

}  // namespace Amplitron
