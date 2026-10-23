#pragma once
// Plain C++ frame collector: no pybind11 in here, so it is reusable from any C++ code.
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <atomic>
#include <chrono>
#include <string>

class FrameCollector
{
public:
    // Takes a PyBullet RGBA frame (H x W x 4, uint8), converts it to BGR into `dst`
    // (caller-allocated H x W x 3) and, if `path` is non-empty, writes it to disk.
    // Returns false if the file could not be written.
    bool collect(const cv::Mat& rgba, cv::Mat& dst, const std::string& path)
    {
        auto t0 = std::chrono::steady_clock::now();
        cv::cvtColor(rgba, dst, cv::COLOR_RGBA2BGR);
        bool ok = path.empty() ? true : cv::imwrite(path, dst, {cv::IMWRITE_PNG_COMPRESSION, 1});
        auto us = std::chrono::duration_cast<std::chrono::microseconds>(
                      std::chrono::steady_clock::now() - t0).count();
        frames_ += 1;
        micros_ += us;
        return ok;
    }
    // Application side: read a dataset / follow-up photo from disk as BGR.
    // Returns an empty Mat if the file is missing or unreadable (same as cv::imread).
    cv::Mat load(const std::string& path)
    {
        auto t0 = std::chrono::steady_clock::now();
        cv::Mat img = cv::imread(path, cv::IMREAD_COLOR);
        add_time(t0, loads_, load_micros_);
        return img;
    }

    // Application side: write a BGR image (JPEG quality only applies to .jpg/.jpeg).
    bool save(const cv::Mat& bgr, const std::string& path, int jpeg_quality)
    {
        auto t0 = std::chrono::steady_clock::now();
        bool ok = cv::imwrite(path, bgr, {cv::IMWRITE_JPEG_QUALITY, jpeg_quality});
        add_time(t0, saves_, save_micros_);
        return ok;
    }

    long long frames() const { return frames_; }
    double mean_ms() const { return frames_ ? micros_ / 1000.0 / frames_ : 0.0; }
    long long loads() const { return loads_; }
    long long saves() const { return saves_; }
    double mean_load_ms() const { return loads_ ? load_micros_ / 1000.0 / loads_ : 0.0; }
    double mean_save_ms() const { return saves_ ? save_micros_ / 1000.0 / saves_ : 0.0; }

private:
    static void add_time(std::chrono::steady_clock::time_point t0,
                         std::atomic<long long>& n, std::atomic<long long>& us)
    {
        n += 1;
        us += std::chrono::duration_cast<std::chrono::microseconds>(
                  std::chrono::steady_clock::now() - t0).count();
    }

    std::atomic<long long> frames_{0};
    std::atomic<long long> micros_{0};
    std::atomic<long long> loads_{0}, saves_{0}, load_micros_{0}, save_micros_{0};
};
