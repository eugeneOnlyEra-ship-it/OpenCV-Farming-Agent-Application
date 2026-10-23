#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cstring>
#include <stdexcept>
#include "NewComponent.h"
#include "FrameCollector.h"

namespace py = pybind11;

// One shared collector for the whole process (the camera agents call it from
// the same Python thread, so there is nothing to manage on the Python side).
static FrameCollector g_collector;

// collect_frame(rgba, path="") -> bgr
//   rgba : numpy uint8 array (H, W, 4) straight from p.getCameraImage
//   path : optional PNG/JPG path; the file is written here, in C++
//   bgr  : new numpy uint8 array (H, W, 3), ready for cv2 / classify_pod
// The GIL is released during conversion + disk write, so the physics loop and
// the inference thread keep running while the frame is being collected.
static py::array_t<uint8_t> collect_frame(
    py::array_t<uint8_t, py::array::c_style | py::array::forcecast> rgba,
    const std::string& path)
{
    if (rgba.ndim() != 3 || rgba.shape(2) != 4)
        throw std::invalid_argument("collect_frame expects a (H, W, 4) uint8 RGBA array");

    const int h = static_cast<int>(rgba.shape(0));
    const int w = static_cast<int>(rgba.shape(1));
    py::array_t<uint8_t> bgr({h, w, 3});

    cv::Mat src(h, w, CV_8UC4, const_cast<uint8_t*>(rgba.data()));
    cv::Mat dst(h, w, CV_8UC3, bgr.mutable_data());   // writes straight into the numpy buffer

    bool ok;
    {
        py::gil_scoped_release release;
        ok = g_collector.collect(src, dst, path);
    }
    if (!ok)
        throw std::runtime_error("collect_frame: could not write " + path);
    return bgr;
}

// load_image(path) -> bgr ndarray (H, W, 3) or None if the file can't be read
// (same contract as cv2.imread, so existing "if img is None" checks keep working).
static py::object load_image(const std::string& path)
{
    cv::Mat img;
    {
        py::gil_scoped_release release;
        img = g_collector.load(path);
    }
    if (img.empty())
        return py::none();
    py::array_t<uint8_t> out({img.rows, img.cols, img.channels()});
    std::memcpy(out.mutable_data(), img.data, static_cast<size_t>(img.total()) * img.elemSize());
    return std::move(out);
}

// save_image(bgr, path, jpeg_quality=92) -> bool
static bool save_image(
    py::array_t<uint8_t, py::array::c_style | py::array::forcecast> bgr,
    const std::string& path, int jpeg_quality)
{
    if (bgr.ndim() != 3 || bgr.shape(2) != 3)
        throw std::invalid_argument("save_image expects a (H, W, 3) uint8 BGR array");
    cv::Mat src(static_cast<int>(bgr.shape(0)), static_cast<int>(bgr.shape(1)), CV_8UC3,
                const_cast<uint8_t*>(bgr.data()));
    py::gil_scoped_release release;
    return g_collector.save(src, path, jpeg_quality);
}

PYBIND11_MODULE(pod_capture, m)
{
    m.doc() = "C++ frame capture for the pod-farm camera agents";

    py::class_<NewComponent>(m, "NewComponent")
        .def(py::init<>())
        .def("greet", &NewComponent::greet);

    m.def("collect_frame", &collect_frame, py::arg("rgba"), py::arg("path") = "",
          "Convert a PyBullet RGBA frame to BGR (and optionally save it) in C++.");
    m.def("load_image", &load_image, py::arg("path"),
          "Read an application/dataset image from disk as BGR in C++ (None if unreadable).");
    m.def("save_image", &save_image, py::arg("bgr"), py::arg("path"), py::arg("jpeg_quality") = 92,
          "Write a BGR image to disk in C++.");
    m.def("stats", [] {
        py::dict d;
        d["sim_frames"] = g_collector.frames();
        d["sim_mean_ms"] = g_collector.mean_ms();
        d["app_images_loaded"] = g_collector.loads();
        d["app_mean_load_ms"] = g_collector.mean_load_ms();
        d["app_images_saved"] = g_collector.saves();
        d["app_mean_save_ms"] = g_collector.mean_save_ms();
        return d;
    }, "Counters for both camera paths: simulation frames and application images.");
    m.def("frames_collected", [] { return g_collector.frames(); },
          "Number of frames collected in C++ so far.");
    m.def("mean_collect_ms", [] { return g_collector.mean_ms(); },
          "Mean C++ time per frame (convert + write) in milliseconds.");
}
