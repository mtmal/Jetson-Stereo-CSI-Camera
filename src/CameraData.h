////////////////////////////////////////////////////////////////////////////////
// Copyright (C) 2023 Mateusz Malinowski
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
////////////////////////////////////////////////////////////////////////////////

#pragma once

#include <opencv2/core/cuda.hpp>

/**
 * Data structure for image data used in updating listeners.
 */
struct CameraData
{
    /** IDs of cameras that provided images. */
    std::vector<uint8_t> mID;
    /** Timestamp in ms of the associated image. */
    std::vector<double> mTimestamp;
    /** Images data allocated with SHARED buffer between CPU (cv::Mat) and GPU (cv::cuda::GpuMat). */
    std::vector<cv::cuda::HostMem> mImage;

    /**
     * Deep copy of the image buffer.
     *  @param[out] out result of a deep copy
     */
    void copyTo(CameraData& out)
    {
        out.mID = mID;
        out.mTimestamp = mTimestamp;
        out.mImage.clear();
        for (const cv::cuda::HostMem& img : mImage)
        {
            out.mImage.emplace_back(img.clone());
        }
    }
};