////////////////////////////////////////////////////////////////////////////////
// Copyright (C) 2024 Mateusz Malinowski
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

#include <string>
#include <opencv2/core/types.hpp>

/**
 * A structure with camera configuration parameters
 */
struct CameraConfig
{
    /** The size to which all images should be resized. */
    cv::Size mImageSize;
    /** Desired framerate in Hz. */
    uint8_t mFramerate;
    /** The camera-specific mode - each camera may have different mode specification. */
    uint8_t mMode;
    /** The ID of the camera. */
    uint8_t mID;
    /** The flip parameter. Usually 0 (no rotation) or 2 (180 deg). */
    uint8_t mFlip;
    /** True to get BGR images, false for greyscale. */
    bool mColour;
    /** True if images should be rectified. */
    bool mRectify;
    /** A path to images that should be loaded instead of starting the actual camera. */
    std::string mOfflineImages;
};