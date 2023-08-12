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

#include <GenericTalker.h>
#include "CameraData.h"

class ICameraTalker : public GenericTalker<CameraData> 
{
public:
    /**
     * Starts the camera. A vector of camera ids is required for a multi-camera systems.
     *  @param imageSize the size to which all images will be resized, if it does not match the size of specified @p mode.
     *  @param framerate the camera's framerate in Hz.
     *  @param mode the mode of the camera - each camera may have different mode specification.
     *  @param ids the list of camera ids.
     *  @param flip the flip parameter. Usually 0 (no rotation) or 2 (180 deg).
     *  @param colour true to get BGR images, false for greyscale.
     *  @param rectify whether to request rectified/undistorted images.
     *	@return true if both cameras have started correctly.
     */
    virtual bool startCamera(const cv::Size& imageSize, const uint8_t framerate, const uint8_t mode, 
                             const std::vector<uint8_t>& ids, const uint8_t flip, const bool colour,
                             const bool rectify) = 0;

    /**
     * Stops the camera.
     */
    virtual void stopCamera() = 0;

    /**
     *  @return true if the camera has been initialised.
     */
    virtual bool isInitialised() const = 0;
};
