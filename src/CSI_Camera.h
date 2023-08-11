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

#include <atomic>
#include <pthread.h>
#include <opencv2/videoio.hpp>
#include <GenericTalker.h>
#include "CameraData.h"


/**
 * This class provides communication with CSI camera via OpenCV.
 * It utilises CUDA-enabled functions for rectification and disparity generation.
 * It constantly pulls images from the camera into a buffer,
 * but the latest images is only processed further when required.
 *
 * @note this class was tested with IMX219-83 camera.
 */
class CSI_Camera : public GenericTalker<CameraData>
{
public:
    /** Camera's focal length in metres. */
    static constexpr double FOCAL_LENGTH_M  = 0.0026;
    /** Camera's sensor width in metres. */
    static constexpr double SENSOR_WIDTH_M  = 0.00275968;
    /** Camera's sensor height in metres. */
    static constexpr double SENSOR_HEIGHT_M = 0.0036736;

    /**
     * The class constructor. Initialises all variables.
     */
    CSI_Camera();

    /**
     * Class destructor. Stops the camera and closes the internal thread.
     */
    virtual ~CSI_Camera();

    /**
     * Starts CSI camera with provided configuration.
     *  @param imageSize the size to which all images will be resized.
     *  @param framerate the camera's framerate in Hz.
     *  @param mode the mode of the camera - each camera may have different mode specification.
     *  @param id the id of the camera in case there are multiple cameras connected.
     *  @param flip the flip parameter. Usually 0 (no rotation) or 2 (180 deg).
     *  @param colour true to get BGR images, false for greyscale.
     *  @return true if the both the camera and insternal thread were started correctly.
     */
    bool startCamera(const cv::Size& imageSize, const uint8_t framerate, const uint8_t mode, 
                     const uint8_t id, const uint8_t flip, const bool colour);

    /**
     * Stops the camera by stopping the internal thread and releasing the camera itself.
     */
    void stopCamera();

    /**
     *  @return true if the camera has been initialised.
     */
    inline bool isInitialised() const
    {
        return mCapture.isOpened();
    }

    inline constexpr const cv::Size& getSize() const
    {
        return mImgSize;
    }

    /**
     * This is somewhat hard-coded for a specific camera. Provides image size and framerate for given mode.
     *  @param mode the mode for which information need to be obtained.
     *  @param[out] size the size of image for given mode.
     *  @return the framerate for given mode in Hz.
     */
    static uint8_t getSizeForMode(const uint8_t mode, cv::Size& size);

    /**
     *  @return camera ID
     */
    inline constexpr uint8_t getId() const
    {
        return mID;
    }

    /**
     *  @return true if image should be BGR, otherwise greyscale.
     */
    inline constexpr bool getColour() const
    {
        return mColour;
    }

protected:
    /**
     *  @return true if the main thread should be running.
     */
    inline bool isRun() const
    {
        return mThreadRun.load(std::memory_order_relaxed);
    }

    /**
     * The main body of the thread that constantly retrieves the latest image from CSI camera.
     */
    void grabThreadBody();

private:
    /**
     * Starts the new thread for pulling images from CSI camera.
     *  @param thread pointer to this class.
     *  @return nullptr.
     */
    static void* startGrabThread(void* thread);

    /** ID of this camera. */
    uint8_t mID;
    /** The size of requested images. */
    cv::Size mImgSize;
    /** The flag to indicate if requested image are to be BGR or greyscale. */
    bool mColour;
    /** Flag that indicates if the thread should be running or not. */
    std::atomic<bool> mThreadRun;
    /** Thread which pulls images from CSI camera. */
    pthread_t mThread;
    /** OpenCV wrapper which allows communication with CSI camera. */
    cv::VideoCapture mCapture;
};
