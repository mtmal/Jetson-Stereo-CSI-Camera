////////////////////////////////////////////////////////////////////////////////
// Copyright (C) 2021 Mateusz Malinowski
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

#ifndef __CSI_CAMERA_H__
#define __CSI_CAMERA_H__

#include <atomic>
#include <pthread.h>
#include <opencv2/core/cuda.hpp>
#include <opencv2/videoio.hpp>
#include <vpi/Stream.h>

/**
 * This class provides communication with CSI camera via OpenCV.
 * It utilises CUDA-enabled functions for rectification and disparity generation.
 * It constantly pulls images from the camera into a buffer,
 * but the latest images is only processed further when required.
 *
 * @note this class was tested with IMX219-83 camera.
 */
class CSI_Camera
{
public:
    /** Camera's focal length in metres. */
    static constexpr double FOCAL_LENGTH_M  = 0.0026;
    /** Camera's sensor width in metres. */
    static constexpr double SENSOR_WIDTH_M  = 0.00275968;
    /** Camera's sensor height in metres. */
    static constexpr double SENSOR_HEIGHT_M = 0.0036736;

    /**
     * The class constructor. Initialises all variables and allocates buffers for various images.
     *  @param imageSize the size to which all images will be resized.
     */
    CSI_Camera(const cv::Size& imageSize);

    /**
     * Class destructor. Stops the camera and closes the internal thread.
     */
    virtual ~CSI_Camera();

    /**
     * Starts CSI camera with provided configuration.
     *  @param framerate the camera's framerate in Hz.
     *  @param mode the mode of the camera - each camera may have different mode specification.
     *  @param id the id of the camera in case there are multiple cameras connected.
     *  @param flip the flip parameter. Usually 0 (no rotation) or 2 (180 deg).
     *  @return true if the both the camera and insternal thread were started correctly.
     */
    bool startCamera(const uint8_t framerate, const uint8_t mode, const uint8_t id, const uint8_t flip);

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

    /**
     * Provides the copy of RGB image as retrieved from the camera.
     *  @param[out] image the preallocated image buffer for CPU operations.
     *  @return the timestamp associated with the image or -1 if there was an error.
     */
    double getRawImage(cv::Mat& image) const;

    /**
     * Takes the copy of the latest raw RGB image and converts it to grey-scale.
     * For multi-camera system it is advised to acquire raw images first before processing them
     * to maximise chances of having them synchronised by software.
     *  @return the timestamp associated with the image or -1 if there was an error.
     */
    double acquireGreyScale();

    double acquireRectified();

    /**
     * Performs image rectification on internal greyscale buffer.
     *  @param[out] optional CPU buffer for rectified image.
     */
    void rectifyImage(cv::Mat* rectified = nullptr);

    /**
     * Sets the two rectification maps.
     *  @param xmap the first rectifiction map.
     *  @param ymap the second rectifiction map.
     */
    void setRMap(const cv::Mat& xmap, const cv::Mat& ymap);

    void initialiseVPIRemap(const cv::Mat& camMat, const cv::Mat& newCamMat, const cv::Mat& dist,
    		const cv::Mat& R = cv::Mat::eye(3, 3, CV_64F), const cv::Mat& T = cv::Mat::zeros(3, 1, CV_64F));

//    /**
//     *  @return the read-only access to the latest raw RGB image.
//     */
//    constexpr const cv::cuda::GpuMat& getImg() const
//    {
//        return mImg;
//    }
//
//    /**
//     *  @return the read-only access to the latest raw grey-scale image.
//     */
//    constexpr const cv::cuda::GpuMat& getGreyImg() const
//    {
//        return mGrey;
//    }
//
//    /**
//     *  @return the read-only access to the latest rectified grey-scale image.
//     */
//    constexpr const cv::cuda::GpuMat& getRectImg() const
//    {
//        return mRectified;
//    }

    void getRectified(cv::Mat& mat);
    void getFiltered(cv::Mat& mat);

    /**
     * This is somewhat hard-coded for a specific camera. Provides image size and framerate for given mode.
     *  @param mode the mode for which information need to be obtained.
     *  @param[out] size the size of image for given mode.
     *  @return the framerate for given mode in Hz.
     */
    static uint8_t getSizeForMode(const uint8_t mode, cv::Size& size);

    constexpr VPIStream getStream() const
    {
    	return mStream;
    }

    constexpr VPIImage getFiltered() const
    {
    	return mVPIFiltered;
    }

    constexpr VPIImage getFiltered16() const
    {
    	return mVPIFiltered16;
    }


protected:
    /**
     * The main body of the thread that constantly retrieves the latest image from CSI camera.
     */
    void mainThreadBody();

private:
    /**
     * Starts the new thread for pulling images from CSI camera.
     *  @param thread pointer to this class.
     *  @return nullptr.
     */
    static void* startThread(void* thread);

    /**
     * Gets the latest raw RGB image and coverts it to greyscale.
     *  @param[out] grey the preallocated image buffer for GPU operations.
     *  @return the timestamp associated with the image or -1 if there was an error.
     */
    double getGreyscale(cv::cuda::GpuMat& grey) const;

    /** ID of this camera. */
    uint8_t mID;
    cv::Size mImgSize;
    /** Flag that indicates if the thread should be running or not. */
    std::atomic<bool> mThreadRun;
    /** Thread which pulls images from CSI camera. */
    pthread_t mThread;
    /** Time of the latest frame in seconds. */
    double mFrameTime;
    /** OpenCV wrapper which allows communication with CSI camera. */
    cv::VideoCapture mCapture;
//    /** Preallocated buffer on GPU for raw RGB image. */
//    cv::cuda::GpuMat mImg;
    /** Preallocated buffer on GPU for raw greyscale image. */
//    cv::cuda::GpuMat mGrey;
    /** Preallocated buffer on GPU for rectified greyscale image. */
    cv::cuda::GpuMat mRectified;
    /** Preallocated buffers for image rectification maps. */
    cv::cuda::GpuMat mRMap[2];
    /** Mutex for synchronous access to data from the camera. */
    mutable pthread_mutex_t mMutex;

    VPIStream mStream;
    VPIPayload mWarp;
    VPIImage mVPIColour;
    VPIImage mVPIGrey;
    VPIImage mVPIRectified;
    VPIImage mVPIResized;
    VPIImage mVPIFiltered;
    VPIImage mVPIFiltered16;
};
#endif // __CSI_CAMERA_H__
