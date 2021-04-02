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

#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>
#include "CSI_Camera.h"

namespace
{
/**
 * Populates the string with nvarguscamerasrc command for GStreamer.
 *  @param id the id of the camera in case there are multiple cameras connected.
 *  @param mode the mode of the camera - each camera may have different mode specification.
 *  @param imageSize the size to which images should be resized.
 *  @param framerate the camera's framerate in Hz.
 *  @param flip the flip parameter. Usually 0 (no rotation) or 2 (180 deg).
 *	@return the string with the command for GStreamer.
 */
std::string gstreamer_pipeline(const uint8_t id, const uint8_t mode, const cv::Size& imageSize,
                               const uint8_t framerate, const uint8_t flip)
{
    char text[512];
    sprintf(text, "nvarguscamerasrc sensor-id=%u sensor-mode=%u ! video/x-raw(memory:NVMM), format=(string)NV12, framerate=(fraction)%u/1 ! "
                  "nvvidconv flip-method=%d ! video/x-raw, width=%d, height=%d, format=(string)BGRx ! videoconvert ! video/x-raw, "
                  "format=(string)BGR ! appsink", id, mode, framerate, flip, imageSize.width, imageSize.height);
#ifdef LOG
    printf("%s\n", text);
#endif /* LOG */
    return text;
}
} /* end of the anonymous namespace */

CSI_Camera::CSI_Camera(const cv::Size& imageSize)
: mID(0), mThreadRun(false), mThread(0), mFrameTime(-1.0), mCapture(),
  mImg(imageSize, CV_8UC3, cv::cuda::HostMem::SHARED), mGrey(imageSize, CV_8UC1),
  mRectified(imageSize, CV_8UC1), mFiltered(imageSize, CV_8UC1, cv::cuda::HostMem::SHARED)
{
    pthread_mutex_init(&mMutex, nullptr);
}

CSI_Camera::~CSI_Camera()
{
	stopCamera();
    pthread_mutex_destroy(&mMutex);
}

bool CSI_Camera::startCamera(const uint8_t framerate, const uint8_t mode, const uint8_t id, const uint8_t flip)
{
	mID = id;
	mThreadRun = true;
	mCapture.open(gstreamer_pipeline(mID, mode, mImg.size(), framerate, flip), cv::CAP_GSTREAMER);
    return (mCapture.isOpened() && (0 == pthread_create(&mThread, nullptr, CSI_Camera::startThread, this)));
}

void CSI_Camera::stopCamera()
{
    mThreadRun = false;
    if (mThread > 0)
    {
    	pthread_join(mThread, nullptr);
    	mThread = 0;
    }
    if (mCapture.isOpened())
    {
    	mCapture.release();
    }
}

double CSI_Camera::getRawImage(cv::Mat& image) const
{
    double time = -1.0;
    int errorCode = pthread_mutex_lock(&mMutex);
    if (0 == errorCode)
    {
    	mImg.createMatHeader().copyTo(image);
        time = mFrameTime;
        pthread_mutex_unlock(&mMutex);
    }
#ifdef LOG
    else
    {
        printf("getRawImage :: ID: %d -- Mutex Lock error %d \n", mID, errorCode);
    }
#endif /* LOG */
    return time;
}

void CSI_Camera::rectifyImage()
{
	cv::cuda::remap(mGrey, mRectified, mRMap[0], mRMap[1], cv::INTER_LINEAR);
	/* it seems like a massive value, but images are small and super-noisy, so we kind of need it. */
	cv::cuda::bilateralFilter(mRectified, mFiltered, 11, 50, 50);
}

void CSI_Camera::setRMap(const cv::Mat& xmap, const cv::Mat& ymap)
{
    mRMap[0].upload(xmap);
    mRMap[1].upload(ymap);
}

double CSI_Camera::getGreyscale(cv::cuda::GpuMat& grey) const
{
    double time = -1.0;
    int errorCode = pthread_mutex_lock(&mMutex);
    if (0 == errorCode)
    {
        cv::cuda::cvtColor(mImg, grey, cv::COLOR_BGR2GRAY);
        time = mFrameTime;
        pthread_mutex_unlock(&mMutex);
    }
#ifdef LOG
    else
    {
        printf("getGreyscale :: ID: %d -- Mutex Lock error %d \n", mID, errorCode);
    }
#endif /* LOG */
    return time;
}

uint8_t CSI_Camera::getSizeForMode(const uint8_t mode, cv::Size& size)
{
	uint8_t framerate;
    switch (mode)
    {
        case 0:
            framerate = 21;
            size = cv::Size(3264, 2464);
            break;
        case 1:
            framerate = 28;
            size = cv::Size(3264, 1848);
            break;
        case 2:
            framerate = 30;
            size = cv::Size(1920, 1080);
            break;
        case 3:
            framerate = 30;
            size = cv::Size(1640, 1232);
            break;
        case 4:
            framerate = 60;
            size = cv::Size(1280,  720);
            break;
        case 5:
            framerate = 120;
            size = cv::Size(1280,  720);
            break;
        default:
        	framerate = 0;
            size = cv::Size(0, 0);
            break;
    }
    return framerate;
}

void CSI_Camera::mainThreadBody()
{
    int errorCode;
    timespec time;
    time.tv_sec = 1;
    time.tv_nsec = 0;
    while (mThreadRun)
    {
        mCapture.grab();
        errorCode = pthread_mutex_timedlock(&mMutex, &time);
        if (0 == errorCode)
        {
            mCapture.retrieve(mImg);
            mFrameTime = mCapture.get(cv::CAP_PROP_POS_MSEC);
            pthread_mutex_unlock(&mMutex);
        }
#ifdef LOG
        else
        {
            printf("ID: %d -- Mutex Lock error %d \n", mID, errorCode);
        }
#endif /* LOG */
    }
}

void* CSI_Camera::startThread(void* thread)
{
    CSI_Camera* camera = static_cast<CSI_Camera*>(thread);
    camera->mainThreadBody();
    return nullptr;
}
