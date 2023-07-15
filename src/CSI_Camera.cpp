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
 *  @param imageType the type of images to convert to. For example, BGR or GRAY8.
 *	@return the string with the command for GStreamer.
 */
std::string gstreamerPipeline(const uint8_t id, const uint8_t mode, const cv::Size& imageSize,
                               const uint8_t framerate, const uint8_t flip, const std::string& imageType = "BGR")
{
    char text[512];
    memset(text, '\0', 512);
    sprintf(text, "nvarguscamerasrc sensor-id=%u sensor-mode=%u ! video/x-raw(memory:NVMM), format=(string)NV12, framerate=(fraction)%u/1 ! "
                  "nvvidconv flip-method=%d ! video/x-raw, width=%d, height=%d, format=(string)BGRx ! videoconvert ! video/x-raw, "
                  "format=(string)%s ! appsink", id, mode, framerate, flip, imageSize.width, imageSize.height, imageType.c_str());
#ifdef LOG
    printf("%s\n", text);
#endif /* LOG */
    return text;
}
} /* end of the anonymous namespace */

CSI_Camera::CSI_Camera()
: GenericTalker<const uint8_t, const double, const cv::cuda::GpuMat&>(), mID(0), mImgSize(), mColour(true), mThreadRun(false), mThread(0), mCapture()
{
}

CSI_Camera::~CSI_Camera()
{
	stopCamera();
}

bool CSI_Camera::startCamera(const cv::Size& imageSize, const uint8_t framerate, const uint8_t mode, 
                             const uint8_t id, const uint8_t flip, const bool colour)
{
    if (isInitialised())
    {
        stopCamera();
    }
    mImgSize = imageSize;
    mColour = colour;
	mID = id;
	mThreadRun = true;
	mCapture.open(gstreamerPipeline(mID, mode, imageSize, framerate, flip, colour ? "BGR" : "GRAY8"), cv::CAP_GSTREAMER);
    return (isInitialised() && (0 == pthread_create(&mThread, nullptr, CSI_Camera::startGrabThread, this)));
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

void CSI_Camera::grabThreadBody()
{
    cv::cuda::GpuMat image(getSize(), getColour() ? CV_8UC3 : CV_8UC1);
    while (isRun())
    {
        if (mCapture.read(image))
        {
            this->notifyListeners(getId(), mCapture.get(cv::CAP_PROP_POS_MSEC), image);
        }
    }
}

void* CSI_Camera::startGrabThread(void* thread)
{
    CSI_Camera* camera = static_cast<CSI_Camera*>(thread);
    camera->grabThreadBody();
    return nullptr;
}
