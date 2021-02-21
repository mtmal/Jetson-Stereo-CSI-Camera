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
#include <vpi/algo/BilateralFilter.h>
#include <vpi/algo/ConvertImageFormat.h>
#include <vpi/algo/Remap.h>
#include <vpi/algo/Rescale.h>
#include <vpi/Image.h>
#include <vpi/LensDistortionModels.h>
#include <vpi/OpenCVInterop.hpp>
#include "CSI_Camera.h"

#define LOG

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
: mID(0), mImgSize(imageSize), mThreadRun(false), mThread(0), mFrameTime(-1.0), mCapture(),
  mRectified(imageSize, CV_8UC1)
{
    pthread_mutex_init(&mMutex, nullptr);
    vpiStreamCreate(0, &mStream);
    mWarp = nullptr;
    mVPIColour = nullptr;
    vpiImageCreate(imageSize.width, imageSize.height, VPI_IMAGE_FORMAT_U8, 0, &mVPIGrey);
    vpiImageCreate(imageSize.width, imageSize.height, VPI_IMAGE_FORMAT_U8, 0, &mVPIRectified);
//    vpiImageCreate(imageSize.width, imageSize.height, VPI_IMAGE_FORMAT_U8, 0, &mVPIFiltered);
    vpiImageCreate(480, 270, VPI_IMAGE_FORMAT_U8, 0, &mVPIResized);
    vpiImageCreate(480, 270, VPI_IMAGE_FORMAT_U8, 0, &mVPIFiltered);
    vpiImageCreate(480, 270, VPI_IMAGE_FORMAT_U16, 0, &mVPIFiltered16);
}

CSI_Camera::~CSI_Camera()
{
	stopCamera();
    pthread_mutex_destroy(&mMutex);
    vpiStreamDestroy(mStream);
    if (mWarp)
    {
    	vpiPayloadDestroy(mWarp);
    }
    vpiImageDestroy(mVPIGrey);
    vpiImageDestroy(mVPIRectified);
    vpiImageDestroy(mVPIResized);
    vpiImageDestroy(mVPIFiltered);
    vpiImageDestroy(mVPIFiltered16);
}

bool CSI_Camera::startCamera(const uint8_t framerate, const uint8_t mode, const uint8_t id, const uint8_t flip)
{
	mID = id;
	mThreadRun = true;
	mCapture.open(gstreamer_pipeline(mID, mode, mImgSize, framerate, flip), cv::CAP_GSTREAMER);
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
	VPIImageData data;
    double time = -1.0;
    int errorCode = pthread_mutex_lock(&mMutex);
    if (0 == errorCode)
    {
    	vpiImageLock(mVPIColour, VPI_LOCK_READ, &data);
    	vpiImageDataExportOpenCVMat(data, &image);
    	vpiImageUnlock(mVPIColour);
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

double CSI_Camera::acquireGreyScale()
{
    double time = -1.0;
//    int errorCode = pthread_mutex_lock(&mMutex);
    int errorCode = (int)vpiImageLock(mVPIColour, VPI_LOCK_READ, nullptr);
    if (0 == errorCode)
    {
    	vpiSubmitConvertImageFormat(mStream, VPI_BACKEND_CUDA, mVPIColour, mVPIGrey, nullptr);
    	vpiStreamSync(mStream);
        time = mFrameTime;
        vpiImageUnlock(mVPIColour);
//        pthread_mutex_unlock(&mMutex);
    }
#ifdef LOG
    else
    {
        printf("acquireGreyScale :: ID: %d -- Mutex Lock error %d \n", mID, errorCode);
    }
#endif /* LOG */
    return time;
}

double CSI_Camera::acquireRectified()
{
    double time = -1.0;
    int errorCode = pthread_mutex_lock(&mMutex);
    if (0 == errorCode)
    {
        time = mFrameTime;
        vpiSubmitConvertImageFormat(mStream, VPI_BACKEND_CUDA, mVPIColour, mVPIGrey, nullptr);
        vpiSubmitRemap(mStream, VPI_BACKEND_CUDA, mWarp, mVPIGrey, mVPIRectified, VPI_INTERP_CATMULL_ROM, VPI_BORDER_ZERO, 0);
        vpiSubmitRescale(mStream, VPI_BACKEND_CUDA, mVPIRectified, mVPIResized, VPI_INTERP_LINEAR, VPI_BORDER_CLAMP, 0);
        vpiSubmitBilateralFilter(mStream, VPI_BACKEND_CUDA, mVPIResized, mVPIFiltered, 9, 50, 1.7, VPI_BORDER_ZERO);
        vpiSubmitConvertImageFormat(mStream, VPI_BACKEND_CUDA, mVPIFiltered, mVPIFiltered16, nullptr);
        pthread_mutex_unlock(&mMutex);
    }
#ifdef LOG
    else
    {
        printf("acquireRectified :: ID: %d -- Mutex Lock error %d \n", mID, errorCode);
    }
#endif /* LOG */
    return time;
}

void CSI_Camera::rectifyImage(cv::Mat* rectified)
{
//	cv::Mat temp;
	cv::Mat temp2;
 	VPIImageData data;
	puts("REMAP 1");
//	cv::cuda::remap(mGrey, mRectified, mRMap[0], mRMap[1], cv::INTER_LINEAR);

//	mGrey.download(temp);
	mRectified.download(temp2);
	puts("REMAP 2");
//	vpiImageCreateOpenCVMatWrapper(temp, 0, &mVPIGrey);

	puts("REMAP 3");
	vpiSubmitRemap(mStream, VPI_BACKEND_CUDA, mWarp, mVPIGrey, mVPIRectified, VPI_INTERP_CATMULL_ROM, VPI_BORDER_ZERO, 0);
	puts("REMAP 4");
	vpiStreamSync(mStream);
	puts("REMAP 5");

	vpiImageLock(mVPIRectified, VPI_LOCK_READ, &data);

	puts("REMAP 6");
	if (rectified)
	{
		vpiImageDataExportOpenCVMat(data, rectified);
		mRectified.upload(*rectified);
	}
	else
	{
		vpiImageDataExportOpenCVMat(data, &temp2);
		mRectified.upload(temp2);
	}
	vpiImageUnlock(mVPIRectified);

//	vpiImageDestroy(mVPIGrey);
}

void CSI_Camera::setRMap(const cv::Mat& xmap, const cv::Mat& ymap)
{
    mRMap[0].upload(xmap);
    mRMap[1].upload(ymap);
}

void CSI_Camera::initialiseVPIRemap(const cv::Mat& camMat, const cv::Mat& newCamMat, const cv::Mat& dist,
		const cv::Mat& R, const cv::Mat& T)
{
    VPIWarpMap mVPIWarpMap;
    const VPICameraIntrinsic K =
    {
        { static_cast<float>(camMat.at<double>(0, 0)), 0, static_cast<float>(camMat.at<double>(0, 2)) },
        { 0, static_cast<float>(camMat.at<double>(1, 1)), static_cast<float>(camMat.at<double>(1, 2)) }
    };
    const VPICameraIntrinsic K_new =
    {
        { static_cast<float>(newCamMat.at<double>(0, 0)), 0, static_cast<float>(newCamMat.at<double>(0, 2)) },
        { 0, static_cast<float>(newCamMat.at<double>(1, 1)), static_cast<float>(newCamMat.at<double>(1, 2)) }
    };
    const VPICameraExtrinsic X =
    {
        { static_cast<float>(R.at<double>(0, 0)), static_cast<float>(R.at<double>(0, 1)),
          static_cast<float>(R.at<double>(0, 2)), static_cast<float>(T.at<double>(0)) },
        { static_cast<float>(R.at<double>(1, 0)), static_cast<float>(R.at<double>(1, 1)),
		  static_cast<float>(R.at<double>(1, 2)), static_cast<float>(T.at<double>(1)) },
        { static_cast<float>(R.at<double>(2, 0)), static_cast<float>(R.at<double>(2, 1)),
          static_cast<float>(R.at<double>(2, 2)), static_cast<float>(T.at<double>(2)) }
    };
    const VPIPolynomialLensDistortionModel lens =
    {
		static_cast<float>(dist.at<double>(0)),
		static_cast<float>(dist.at<double>(1)),
		static_cast<float>(dist.at<double>(4)),
		static_cast<float>(dist.at<double>(5)),
		static_cast<float>(dist.at<double>(6)),
		static_cast<float>(dist.at<double>(7)),
		static_cast<float>(dist.at<double>(2)),
		static_cast<float>(dist.at<double>(3))
    };
    memset(&mVPIWarpMap, 0, sizeof(mVPIWarpMap));
    mVPIWarpMap.grid.numHorizRegions  = 1;
    mVPIWarpMap.grid.numVertRegions   = 1;
    mVPIWarpMap.grid.regionWidth[0]   = static_cast<int16_t>(mImgSize.width);
    mVPIWarpMap.grid.regionHeight[0]  = static_cast<int16_t>(mImgSize.height);
    mVPIWarpMap.grid.horizInterval[0] = 1;
    mVPIWarpMap.grid.vertInterval[0]  = 1;
    vpiWarpMapAllocData(&mVPIWarpMap);
    vpiWarpMapGenerateFromPolynomialLensDistortionModel(K, X, K_new, &lens, &mVPIWarpMap);
    vpiCreateRemap(VPI_BACKEND_CUDA, &mVPIWarpMap, &mWarp);
    vpiWarpMapFreeData(&mVPIWarpMap);
}

void CSI_Camera::getRectified(cv::Mat& mat)
{
	VPIImageData data;
	vpiImageLock(mVPIRectified, VPI_LOCK_READ, &data);
	vpiImageDataExportOpenCVMat(data, &mat);
	vpiImageUnlock(mVPIRectified);
}

void CSI_Camera::getFiltered(cv::Mat& mat)
{
	VPIImageData data;
	vpiImageLock(mVPIFiltered, VPI_LOCK_READ, &data);
	vpiImageDataExportOpenCVMat(data, &mat);
	vpiImageUnlock(mVPIFiltered);
}

double CSI_Camera::getGreyscale(cv::cuda::GpuMat& grey) const
{
    double time = -1.0;
//    int errorCode = pthread_mutex_lock(&mMutex);
//    if (0 == errorCode)
//    {
//        cv::cuda::cvtColor(mImg, grey, cv::COLOR_BGR2GRAY);
//        time = mFrameTime;
//        pthread_mutex_unlock(&mMutex);
//    }
//#ifdef LOG
//    else
//    {
//        printf("getGreyscale :: ID: %d -- Mutex Lock error %d \n", mID, errorCode);
//    }
//#endif /* LOG */
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
    cv::Mat mImg(mImgSize, CV_8UC3);
    timespec time;
    time.tv_sec = 1;
    time.tv_nsec = 0;
    vpiImageCreateOpenCVMatWrapper(mImg, 0, &mVPIColour);
    while (mThreadRun)
    {
        mCapture.grab();
        mCapture.retrieve(mImg);
        errorCode = pthread_mutex_timedlock(&mMutex, &time);
        if (0 == errorCode)
        {
            mFrameTime = mCapture.get(cv::CAP_PROP_POS_MSEC);
            vpiImageSetWrappedOpenCVMat(mVPIColour, mImg);
            pthread_mutex_unlock(&mMutex);
        }
#ifdef LOG
        else
        {
            printf("ID: %d -- Mutex Lock error %d \n", mID, errorCode);
        }
#endif /* LOG */
    }
    vpiImageDestroy(mVPIColour);
}

void* CSI_Camera::startThread(void* thread)
{
    CSI_Camera* camera = static_cast<CSI_Camera*>(thread);
    camera->mainThreadBody();
    return nullptr;
}
