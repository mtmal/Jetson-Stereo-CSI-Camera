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

#include <unistd.h>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudafilters.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudastereo.hpp>
#include <opencv2/ximgproc/disparity_filter.hpp>
#include "CameraConstants.h"
#include "FrameTimeChecker.h"
#include "CSI_StereoCamera.h"

namespace
{
/**
 * Scales the camera matrix based on the ratio between the current (resized) size
 * and the maximum size at which the camera was calibrated.
 * @note this does not take into account different mode! If different mode is used
 * than for which calibration was done, image centre needs to be shifted first to
 * then all parameters should be scaled as currently done.
 * TODO: fix what is explained in the note above.
 *  @param imgSize the size of images which will be acquired from the cameras.
 *  @param maxSize the max size of images for which the calibration was done.
 *  @param[in/out] camMat camera matrix which parameters are being scaled.
 */
void scaleCameraMatrix(const cv::Size& imgSize, const cv::Size& maxSize, cv::Mat& camMat)
{
    camMat.at<double>(0, 0) *= static_cast<double>(imgSize.width)  / static_cast<double>(maxSize.width);
    camMat.at<double>(0, 2) *= static_cast<double>(imgSize.width)  / static_cast<double>(maxSize.width);
    camMat.at<double>(1, 1) *= static_cast<double>(imgSize.height) / static_cast<double>(maxSize.height);
    camMat.at<double>(1, 2) *= static_cast<double>(imgSize.height) / static_cast<double>(maxSize.height);

    /* In case the new camera matrix is passed, additional parameters need to be resized. */
    if (camMat.cols == 4)
    {
        camMat.at<double>(0, 3) *= static_cast<double>(imgSize.width)   / static_cast<double>(maxSize.width);
        camMat.at<double>(1, 3) *= static_cast<double>(imgSize.height)  / static_cast<double>(maxSize.height);
    }
}
} /* end of the anonymous namespace */

CSI_StereoCamera::CSI_StereoCamera(const cv::Size& imageSize)
: GenericListener<CameraData>(), 
  ICameraTalker(),
  GenericThread<CSI_StereoCamera>(),
  mImageSize(imageSize),
  mRequestedRect(false), 
  mLCam(), 
  mRCam(), 
  mDisparity(imageSize, CV_8UC1, cv::cuda::HostMem::AllocType::SHARED), 
  mLeftGPU(imageSize, CV_8UC1, cv::cuda::HostMem::AllocType::SHARED), 
  mRightGPU(imageSize, CV_8UC1),
  mFlippedLeft(imageSize, CV_8UC1), mFlippedRight(imageSize, CV_8UC1),
  mDisparityRLCPU(imageSize, CV_8UC1, cv::cuda::HostMem::AllocType::SHARED), 
  mMedianFilter(cv::cuda::createMedianFilter(CV_8UC1, 9)),
  mStereoBM(cv::cuda::createStereoBM(160))
{
	mStereoBM->setPreFilterType(1);
	restartDispFilter(8000.0, 2.0);
    mRectMaps[0][0] = cv::cuda::HostMem(mImageSize, CV_32FC1, cv::cuda::HostMem::AllocType::SHARED);
    mRectMaps[0][1] = cv::cuda::HostMem(mImageSize, CV_32FC1, cv::cuda::HostMem::AllocType::SHARED);
    mRectMaps[1][0] = cv::cuda::HostMem(mImageSize, CV_32FC1, cv::cuda::HostMem::AllocType::SHARED);
    mRectMaps[1][1] = cv::cuda::HostMem(mImageSize, CV_32FC1, cv::cuda::HostMem::AllocType::SHARED);
}

CSI_StereoCamera::~CSI_StereoCamera()
{
    stopCamera();
}

bool CSI_StereoCamera::startCamera(const cv::Size& imageSize, const uint8_t framerate, const uint8_t mode, 
                                   const std::vector<uint8_t>& ids, const uint8_t flip, 
                                   const bool colour, const bool rectified)
{
    bool retVal = (ids.size() >= 2 && imageSize == mImageSize);
    if (retVal)
    {
        if (isRunning())
        {
            stopCamera();
        }

        retVal = mLCam.startCamera(mImageSize, framerate, mode, {ids[0]}, flip, colour, rectified)
               & mRCam.startCamera(mImageSize, framerate, mode, {ids[1]}, flip, colour, rectified);

        if (retVal)
        {
            mRequestedRect = rectified;
            retVal = startThread();

            if (!retVal)
            {
                /* one of the cameras have failed to start, so we stop all */
                stopCamera();
            }
            else
            {
                mCamDatas.mID = ids;
                mCamDatas.mTimestamp = {0.0, 0.0};
                mCamDatas.mImage = {cv::cuda::HostMem(mImageSize, colour ? CV_8UC3 : CV_8UC1, cv::cuda::HostMem::AllocType::SHARED),
                                    cv::cuda::HostMem(mImageSize, colour ? CV_8UC3 : CV_8UC1, cv::cuda::HostMem::AllocType::SHARED)};

                mLCam.registerTo(static_cast<GenericListener<CameraData>*>(this));
                mRCam.registerTo(static_cast<GenericListener<CameraData>*>(this));
            }
        }
    }
	return retVal;
}

void CSI_StereoCamera::stopCamera()
{
    stopThread();
    mLCam.stopCamera();
    mRCam.stopCamera();
    mLCam.unregisterFrom(static_cast<GenericListener<CameraData>*>(this));
    mRCam.unregisterFrom(static_cast<GenericListener<CameraData>*>(this));
}

bool CSI_StereoCamera::isInitialised() const
{
    return mLCam.isInitialised() && mRCam.isInitialised();
}

void CSI_StereoCamera::restartDispFilter(const double lambda, const double sigmaColour)
{
	mDispWLSFilter = cv::ximgproc::createDisparityWLSFilter(mStereoBM);
	mDispWLSFilter->setLambda(lambda);
	mDispWLSFilter->setSigmaColor(sigmaColour);
}

bool CSI_StereoCamera::loadCalibration(const std::string& folder)
{
    cv::Size maxSize;
    cv::Mat lCamMat, rCamMat, lDist, rDist;
    cv::Mat R, T, R1, R2, P1, P2, Q;
    cv::Mat rectMap[2];
    bool retVal = true;
    cv::FileStorage fs;

    static_cast<void>(CSI_Camera::getSizeForMode(0, maxSize));
    
    fs.open(folder + "/" + LEFT_CALIB_FILE + CALIB_FILE_EXTENSION, cv::FileStorage::READ);
    retVal &= fs.isOpened();
    if (retVal)
    {
        fs[CAMERA_MATRIX] >> lCamMat;
        fs[DISTORTION] >> lDist;
        fs.release();
        scaleCameraMatrix(mImageSize, maxSize, lCamMat);

        fs.open(folder + "/" + RIGHT_CALIB_FILE + CALIB_FILE_EXTENSION, cv::FileStorage::READ);
        retVal &= fs.isOpened();
        if (retVal)
        {
            fs[CAMERA_MATRIX] >> rCamMat;
            fs[DISTORTION] >> rDist;
            fs.release();
            scaleCameraMatrix(mImageSize, maxSize, rCamMat);

            fs.open(folder + "/" + STEREO_CALIB_FILE + CALIB_FILE_EXTENSION, cv::FileStorage::READ);
            retVal &= fs.isOpened();
            if (retVal)
            {
                fs[ROTATION]  			>> R;
                fs[TRANSLATION]  		>> T;
                fs[RECTIFICATION_LEFT]  >> R1;
                fs[RECTIFICATION_RIGHT] >> R2;
                fs[NEW_CAM_MATRIX_LEFT] >> P1;
                fs[NEW_CAM_MATRIX_RIGHT]>> P2;
                fs[DISPARITY_TO_DEPTH]  >> Q;
                fs.release();
                scaleCameraMatrix(mImageSize, maxSize, P1);
                scaleCameraMatrix(mImageSize, maxSize, P2);

                initUndistortRectifyMap(lCamMat, lDist, R1, P1, mImageSize, CV_32FC1, mRectMaps[0][0], mRectMaps[0][1]);
                initUndistortRectifyMap(rCamMat, rDist, R2, P2, mImageSize, CV_32FC1, mRectMaps[1][0], mRectMaps[1][1]);
            }
        }
    }
    return retVal;
}

void CSI_StereoCamera::computeDisp(const bool filter, const cv::cuda::HostMem& lImg, const cv::cuda::HostMem& rImg, cv::Mat& disparity)
{
	int minDisp;
	int specklewindowsize;
	int disp12diff;

    if (mLCam.getColour())
    {
        cv::cuda::GpuMat lMono(mImageSize, CV_8UC1);
        cv::cuda::GpuMat rMono(mImageSize, CV_8UC1);
        cv::cuda::cvtColor(lImg, lMono, cv::COLOR_BGR2GRAY, 1);
        cv::cuda::cvtColor(rImg, rMono, cv::COLOR_BGR2GRAY, 1);
        mMedianFilter->apply(lMono, mLeftGPU);
        mMedianFilter->apply(rMono, mRightGPU);
    }
    else
    {
        mMedianFilter->apply(lImg, mLeftGPU);
        mMedianFilter->apply(rImg, mRightGPU);
    }

#ifdef LOG
	int64 time1 = cv::getTickCount();
#endif /* LOG */
	mStereoBM->compute(mLeftGPU, mRightGPU, mDisparity);
#ifdef LOG
	int64 time2 = cv::getTickCount();
#endif /* LOG */

    if (filter)
    {
    	/* Store parameters specific for left-right disparity. */
    	minDisp = mStereoBM->getMinDisparity();
    	specklewindowsize = mStereoBM->getSpeckleWindowSize();
    	disp12diff = mStereoBM->getDisp12MaxDiff();

    	// mDisparity.download(mDisparityCPU);

    	/* We need to change parameters for calculating right-left disparity. */
    	mStereoBM->setMinDisparity(-(mStereoBM->getMinDisparity() + mStereoBM->getNumDisparities()) + 1);
    	mStereoBM->setDisp12MaxDiff(1000000);
		mStereoBM->setSpeckleWindowSize(0);

        cv::cuda::flip(mLeftGPU, mFlippedLeft, 1);
        cv::cuda::flip(mRightGPU, mFlippedRight, 1);
		mStereoBM->compute(mFlippedRight, mFlippedLeft, mDisparityRLCPU);
#ifdef LOG
    	int64 time3 = cv::getTickCount();
#endif /* LOG */

    	/* We rever parameters back to calculating left-right disparity. */
    	mStereoBM->setMinDisparity(minDisp);
    	mStereoBM->setDisp12MaxDiff(disp12diff);
		mStereoBM->setSpeckleWindowSize(specklewindowsize);

    	mDispWLSFilter->filter(mDisparity, mLeftGPU, disparity, mDisparityRLCPU);
#ifdef LOG
    	int64 time4 = cv::getTickCount();
    	printf("LR Disparity calculated in %f \t RL Disparity calculated in %f \t WLS filter calculated in %f \n",
    			static_cast<double>(time2 - time1) / cv::getTickFrequency(),
    			static_cast<double>(time3 - time2) / cv::getTickFrequency(),
    			static_cast<double>(time4 - time3) / cv::getTickFrequency());
#endif /* LOG */
    }
    else
    {
        mDisparity.createMatHeader().copyTo(disparity);
#ifdef LOG
    	printf("Disparity calculated in %f \n", static_cast<double>(time2 - time1) / cv::getTickFrequency());
#endif /* LOG */
    }
}

void CSI_StereoCamera::update(const CameraData& camData)
{
    static std::atomic<int> counter{0};
    ScopedLock lock(mMutex);

    mCamDatas.mTimestamp[camData.mID[0]] = camData.mTimestamp[0];
    mCamDatas.mImage[camData.mID[0]] = camData.mImage[0];
    if (++counter % 2 == 0)
    {
        sem_post(&mSemaphore);
    }
}

void* CSI_StereoCamera::threadBody()
{
    /** Utility class used for comparing stereo pair frame times to access if they are synchronised or not. */
    FrameTimeChecker ftc;

    CameraData camData;
    camData.mID = {mLCam.getId(), mRCam.getId()};
    camData.mTimestamp = {0.0, 0.0};
    camData.mImage = {cv::cuda::HostMem(mImageSize, mLCam.getColour() ? CV_8UC3 : CV_8UC1, cv::cuda::HostMem::AllocType::SHARED),
                      cv::cuda::HostMem(mImageSize, mRCam.getColour() ? CV_8UC3 : CV_8UC1, cv::cuda::HostMem::AllocType::SHARED)};

    while (isRunning())
    {
        if (0 == sem_wait(&mSemaphore))
        {
            pthread_mutex_lock(&mMutex);
            if (mCamDatas.mTimestamp[0] > 0 && 
                mCamDatas.mTimestamp[1] > 0 && 
                ftc.checkTimes(mCamDatas.mTimestamp[0], mCamDatas.mTimestamp[1]))
            {
                if (!mRequestedRect)
                {
                    mCamDatas.mImage[0].swap(camData.mImage[0]);
                    mCamDatas.mImage[1].swap(camData.mImage[1]);
                }
                else
                {
                    cv::cuda::remap(mCamDatas.mImage[0], camData.mImage[0], mRectMaps[0][0], mRectMaps[0][1], cv::INTER_LINEAR);
                    cv::cuda::remap(mCamDatas.mImage[1], camData.mImage[1], mRectMaps[1][0], mRectMaps[1][1], cv::INTER_LINEAR);
                }
                pthread_mutex_unlock(&mMutex);
                this->notifyListeners(camData);
            }
            else
            {
                pthread_mutex_unlock(&mMutex);
                puts("Image timestamps too far away, discarding images.");
            }
        }
    }
    return nullptr;
}
