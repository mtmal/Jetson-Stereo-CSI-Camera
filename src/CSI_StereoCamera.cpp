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

#include <opencv2/cudastereo.hpp>
#include <opencv2/ximgproc/disparity_filter.hpp>
#include <opencv2/cudafilters.hpp>
#include <vpi/algo/ConvertImageFormat.h>
#include <vpi/algo/StereoDisparity.h>
#include <vpi/Event.h>
#include <vpi/Image.h>
#include <vpi/OpenCVInterop.hpp>
#include "CameraConstants.h"
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
: mLCam(imageSize), mRCam(imageSize), mDisparity(imageSize, CV_8UC1), mLeftGPU(imageSize, CV_8UC1),
  mRightGPU(imageSize, CV_8UC1), mLeftCPU(imageSize, CV_8UC1), mDisparityCPU(imageSize, CV_16UC1),
  mDisparityRLCPU(imageSize, CV_16UC1), mMedianFilter(cv::cuda::createMedianFilter(CV_8UC1, 9)),
  mStereoBM(cv::cuda::createStereoBM(160))
{
	const VPIStereoDisparityEstimatorCreationParams stereoParams = {32};
	mStereoBM->setPreFilterType(1);
	restartDispFilter(8000.0, 2.0);
	vpiEventCreate(0, &mSyncEvent);

	vpiCreateStereoDisparityEstimator(VPI_BACKEND_CUDA, imageSize.width, imageSize.height, VPI_IMAGE_FORMAT_U16, &stereoParams, &mVPIStereo);

	vpiImageCreate(480, 270, VPI_IMAGE_FORMAT_U16, 0, &mVPILeftRightDisp);
	vpiImageCreate(480, 270, VPI_IMAGE_FORMAT_U16, 0, &mVPIRightLeftDisp);
	vpiImageCreate(480, 270, VPI_IMAGE_FORMAT_U8,  0, &mVPIOutDisp);

	filteredDisp = cv::Mat(imageSize, CV_16SC1);
}

CSI_StereoCamera::~CSI_StereoCamera()
{
	vpiEventDestroy(mSyncEvent);
	vpiPayloadDestroy(mVPIStereo);
	vpiImageDestroy(mVPILeftRightDisp);
	vpiImageDestroy(mVPIRightLeftDisp);
	vpiImageDestroy(mVPIOutDisp);
}

bool CSI_StereoCamera::startCamera(const uint8_t framerate, const uint8_t mode, const uint8_t lCamID,
		const uint8_t rCamID, const uint8_t flip)
{
	bool lFlag = mLCam.startCamera(framerate, mode, lCamID, flip);
	bool rFlag = mRCam.startCamera(framerate, mode, rCamID, flip);
	if (!lFlag || !rFlag)
	{
		/* one of the cameras have failed to start, so we stop both */
		mLCam.stopCamera();
		mRCam.stopCamera();
	}
	return (lFlag && rFlag);
}

bool CSI_StereoCamera::getRectified(const bool forceProcessing, cv::Mat& left, cv::Mat& right)
{
	VPIImageData data;
//	bool retVal = mFTC.checkTimes(mLCam.acquireGreyScale(), mRCam.acquireGreyScale());
//	if (retVal || forceProcessing)
//	{
//		/* perform processing only if images are OK or we specifically required it. */
//		mLCam.rectifyImage(&left);
//		mRCam.rectifyImage(&right);
//	}
//    return retVal;
	bool retVal = mFTC.checkTimes(mRCam.acquireRectified(), mLCam.acquireRectified());
	if (retVal || forceProcessing)
	{
		vpiEventRecord(mSyncEvent, mRCam.getStream());
		vpiStreamWaitEvent(mLCam.getStream(), mSyncEvent);

    	vpiImageLock(mLCam.getFiltered(), VPI_LOCK_READ, &data);
    	vpiImageDataExportOpenCVMat(data, &left);
        vpiImageUnlock(mLCam.getFiltered());
    	vpiImageLock(mRCam.getFiltered(), VPI_LOCK_READ, &data);
    	vpiImageDataExportOpenCVMat(data, &right);
        vpiImageUnlock(mRCam.getFiltered());
	}
	return retVal;
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
        scaleCameraMatrix(mLeftGPU.size(), maxSize, lCamMat);

        fs.open(folder + "/" + RIGHT_CALIB_FILE + CALIB_FILE_EXTENSION, cv::FileStorage::READ);
        retVal &= fs.isOpened();
        if (retVal)
        {
            fs[CAMERA_MATRIX] >> rCamMat;
            fs[DISTORTION] >> rDist;
            fs.release();
            scaleCameraMatrix(mLeftGPU.size(), maxSize, rCamMat);

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
                scaleCameraMatrix(mLeftGPU.size(), maxSize, P1);
                scaleCameraMatrix(mLeftGPU.size(), maxSize, P2);

//                initUndistortRectifyMap(lCamMat, lDist, R1, P1, mLeftGPU.size(), CV_32FC1, rectMap[0], rectMap[1]);
//                mLCam.setRMap(rectMap[0], rectMap[1]);
//                initUndistortRectifyMap(rCamMat, rDist, R2, P2, mLeftGPU.size(), CV_32FC1, rectMap[0], rectMap[1]);
//                mRCam.setRMap(rectMap[0], rectMap[1]);

                mLCam.initialiseVPIRemap(lCamMat, P1, lDist, R1);
                mRCam.initialiseVPIRemap(rCamMat, P2, rDist, R2);
//                mRCam.initialiseVPIRemap(rCamMat, P2, rDist, R, T);
            }
        }
    }
    return retVal;
}

void CSI_StereoCamera::computeDisp(const bool filter, cv::Mat& disparity)
{
	VPIImageData data;
    VPIStereoDisparityEstimatorParams params;
    params.windowSize   = 5;
    params.maxDisparity = 32;

	int minDisp;
	int specklewindowsize;
	int disp12diff;

//	mLCam.getFiltered(lFil);
//	mRCam.getFiltered(rFil);
////	mMedianFilter->apply(lRect, mLeftGPU);
////	mMedianFilter->apply(rRect, mRightGPU);
//	mLeftGPU.upload(lFil);
//	mRightGPU.upload(rFil);

#ifdef LOG
	int64 time1 = cv::getTickCount();
#endif /* LOG */
//	mStereoBM->compute(mLeftGPU, mRightGPU, mDisparity);
	vpiSubmitStereoDisparityEstimator(mLCam.getStream(), VPI_BACKEND_CUDA, mVPIStereo,
			mLCam.getFiltered16(), mRCam.getFiltered16(), mVPILeftRightDisp, nullptr, &params);
#ifdef LOG
	int64 time2 = cv::getTickCount();
#endif /* LOG */

    if (filter)
    {
//    	/* Store parameters specific for left-right disparity. */
//    	minDisp = mStereoBM->getMinDisparity();
//    	specklewindowsize = mStereoBM->getSpeckleWindowSize();
//    	disp12diff = mStereoBM->getDisp12MaxDiff();
//
//    	mDisparity.download(mDisparityCPU);
//
//    	/* We need to change parameters for calculating right-left disparity. */
//    	mStereoBM->setMinDisparity(-(mStereoBM->getMinDisparity() + mStereoBM->getNumDisparities()) + 1);
//    	mStereoBM->setDisp12MaxDiff(1000000);
//		mStereoBM->setSpeckleWindowSize(0);
//
//		mStereoBM->compute(mRightGPU, mLeftGPU, mDisparity);
    	vpiSubmitStereoDisparityEstimator(mLCam.getStream(), VPI_BACKEND_CUDA, mVPIStereo,
    			mRCam.getFiltered16(), mLCam.getFiltered16(), mVPIRightLeftDisp, nullptr, &params);
    	vpiStreamSync(mLCam.getStream());
#ifdef LOG
    	int64 time3 = cv::getTickCount();
#endif /* LOG */

//    	/* We rever parameters back to calculating left-right disparity. */
//    	mStereoBM->setMinDisparity(minDisp);
//    	mStereoBM->setDisp12MaxDiff(disp12diff);
//		mStereoBM->setSpeckleWindowSize(specklewindowsize);

//    	mLeftGPU.download(mLeftCPU);
//    	mDisparity.download(mDisparityRLCPU);


    	vpiImageLock(mLCam.getFiltered(), VPI_LOCK_READ, &data);
    	vpiImageDataExportOpenCVMat(data, &mLeftCPU);
        vpiImageUnlock(mLCam.getFiltered());
    	vpiImageLock(mVPILeftRightDisp, VPI_LOCK_READ, &data);
    	vpiImageDataExportOpenCVMat(data, &mDisparityCPU);
        vpiImageUnlock(mVPILeftRightDisp);
    	vpiImageLock(mVPIRightLeftDisp, VPI_LOCK_READ, &data);
    	vpiImageDataExportOpenCVMat(data, &mDisparityRLCPU);
        vpiImageUnlock(mVPIRightLeftDisp);

        mDisparityCPU.convertTo(mDisparityCPU, CV_16SC1);
        mDisparityRLCPU.convertTo(mDisparityRLCPU, CV_16SC1);

    	mDispWLSFilter->filter(mDisparityCPU, mLeftCPU, filteredDisp, mDisparityRLCPU);

    	cv::ximgproc::getDisparityVis(filteredDisp, disparity, 1/256.0);
//    	filteredDisp.convertTo(disparity, CV_8UC1, 1.0 / 256.0);

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
    	VPIConvertImageFormatParams cvtParams;
    	vpiInitConvertImageFormatParams(&cvtParams);
    	cvtParams.scale = 255.0f/(64*32-1);
    	vpiSubmitConvertImageFormat(mLCam.getStream(), VPI_BACKEND_CUDA, mVPILeftRightDisp, mVPIOutDisp, &cvtParams);
    	vpiStreamSync(mLCam.getStream());

    	vpiImageLock(mVPIOutDisp, VPI_LOCK_READ, &data);
    	vpiImageDataExportOpenCVMat(data, &disparity);
    	vpiImageUnlock(mVPIOutDisp);
//    	mDisparity.download(disparity);
#ifdef LOG
    	printf("Disparity calculated in %f \n", static_cast<double>(time2 - time1) / cv::getTickFrequency());
#endif /* LOG */
    }
}
