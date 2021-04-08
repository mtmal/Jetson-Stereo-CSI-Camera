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
#include <opencv2/ximgproc/edge_filter.hpp>
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

/**
 * Scales centre of image and focal length in the Q matrix.
 * TODO: same fix needs to be applied as described for scaleCameraMatrix function.
 *  @param imgSize the size of images which will be acquired from the cameras.
 *  @param maxSize the max size of images for which the calibration was done.
 *  @param[in/out] Q the perspective transformation matrix.
 */
void scaleQMatrix(const cv::Size& imgSize, const cv::Size& maxSize, cv::Mat& Q)
{
	Q.at<double>(0, 3) *= static_cast<double>(imgSize.width)  / static_cast<double>(maxSize.width);
	Q.at<double>(1, 3) *= static_cast<double>(imgSize.height) / static_cast<double>(maxSize.height);
	Q.at<double>(2, 3) *= static_cast<double>(imgSize.width)  / static_cast<double>(maxSize.width);
}

cv::Ptr<cv::StereoMatcher> mRightMatcher;

} /* end of the anonymous namespace */

CSI_StereoCamera::CSI_StereoCamera(const cv::Size& imageSize)
: mLCam(imageSize), mRCam(imageSize), mDisparityLR(imageSize, CV_16S, cv::cuda::HostMem::SHARED),
  mDisparityRL(imageSize, CV_8UC1, cv::cuda::HostMem::SHARED),
  mDisparityF(imageSize, CV_8UC1, cv::cuda::HostMem::SHARED), mQ(),
  mPointCloud(imageSize, CV_32FC3, cv::cuda::HostMem::SHARED),
  mStereoBM(cv::StereoBM::create(160, 15))
{
	mStereoBM->setPreFilterType(1);
	restartDispFilter(8000.0, 1.5);

	mConfidence = cv::Mat(imageSize, CV_8UC1);
}

CSI_StereoCamera::~CSI_StereoCamera()
{
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
	bool retVal = mFTC.checkTimes(mLCam.acquireGreyScale(), mRCam.acquireGreyScale());
	if (retVal || forceProcessing)
	{
		/* perform processing only if images are OK or we specifically required it. */
		mLCam.rectifyImage();
		mRCam.rectifyImage();
		left = mLCam.getFilteredImg().createMatHeader();
		right = mRCam.getFilteredImg().createMatHeader();
	}
    return retVal;
}

void CSI_StereoCamera::restartDispFilter(const double lambda, const double sigmaColour)
{
	mRightMatcher = cv::ximgproc::createRightMatcher(mStereoBM);
	mDispWLSFilter = cv::ximgproc::createDisparityWLSFilter(mStereoBM);
	mDispWLSFilter->setLambda(lambda);
	mDispWLSFilter->setSigmaColor(sigmaColour);
}

bool CSI_StereoCamera::loadCalibration(const std::string& folder)
{
    cv::Size maxSize;
    cv::Mat lCamMat, rCamMat, lDist, rDist;
    cv::Mat R, T, R1, R2, P1, P2;
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
        scaleCameraMatrix(mDisparityLR.size(), maxSize, lCamMat);

        fs.open(folder + "/" + RIGHT_CALIB_FILE + CALIB_FILE_EXTENSION, cv::FileStorage::READ);
        retVal &= fs.isOpened();
        if (retVal)
        {
            fs[CAMERA_MATRIX] >> rCamMat;
            fs[DISTORTION] >> rDist;
            fs.release();
            scaleCameraMatrix(mDisparityLR.size(), maxSize, rCamMat);

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
                fs[DISPARITY_TO_DEPTH]  >> mQ;
                fs.release();
                scaleCameraMatrix(mDisparityLR.size(), maxSize, P1);
                scaleCameraMatrix(mDisparityLR.size(), maxSize, P2);
                scaleQMatrix(mDisparityLR.size(), maxSize, mQ);
                mQ.convertTo(mQ, CV_32F);

                initUndistortRectifyMap(lCamMat, lDist, R1, P1, mDisparityLR.size(), CV_32FC1, rectMap[0], rectMap[1]);
                mLCam.setRMap(rectMap[0], rectMap[1]);
                initUndistortRectifyMap(rCamMat, rDist, R2, P2, mDisparityLR.size(), CV_32FC1, rectMap[0], rectMap[1]);
                mRCam.setRMap(rectMap[0], rectMap[1]);
            }
        }
    }
    return retVal;
}

void CSI_StereoCamera::computeDisp(const bool filter, cv::Mat& disparity, cv::Mat& pointCloud)
{
	int minDisp;
	int specklewindowsize;
	int disp12diff;

#ifdef LOG
	int64 time1 = cv::getTickCount();
#endif /* LOG */
	mStereoBM->compute(mLCam.getFilteredImg(), mRCam.getFilteredImg(), mDisparityLR);
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
//    	/* We need to change parameters for calculating right-left disparity. */
//    	mStereoBM->setMinDisparity(-(mStereoBM->getMinDisparity() + mStereoBM->getNumDisparities()) + 1);
//    	mStereoBM->setDisp12MaxDiff(1000000);
//		mStereoBM->setSpeckleWindowSize(0);

		mRightMatcher->compute(mRCam.getFilteredImg(), mLCam.getFilteredImg(), mDisparityRL);

//    	mStereoBM->compute(mRCam.getFilteredImg(), mLCam.getFilteredImg(), mDisparityRL);
#ifdef LOG
    	int64 time3 = cv::getTickCount();
#endif /* LOG */

//    	/* We revert parameters back to calculating left-right disparity. */
//    	mStereoBM->setMinDisparity(minDisp);
//    	mStereoBM->setDisp12MaxDiff(disp12diff);
//		mStereoBM->setSpeckleWindowSize(specklewindowsize);

    	mDispWLSFilter->filter(mDisparityLR, mLCam.getFilteredImg(), mDisparityF, mDisparityRL);
    	mConfidence = mDispWLSFilter->getConfidenceMap();

//    	cv::Mat disp;
//    	double fbs_spatial = 16.0;
//    	double fbs_luma = 8.0;
//    	double fbs_chroma = 8.0;
//    	double fbs_lambda = 128.0;
//    	cv::ximgproc::fastBilateralSolverFilter(mLCam.getFilteredImg(), mDisparityF, mConfidence/255.0f, disp,
//    			fbs_spatial, fbs_luma, fbs_chroma, fbs_lambda);



//    	cv::Mat temp;
//    	printf("type: %d \n", mDisparityF.type());
//    	mDisparityF.createMatHeader().convertTo(temp, CV_32F, 1.0 / 16.0);
    	cv::reprojectImageTo3D(mDisparityF, mPointCloud, mQ);
    	mDisparityF.createMatHeader().convertTo(disparity, CV_8U, 255/(mStereoBM->getNumDisparities()*16.));
//    	disparity = mDisparityF.createMatHeader();
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
//    	cv::Mat temp;
//    	mDisparityLR.createMatHeader().convertTo(temp, CV_32F, 1.0 / 16.0);
    	cv::reprojectImageTo3D(mDisparityLR, mPointCloud, mQ);
    	mDisparityLR.createMatHeader().convertTo(disparity, CV_8U, 255/(mStereoBM->getNumDisparities()*16.));
//    	disparity = mDisparityLR.createMatHeader();
#ifdef LOG
    	printf("Disparity calculated in %f \n", static_cast<double>(time2 - time1) / cv::getTickFrequency());
#endif /* LOG */
    }
    pointCloud = mPointCloud.createMatHeader();
}
