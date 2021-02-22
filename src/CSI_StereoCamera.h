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

#ifndef __CSI_STEREOCAMERA_H__
#define __CSI_STEREOCAMERA_H__

#include "CSI_Camera.h"
#include "FrameTimeChecker.h"

/** Forward declaration of OpenCV classes. */
namespace cv
{
	namespace cuda
	{
		class Filter;
		class StereoBM;
	}
	namespace ximgproc
	{
		class DisparityWLSFilter;
	}
}

/**
 * This class wraps two CSI cameras to treat them as a stereo rig. It provides
 * software synchronisation which minimises the risk of out of sync frames,
 * but will never guarantee perfect synchronisation.
 *
 * This class tries to use as much CUDA-enabled OpenCV functions as possible.
 * However, for the best disparity map creation performance, it applies a filter
 * from Contrib repository which currently offers CPU-only implementation.
 *
 * @note Usually a disparity is of 16-bit fixed-point or 32-bit floating-point type.
 * However, CUDA implementation changes that to 8-bit fixed-point. There would be
 * a loss in precision, but for small project where stereo camera does not look far away
 * it may be enough.
 *
 * @note Minimum number of disparities seems to have no effect on CUDA-enabled stereBM
 * implementation.
 *
 * @note this class was tested with two IMX219-83 cameras.
 *
 * TODO: implement disparity filter using CUDA.
 * TODO: add functionality to reproject disparity to point cloud.
 */
class CSI_StereoCamera
{
public:
	/**
	 * Basic constructor which initialises all variables.
	 *  @param imageSize the size of images for buffers allocation.
	 */
    CSI_StereoCamera(const cv::Size& imageSize);

    /**
     * Class destructor. Empty at the moment.
     */
    virtual ~CSI_StereoCamera();

    /**
     * Starts both cameras. If at least one fails to start, it ensures both are stopped.
     *  @param framerate the camera's framerate in Hz.
     *  @param mode the mode of the camera - each camera may have different mode specification.
     *  @param lCamID the id of the left camera.
     *  @param rCamID the id of the right camera.
     *  @parma flip the flip parameter. Usually 0 (no rotation) or 2 (180 deg).
     *	@return true if both cameras have started correctly.
     */
    bool startCamera(const uint8_t framerate, const uint8_t mode = 0, const uint8_t lCamID = 0,
    		const uint8_t rCamID = 1, const uint8_t flip = 2);

    /**
     * Stops both cameras.
     */
    inline void stopCamera()
    {
    	mLCam.stopCamera();
    	mRCam.stopCamera();
    }

    /**
     *  @reutrn true if both cameras have been initialised.
     */
    inline bool isInitialised() const
    {
        return mLCam.isInitialised() && mRCam.isInitialised();
    }

    /**
     * Captures raw RGB images from stereo camera.
     *  @param[out] lImg the raw RGB image from the left camera.
     *  @param[out] rImg the raw RGB image from the right camera.
     *  @return true if it is safe to assume that both images are synchronised.
     */
    inline bool getRawImages(cv::Mat& lImg, cv::Mat& rImg)
    {
        return mFTC.checkTimes(mLCam.getRawImage(lImg), mRCam.getRawImage(rImg));
    }

    /**
     * Captures image, converts it to greyscale and rectifies it.
     *  @param forceProcessing if set to true raw images will be rectified
     *  even if the images might be not synchronised. If set to false, processing
     *  is not performed and @p lImg and @p rImg are not updated.
     *  @param[out] lImg the rectified greyscale image from the left camera.
     *  @param[out] rImg the rectified greyscale image from the right camera.
     *  @return true if it is safe to assume that both images are synchronised.
     */
    bool getRectified(const bool forceProcessing, cv::Mat& lImg, cv::Mat& rImg);

    /**
     *  @return the pointer to stereoBM class for configuration.
     *  Handy for dynamic parameter selection with custom UI.
     */
    inline cv::Ptr<cv::cuda::StereoBM> getStereoBM() const
    {
        return mStereoBM;
    }

    /**
     *  @return the pointer to disparity map filter class for configuration.
     *  Handy for dynamic parameter selection with custom UI.
     */
    inline cv::Ptr<cv::ximgproc::DisparityWLSFilter> getDispFilter() const
    {
        return mDispWLSFilter;
    }

    /**
     * Restarts disparity filter. It needs to be called whenever new stereoBM
     * or disparity map filter parameters were set, because when disparity map
     * filter is created, it overrides some parameters of stereoBM.
     * @note Do not call this function if you do not want to use disparity map filter.
     *  @param lambda one of the parameters for disparity map filter. Typical value is 8000.
     *  @param sigmaColour one of the parameters for disparity map filter. Typical values range from 0.8 to 2.0.
     */
    void restartDispFilter(const double lambda, const double sigmaColour);

    /**
     * Loads stereo camera intrinsics and extrinsics from files located in the @p folder.
     * Expects left.xml, right.xml, and stereo.xml.
     *  @param folder the folder with XML configuration files.
     *  @return true if all configuration was read correctly. Error may occur only if at least one file failed to open.
     */
    bool loadCalibration(const std::string& folder);

    /**
     * Computes a disparity map using rectified greyscale images. Because IMX219-83 is quite noisy,
     * a median filter is applied to rectified images before calculating the disparity map.
     *  @param filter if set to true the disparity map will be filtered.
     *  @param[out] disparity the preallocated buffer for disparity map.
     */
    void computeDisp(const bool filter, cv::Mat& disparity);

private:
    /** The wrapper for left CSI camera. */
    CSI_Camera mLCam;
    /** The wrapper for right CSI camera. */
    CSI_Camera mRCam;
    /** Preallocated shared buffer for the left-right disparity map. */
    cv::cuda::HostMem mDisparityLR;
    /** Preallocated shared buffer for the right-left disparity map used in filtering. */
    cv::cuda::HostMem mDisparityRL;
    /** Pointer to CUDA-enabled stereo block-matching algorithm. */
    cv::Ptr<cv::cuda::StereoBM> mStereoBM;
    /** Pointer to disparity filter algorithm. */
    cv::Ptr<cv::ximgproc::DisparityWLSFilter> mDispWLSFilter;
    /** Utility class used for comparing stereo pair frame times to access if they are synchronised or not. */
    FrameTimeChecker mFTC;
};

#endif // __CSI_STEREOCAMERA_H__
