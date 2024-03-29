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

#pragma once

#include <opencv2/cudawarping.hpp>
#include <GenericListener.h>
#include <GenericThread.h>
#include "CameraData.h"
#include "CSI_Camera.h"


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
class CSI_StereoCamera : public GenericListener<CameraData>,
                         public ICameraTalker,
                         protected GenericThread<CSI_StereoCamera>
{
    /* Relax the access control to baseclass which is inherited as protected. GenericThread is inherited as protected, because
     * ICameraTalker controls the camera and starting/stopping threads. */
    friend class GenericThread<CSI_StereoCamera>;

public:
	/**
	 * Basic constructor which initialises all variables.
	 *  @param imageSize the size of images for buffers allocation.
	 */
    explicit CSI_StereoCamera(const cv::Size& imageSize);

    /**
     * Class destructor. Empty at the moment.
     */
    virtual ~CSI_StereoCamera();

    /**
     * Starts both cameras. If at least one fails to start, it ensures both are stopped.
     *  @param imageSize the size to which all images will be resized, if it does not match the size of specified @p mode.
     *  @param framerate the camera's framerate in Hz.
     *  @param mode the mode of the camera - each camera may have different mode specification.
     *  @param ids the list of camera ids.
     *  @param flip the flip parameter. Usually 0 (no rotation) or 2 (180 deg).
     *  @param colour true to get BGR images, false for greyscale.
     *  @param rectify whether to request rectified/undistorted images.
     *	@return true if both cameras have started correctly.
     */
    bool startCamera(const cv::Size& imageSize, const uint8_t framerate, const uint8_t mode = 0, 
                     const std::vector<uint8_t>& ids = {0, 1}, const uint8_t flip = 2, 
                     const bool colour = false, const bool rectified = true) override;

    /**
     * Stops both cameras.
     */
    void stopCamera() override;

    /**
     *  @reutrn true if both cameras have been initialised.
     */
    bool isInitialised() const override;

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
     *  @param lImg undistorted image from the left camera.
     *  @param rImg undistorted image from the right camera.
     *  @param[out] disparity the preallocated buffer for disparity map.
     */
    void computeDisp(const bool filter, const cv::cuda::HostMem& lImg, const cv::cuda::HostMem& rImg, cv::Mat& disparity);

    // override
    void update(const CameraData& camData) override;
    inline bool isRunning() const override
    {
        return GenericThread<CSI_StereoCamera>::isRunning();
    }

    /**
     * The main body of the thread that constantly retrieves the latest image from CSI cameras.
     *  @return nullptr
     */
    void* threadBody();

private:
    /** The size of images. */
    cv::Size mImageSize;
    /** Flag to indicate if rectified images should be requested. */
    bool mRequestedRect;
    /** The wrapper for left CSI camera. */
    CSI_Camera mLCam;
    /** The wrapper for right CSI camera. */
    CSI_Camera mRCam;
    /** Shared buffer for initial disparity map. */
    cv::cuda::HostMem mDisparity;
    /** Shared buffer for the result of median filtering on left image. */
    cv::cuda::HostMem mLeftGPU;
    /** GPU buffer for the result of median filtering on right image. */
    cv::cuda::GpuMat mRightGPU;
    /** Buffer for flipped left camera image. */
    cv::cuda::GpuMat mFlippedLeft;
    /** Buffer for flipped right camera image. */
    cv::cuda::GpuMat mFlippedRight;
    /** Shared buffer for the right-left disparity used in filtering. */
    cv::cuda::HostMem mDisparityRLCPU;
    /** Pointer to CUDA-enabled median filter. */
    cv::Ptr<cv::cuda::Filter> mMedianFilter;
    /** Pointer to CUDA-enabled stereo block-matching algorithm. */
    cv::Ptr<cv::cuda::StereoBM> mStereoBM;
    /** Pointer to disparity filter algorithm. */
    cv::Ptr<cv::ximgproc::DisparityWLSFilter> mDispWLSFilter;
    /** Shared buffers for rectification maps. */
    cv::cuda::HostMem mRectMaps[2][2];
    /** Shared buffers for left and right camera data. */
    CameraData mCamDatas;
};
