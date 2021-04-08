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

#include <opencv2/highgui.hpp>
#include <opencv2/cudastereo.hpp>
#include <opencv2/ximgproc/disparity_filter.hpp>
#include <CSI_StereoCamera.h>

// to restart CSI camera in system use command:
// $ sudo systemctl restart nvargus-daemon

/** Mutex for synchronisation between the main thread and callbacks. */
pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
/* Flag to indicate if disparity map filtering should be applied. */
std::atomic<bool> useFiltered = false;

/* Parameters used by tracker bars. */
int filtered;
int disp12Diff;
int preFilterType;
int preFilterSize;
int preFilterCap;
int blockSize;
int minDisparity;
int numDisparity;
int uniqueness;
int texThreshold;
int speckleSize;
int speckleRange;
int lambda;
int sigma;

/**
 * Prints help information about this application.
 *  @param name the name of the executable.
 */
static void printHelp(const char* name)
{
	printf("Usage: %s [options] \n", name);
	printf("    -h, --help      -> prints this information \n");
	printf("    -m, --mode      -> sets the specific camera mode, default: 0 \n");
	printf("    -f, --framerate -> sets the camera framerate in Hz, default: 20 \n");
	printf("    -c, --cols      -> sets the number of columns (width) in resized image, default: 640 \n");
	printf("    -r, --rows      -> sets the number of rows (height) in resized image, default: 480 \n");
	printf("\nExample: %s -c 320 -r 240 \n\n", name);
	printf("NOTE: if the application that uses nvargus to control cameras was killed without releasing the cameras,"
			" execute the following:\n\n"
			"$ sudo systemctl restart nvargus-daemon \n\n");
}

/**
 * Call-back assigned to all track bars. Updates all parameters.
 *  @param
 *  @param data the pointer to CSI_StereoCamera class.
 */
static void onTrackbar(int, void* data)
{
    CSI_StereoCamera* stereo = static_cast<CSI_StereoCamera*>(data);
    useFiltered = static_cast<bool>(filtered);

    if (0 == blockSize % 2)
    {
        ++blockSize;
    }
    if (0 == preFilterSize % 2)
    {
        ++preFilterSize;
    }

    pthread_mutex_lock(&mutex);
    stereo->getStereoBM()->setPreFilterType(preFilterType);
    stereo->getStereoBM()->setPreFilterSize(preFilterSize);
    stereo->getStereoBM()->setPreFilterCap(preFilterCap);
    stereo->getStereoBM()->setBlockSize(blockSize);
    stereo->getStereoBM()->setMinDisparity(minDisparity * 16);
    stereo->getStereoBM()->setNumDisparities(numDisparity * 16);
    if (useFiltered)
    {
		stereo->getStereoBM()->setSpeckleWindowSize(0);
		stereo->getStereoBM()->setSpeckleRange(0);
		stereo->getStereoBM()->setDisp12MaxDiff(1000000);
	    stereo->getStereoBM()->setUniquenessRatio(0);
	    stereo->getStereoBM()->setTextureThreshold(0);
    }
    else
    {
		stereo->getStereoBM()->setSpeckleWindowSize(speckleSize);
		stereo->getStereoBM()->setSpeckleRange(speckleRange);
		stereo->getStereoBM()->setDisp12MaxDiff(disp12Diff);
	    stereo->getStereoBM()->setUniquenessRatio(uniqueness);
	    stereo->getStereoBM()->setTextureThreshold(texThreshold);
    }

    stereo->restartDispFilter(static_cast<double>(lambda * 10), static_cast<double>(sigma) / 1000.0);
    pthread_mutex_unlock(&mutex);

    if (useFiltered)
    {
		printf("Parameters: \n"
			   "\tPrefilter Type: %d\n"
			   "\tPrefilter Size: %d\n"
			   "\tPrefilter Cap: %d\n"
			   "\tBlock Size: %d\n"
			   "\tMin Disparity: %d\n"
			   "\tNum Disparities: %d\n"
			   "\tUniqueness Ratio: 0\n"
			   "\tTexture threshold: 0\n"
			   "\tSpeckle Window Size: 0\n"
			   "\tSpeckle Range: 0\n", preFilterType, preFilterSize, preFilterCap, blockSize,
			   minDisparity * 16, numDisparity * 16);
    }
    else
    {
		printf("Parameters: \n"
			   "\tPrefilter Type: %d\n"
			   "\tPrefilter Size: %d\n"
			   "\tPrefilter Cap: %d\n"
			   "\tBlock Size: %d\n"
			   "\tMin Disparity: %d\n"
			   "\tNum Disparities: %d\n"
			   "\tUniqueness Ratio: %d\n"
			   "\tTexture threshold: %d\n"
			   "\tSpeckle Window Size: %d\n"
			   "\tSpeckle Range: %d\n", preFilterType, preFilterSize, preFilterCap, blockSize,
			   minDisparity * 16, numDisparity * 16, uniqueness, texThreshold, speckleSize, speckleRange);
    }

    printf("Filter parameters: \n"
           "\tlambda: %d \n"
           "\tsigma colour: %f \n", lambda * 10, static_cast<double>(sigma) / 1000.0);
}

/**
 * Creates sliders to control disparity map parameters.
 *  @param stereoCam the stereo camera from which initial parameters can be taken and which will be used in callbacks.
 */
static void createSlides(CSI_StereoCamera& stereoCam)
{
	preFilterType = stereoCam.getStereoBM()->getPreFilterType();
    preFilterSize = stereoCam.getStereoBM()->getPreFilterSize();
    preFilterCap  = stereoCam.getStereoBM()->getPreFilterCap();
    blockSize     = stereoCam.getStereoBM()->getBlockSize();
    minDisparity  = stereoCam.getStereoBM()->getMinDisparity() / 16;
    numDisparity  = stereoCam.getStereoBM()->getNumDisparities() / 16;
    uniqueness    = stereoCam.getStereoBM()->getUniquenessRatio();
    texThreshold  = stereoCam.getStereoBM()->getTextureThreshold();
    speckleSize   = stereoCam.getStereoBM()->getSpeckleWindowSize();
    speckleRange  = stereoCam.getStereoBM()->getSpeckleRange();
    disp12Diff	  = stereoCam.getStereoBM()->getDisp12MaxDiff();

    lambda		  = static_cast<int>(stereoCam.getDispFilter()->getLambda() / 10);
    sigma         = static_cast<int>(stereoCam.getDispFilter()->getSigmaColor() * 1000.0);

    cv::createTrackbar("Filter Disparity", "Disparity", &filtered, 1, onTrackbar, &stereoCam);

    cv::createTrackbar("Prefilter Type", "Disparity", &preFilterType, 1, onTrackbar, &stereoCam);
    cv::createTrackbar("Prefilter Size", "Disparity", &preFilterSize, 255, onTrackbar, &stereoCam);
    cv::setTrackbarMin("Prefilter Size", "Disparity", 5);

    cv::createTrackbar("Prefilter Cap", "Disparity", &preFilterCap, 63, onTrackbar, &stereoCam);
    cv::setTrackbarMin("Prefilter Cap", "Disparity", 1);

    cv::createTrackbar("Block Size", "Disparity", &blockSize, 51, onTrackbar, &stereoCam);
    cv::setTrackbarMin("Block Size", "Disparity", 5);

    cv::createTrackbar("Minimum Disparity Mult", "Disparity", &minDisparity, 8, onTrackbar, &stereoCam);
    cv::setTrackbarMin("Minimum Disparity Mult", "Disparity", -8);

    cv::createTrackbar("Number of Disparities Mult", "Disparity", &numDisparity, 16, onTrackbar, &stereoCam);
    cv::setTrackbarMin("Number of Disparities Mult", "Disparity", 2);

    cv::createTrackbar("Uniqueness Ratio", "Disparity", &uniqueness, 100, onTrackbar, &stereoCam);

    cv::createTrackbar("Texture Threshold", "Disparity", &texThreshold, 100, onTrackbar, &stereoCam);
    cv::createTrackbar("Speckle Size", "Disparity", &speckleSize, 1000, onTrackbar, &stereoCam);
    cv::createTrackbar("Speckle range", "Disparity", &speckleRange, 100, onTrackbar, &stereoCam);

    cv::createTrackbar("Disp 12 Diff", "Disparity", &disp12Diff, 100, onTrackbar, &stereoCam);

    cv::createTrackbar("Filter Lambda", "Disparity", &lambda, 1000, onTrackbar, &stereoCam);
    cv::setTrackbarMin("Filter Lambda", "Disparity", 500);
    cv::createTrackbar("Filter Sigma Colour", "Disparity", &sigma, 3000, onTrackbar, &stereoCam);
}

static void saveXYZ(const std::string& filename, const cv::Mat& mat)
{
    static const double max_z = 1;
    FILE* fp = fopen(filename.c_str(), "wt");
    for (int y = 0; y < mat.rows; ++y)
    {
        for (int x = 0; x < mat.cols; ++x)
        {
        	const cv::Vec3f& point = mat.at<cv::Vec3f>(y, x);
            if (fabs(point[2] - max_z) >= FLT_EPSILON && fabs(point[2]) <= max_z)
            {
            	fprintf(fp, "%f %f %f\n", point[0], point[1], point[2]);
            }
        }
    }
    fclose(fp);
}

/**
 * Runs the application.
 *  @param imageSize the size to which images should be resized.
 *  @param framerate the framerate at which cameras should acquire images.
 *  @param mode the mode at which cameras should operate.
 */
static void run(const cv::Size& imageSize, const uint8_t framerate, const uint8_t mode)
{
    /** Flag to pause processing images. */
    bool pause = false;
    /** Current key pressed by the user. */
    int key = 0;
    /* Timestamps at various steps for providing elapsed time. */
    int64 time1, time2, time3, time4;
    /** Buffers for images. */
    cv::Mat imgs[2] = {cv::Mat(imageSize, CV_8UC1), cv::Mat(imageSize, CV_8UC1)};
    /** Buffer for disparity map. */
    cv::Mat disparity(imageSize, CV_8UC1);
    cv::Mat pointCloud;
    /** The stereo camera class. */
    CSI_StereoCamera stereo(imageSize);

    stereo.loadCalibration("./config");
    if (stereo.startCamera(framerate, mode))
    {
		cv::namedWindow("Left CSI Camera", cv::WINDOW_AUTOSIZE);
		cv::namedWindow("Right CSI Camera", cv::WINDOW_AUTOSIZE);
		cv::namedWindow("Disparity", cv::WINDOW_AUTOSIZE);

		createSlides(stereo);

		puts("Hit ESC to exit");
		while (key != 27)
		{
			time1 = cv::getTickCount();
			if (!pause && stereo.getRectified(false, imgs[0], imgs[1]))
			{
				time2 = cv::getTickCount();

				pthread_mutex_lock(&mutex);
				stereo.computeDisp(useFiltered, disparity, pointCloud);
				pthread_mutex_unlock(&mutex);
				time3 = cv::getTickCount();

				cv::imshow("Left CSI Camera", imgs[0]);
				cv::imshow("Right CSI Camera", imgs[1]);
				cv::imshow("Disparity", disparity);
				time4 = cv::getTickCount();

				printf("%f: Processed in: %f s; Disparity: %f s; Displayed: %f s\n",
					 static_cast<double>(time1) / cv::getTickFrequency(),
					 static_cast<double>(time2 - time1) / cv::getTickFrequency(),
					 static_cast<double>(time3 - time2) / cv::getTickFrequency(),
					 static_cast<double>(time4 - time3) / cv::getTickFrequency());
//				saveXYZ(std::to_string(static_cast<double>(time1) / cv::getTickFrequency()) + ".xyz", pointCloud);
			}
			key = cv::waitKey(30) & 0xff;
			/* when space bar is pressed, pause processing images and save current rectified images with disparity map to files. */
			if (key == 32)
			{
				pause = !pause;

				cv::imwrite("left.png", imgs[0]);
				cv::imwrite("right.png", imgs[1]);
				cv::imwrite("disparity.png", disparity);
				saveXYZ("pc.xyz", pointCloud);
			}
		}

		cv::destroyAllWindows();
    }
    else
    {
    	puts("Failed to open camera.");
    }
}

int main(int argc, char** argv)
{
	/** Image size for final images. */
    cv::Size imageSize(640, 480);
    /** Framerate at which CSI cameras should capture images. */
    uint8_t framerate = 20;
    /** The mode in which CSI cameras should operate. */
    uint8_t mode = 0;

    for (int i = 1; i < argc; ++i)
    {
        if ((0 == strcmp(argv[i], "--mode")) || (0 == strcmp(argv[i], "-m")))
        {
            mode = static_cast<uint8_t>(atoi(argv[i + 1]));
        }
        else if ((0 == strcmp(argv[i], "--framerate")) || (0 == strcmp(argv[i], "-f")))
        {
        	framerate = static_cast<uint8_t>(atoi(argv[i + 1]));
        }
        else if ((0 == strcmp(argv[i], "--cols")) || (0 == strcmp(argv[i], "-c")))
        {
        	imageSize.width = atoi(argv[i + 1]);
        }
        else if ((0 == strcmp(argv[i], "--rows")) || (0 == strcmp(argv[i], "-r")))
        {
        	imageSize.height = atoi(argv[i + 1]);
        }
        else if ((0 == strcmp(argv[i], "--help")) || (0 == strcmp(argv[i], "-h")))
        {
        	printHelp(argv[0]);
        	return 0;
        }
		else
		{
			/* nothing to do in here */
		}
    }
    run(imageSize, framerate, mode);
    return 0;
}
