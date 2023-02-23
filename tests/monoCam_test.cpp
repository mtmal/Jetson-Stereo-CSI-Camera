////////////////////////////////////////////////////////////////////////////////
// Copyright (C) 2023 Mateusz Malinowski
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
#include <CSI_Camera.h>

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
    printf("    -i, --id        -> sets the ID of the camera to start, default: 0 \n");
    printf("\nExample: %s -c 320 -r 240 \n\n", name);
    printf("NOTE: if the application that uses nvargus to control cameras was killed without releasing the cameras,"
        " execute the following:\n\n"
        "$ sudo systemctl restart nvargus-daemon \n\n");
}

/**
 * Runs the application.
 *  @param imageSize the size to which images should be resized.
 *  @param framerate the framerate at which cameras should acquire images.
 *  @param id the ID of the camera to start.
 *  @param mode the mode at which cameras should operate.
 */
static void run(const cv::Size& imageSize, const uint8_t framerate, const uint8_t mode, const uint8_t id)
{
    /** Current key pressed by the user. */
    int key = 0;
    /** The mono camera class. */
    CSI_Camera camera(imageSize);
    /** Buffer for image */
    cv::Mat img;

    if (camera.startCamera(framerate, mode, id, 0))
    {
        cv::namedWindow("CSI Camera", cv::WINDOW_AUTOSIZE);
        puts("Hit ESC to exit");

        while (key != 27)
        {
            if (camera.getRawImage(img))
            {
                cv::imshow("CSI Camera", img);
            }
            key = cv::waitKey(30) & 0xff;
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
    /** Image size for final images (after resize). */
    cv::Size imageSize(640, 480);
    /** Framerate at which CSI cameras should capture images. */
    uint8_t framerate = 20;
    /** The mode in which CSI cameras should operate. */
    uint8_t mode = 0;
    /** The ID of the CSI camera in case there are mutliple cameras. */
    uint8_t id = 0;

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
        else if ((0 == strcmp(argv[i], "--id")) || (0 == strcmp(argv[i], "-i")))
        {
            id = static_cast<uint8_t>(atoi(argv[i + 1]));
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
    run(imageSize, framerate, mode, id);

    return 0;
}
