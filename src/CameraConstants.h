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

#ifndef __CAMERACONSTANTS_H__
#define __CAMERACONSTANTS_H__

#include <string>

/** The expected extension for calibration files. */
static const std::string CALIB_FILE_EXTENSION = ".xml";
/** The expected name of file with left camera intrinsics. */
static const std::string LEFT_CALIB_FILE = "left";
/** The expected name of file with right camera intrinsics. */
static const std::string RIGHT_CALIB_FILE = "right";
/** The expected name of file with stereo camera extrinsic. */
static const std::string STEREO_CALIB_FILE = "stereo";
/** The name of the node in XML file with camera matrix. */
static const std::string CAMERA_MATRIX = "CameraMatrix";
/** The name of the node in XML file with distortion parameters. */
static const std::string DISTORTION = "Distortion";
/** The name of the node in XML file with right camera rotation */
static const std::string ROTATION = "R";
/** The name of the node in XML file with right camera translation */
static const std::string TRANSLATION = "T";
/** The name of the node in XML file with left camera rectification rotation */
static const std::string RECTIFICATION_LEFT = "R1";
/** The name of the node in XML file with right camera rectification rotation */
static const std::string RECTIFICATION_RIGHT = "R2";
/** The name of the node in XML file with new left camera matrix. */
static const std::string NEW_CAM_MATRIX_LEFT = "P1";
/** The name of the node in XML file with new right camera matrix. */
static const std::string NEW_CAM_MATRIX_RIGHT = "P2";
/** The name of the node in XML file with disparity to depth parameters. */
static const std::string DISPARITY_TO_DEPTH = "Q";

#endif // __CAMERACONSTANTS_H__
