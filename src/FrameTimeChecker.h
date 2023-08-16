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

/**
 * This class encapsulates variables and methods for the comparison
 * of two camera frame times to make sure they are close to each other
 * and therefore belong to the same stereo pair. This is an additional
 * support for software-synchronised stereo image capture.
 */
class FrameTimeChecker
{
public:
	/**
	 * Constructor, initialises all variables.
	 *  @param stdDevMult the multiplier for standard deviation to test when stereo pair is out of sync.
	 *  @param counterThreshold the threshold for the number of stereo pairs before the failure can occur.
	 *  Basically, the first @p counterThreshold stereo pairs will be always assumed OK so that we accumulate
	 *  enough data for mean and standard deviation estimations.
	 */
	FrameTimeChecker(const double stdDevMult = 1.5, unsigned long long counterThreshold = 50);

	/**
	 * @brief Compares two frame times to check if they are close enough to constitute a stereo pair.
	 *
	 * What happens here, we create a mean value and standard deviation
	 * for frames time difference (i.e. time1-time2). If the difference
	 * is greater than configured value, the check fails and it is not safe
	 * to use both images as a stereo pair as they may be out of sync.
	 *
	 *  @param time1 the time of the first camera image.
	 *  @param time2 the time of the second camera image.
	 *  @return true if both images can be used as a stereo pair.
	 */
	bool checkTimes(const double time1, const double time2);

private:
	/** Multiplier for standard deviation. */
	double mStdDevMult;
	/** Mean value of time differences. */
    double mMeanValue;
    /** Squared mean value of time differences. */
    double mMeanValueSq;
    /** Estimated standard deviation of time differences. */
    double mStd;
    /** Frame counter. The first @p mCountThres will be always OK. */
    unsigned long long mCounter;
    /** The threshold below which all frames are assumed OK. */
    unsigned long long mCountThres;
};
