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

#include <cmath>
#include "FrameTimeChecker.h"

FrameTimeChecker::FrameTimeChecker(const double stdDevMult, unsigned long long counterThreshold)
: mStdDevMult(stdDevMult), mMeanValue(0.0), mMeanValueSq(0.0), mStd(1000.0), mCounter(0), mCountThres(counterThreshold)
{
}

bool FrameTimeChecker::checkTimes(const double time1, const double time2)
{
    bool retVal = (++mCounter <= mCountThres || abs(mMeanValue - (time1 - time2)) <= mStdDevMult * mStd);
    if (retVal)
    {
        mMeanValue = 0.99 * mMeanValue + 0.01 * (time1 - time2);
        mMeanValueSq = 0.99 * mMeanValueSq + 0.01 * (time1 - time2) * (time1 - time2);
        mStd = sqrt(mMeanValueSq - mMeanValue*mMeanValue);
    }
    return retVal;
}
