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

#pragma once

#include <GenericTalker.h>
#include "CameraConfig.h"
#include "CameraData.h"

template<typename... Args>
class ICameraTalker : public GenericTalker<CameraData>
{
public:
    /**
     * Virtual destructor for polymorphic behaviour.
     */
    virtual ~ICameraTalker() = default;

    // Enforces that all Args are of the CameraConfig type
    static_assert((std::is_same_v<Args, CameraConfig> && ...), "All arguments must be of CameraConfig type");

    /**
     * Starts the camera(s).
     *  @param camConfig a camera configuration parameters, one per camera.
     *	@return true if all cameras have started correctly.
     */
    virtual bool startCamera(const Args&... camConfig) = 0;

    /**
     * Stops the camera.
     */
    virtual void stopCamera() = 0;

    /**
     *  @return true if the camera has been initialised.
     */
    virtual bool isInitialised() const = 0;

    /**
     *  @return true if the camera was started and is working. 
     */
    virtual bool isRunning() const = 0;
};
