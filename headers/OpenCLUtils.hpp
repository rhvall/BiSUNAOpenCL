//
//  OpenCLUtils.hpp
//  BiSUNAOpenCL
//
//  Created by RHVT on 15/5/19.
//  Copyright © 2019 R. All rights reserved.
//

#ifndef OpenCLUtils_hpp
#define OpenCLUtils_hpp

#include <stdio.h>
#include <OpenCL/opencl.h>

typedef void (openCLKernelFP)(const cl_ndrange *, void *, void *);

struct OpenCLUtils {
    // Show information about the device OpenCL is going to execute
    static void displayCLDevice(cl_device_id device);
    
    // Will create a new dispatch queue to be used within a OpenCL context,
    // but make sure that a call to "dispatch_release" is executed when this
    // object is used no more
    static dispatch_queue_t prepareOpenCLQueue();

    /**
     This function is an abstraction to the calls made to OpenCL, it creates a queue, copies
     values and executes the funtion passed as a kernel value, which will run inside a GCL queue.
     The list of parameters are as follows:
     @param dq The OpenCL dispatch queue to execute this function
     @param hostSize Integer that signals how big the hostValues array is. Ex. 'sizeof(int) * 30'
     would signal an array of 30 int elements
     @param workItems Tell OpenCL how many work items are going to be sent to the device
     @param hostValues Pointer to the memory section that contains the values and later is
     again to return the results of the computation
     @param f openCLKernelCall function pointer to be executed inside a GCL queue and will run the OCL kernel
     */
    static void executeOpenCLKernels(dispatch_queue_t dq, const size_t hostSize, const size_t workItems, void *hostValues, openCLKernelFP f);
};

#endif /* OpenCLUtils_hpp */
