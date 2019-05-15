//
//  OpenCLUtils.cpp
//  BiSUNAOpenCL
//
//  Created by RHVT on 15/5/19.
//  Copyright © 2019 R. All rights reserved.
//

#include "OpenCLUtils.hpp"

void OpenCLUtils::displayCLDevice(cl_device_id device)
{
    char name[128];
    char version[128];
    int maxGroup = 0;
    clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(char) * 128, name, NULL);
    clGetDeviceInfo(device, CL_DEVICE_VERSION, sizeof(char) * 128, version, NULL);
    clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &maxGroup, NULL);
    fprintf(stdout, "%s device: %s; Max Group: %i\n", version, name, maxGroup);
}

dispatch_queue_t OpenCLUtils::prepareOpenCLQueue()
{
    // Ask for the global OpenCL context:
    // Note: If you will not be enqueing to a specific device, you do not need
    // to retrieve the context.
    cl_context ctx = gcl_get_context();
    
    // Query this context to see what kinds of devices are available.
    size_t length;
    cl_device_id devices[8];
    clGetContextInfo(ctx, CL_CONTEXT_DEVICES, sizeof(devices), devices, &length);
    
    // Walk over these devices, printing out some basic information.  You could
    // query any of the information available about the device here.
    fprintf(stdout, "The following devices are available for use:\n");
    int num_devices = (int)(length / sizeof(cl_device_id));
    for (int i = 0; i < num_devices; i++) {
        displayCLDevice(devices[i]);
    }
    
    dispatch_queue_t dq = gcl_create_dispatch_queue(CL_DEVICE_TYPE_GPU, NULL);
    // If it is necessary to get a specific device, uncomment the line below:
//    dispatch_queue_t dq = gcl_create_dispatch_queue(CL_DEVICE_TYPE_USE_ID, devices[1]);
    
    if (dq == NULL) {
        dq = gcl_create_dispatch_queue(CL_DEVICE_TYPE_CPU, NULL);
    }
    
    if (!dq)
    {
        fprintf(stdout, "Unable to create a dispatch queue.\n");
        exit(1);
    }
    
    printf("\nContext for: ");
    auto dID = gcl_get_device_id_with_dispatch_queue(dq);
    displayCLDevice(dID);
    
    return dq;
}

void OpenCLUtils::executeOpenCLKernels(dispatch_queue_t dq, const size_t hostSize, const size_t workItems, void *hostValues, openCLKernelFP f)
{
    // We'll use a semaphore to synchronize the host and OpenCL device.
    dispatch_semaphore_t dsema = dispatch_semaphore_create(0);
    
    // Let's use OpenCL to square this array of floats.
    // First, allocate some memory on our OpenCL device to hold the input.
    // We *could* write the output to the same buffer in this case,
    //  but let's use a separate buffer.
    // Memory allocation 1: Create a buffer big enough to hold the input.
    // Notice that we use the flag 'CL_MEM_COPY_HOST_PTR' and pass the
    // host-side input data.  This instructs OpenCL to initialize the
    // device-side memory region with the supplied host data.
    void *device_input = gcl_malloc(hostSize, hostValues, CL_MEM_COPY_HOST_PTR);
    
    // Memory allocation 2: Create a buffer to store the results
    // of our kernel computation.
    void *device_results = gcl_malloc(hostSize, NULL, 0);
    
    // That's it -- we're ready to send the work to OpenCL.
    // Note that this will execute asynchronously with respect
    // to the host application.
    dispatch_async(dq, ^{
        cl_ndrange range = {
            1,          // We're using a 1-dimensional execution.
            {0},        // Start at the beginning of the range.
            {workItems},// Execute the number of work items.
            {0}         // Let OpenCL decide how to divide work items into workgroups.
        };
        
//        agent_kernel(&range, (CLNeuron *)device_input, (CLNeuron *)device_results);
        f(&range, device_input, device_results);
        
        // The computation is done at this point,
        // but the results are still "on" the device.
        // If we want to examine the results on the host,
        // we need to copy them back to the host's memory space.
        // Let's reuse the host-side input buffer.
        gcl_memcpy(hostValues, device_results, hostSize);
        
        // Okay -- signal the dispatch semaphore so the host knows
        // it can continue.
        dispatch_semaphore_signal(dsema);
    });
    
    // Here the host could do other, unrelated work while the OpenCL
    // device works on the kernel-based computation...
    // But now we wait for OpenCL to finish up.
    dispatch_semaphore_wait(dsema, DISPATCH_TIME_FOREVER);
    
    // Clean up device-side memory allocations:
    gcl_free(device_input);
}
