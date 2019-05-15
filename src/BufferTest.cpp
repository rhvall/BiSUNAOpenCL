//
//  BufferTest.cpp
//  BiSUNAOpenCL
//
//  Created by RHVT on 12/5/19.
//  Copyright © 2019 R. All rights reserved.
//
#include "BufferTest.hpp"
#include <stdio.h>
#include <stdlib.h>
#include <OpenCL/opencl.h>

// Include the automatically-generated header which provides the
// kernel block declaration.
#include "SquareFunction.cl.h"

#define COUNT 2048

static void display_device(cl_device_id device)
{
    char name_buf[128];
    char vendor_buf[128];
    clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(char) * 128, name_buf, NULL);
    clGetDeviceInfo(device, CL_DEVICE_VENDOR, sizeof(char) * 128, vendor_buf, NULL);
    fprintf(stdout, "Using OpenCL device: %s %s\n", vendor_buf, name_buf);
}

static void buffer_test(const dispatch_queue_t dq)
{
    unsigned int i;
    // We'll use a semaphore to synchronize the host and OpenCL device.
    dispatch_semaphore_t dsema = dispatch_semaphore_create(0);
    // Create some input data on the _host_ ...
    cl_float *host_input = (float *)malloc(sizeof(cl_float) * COUNT);
    // ... and fill it with some initial data.
    for (i = 0; i < COUNT; i++) {
        host_input[i] = (cl_float)i;
    }
    // Let's use OpenCL to square this array of floats.
    // First, allocate some memory on our OpenCL device to hold the input.
    // We *could* write the output to the same buffer in this case,
    //  but let's use a separate buffer.
    // Memory allocation 1: Create a buffer big enough to hold the input.
    // Notice that we use the flag 'CL_MEM_COPY_HOST_PTR' and pass the
    // host-side input data.  This instructs OpenCL to initialize the
    // device-side memory region with the supplied host data.
    void *device_input = gcl_malloc(sizeof(cl_float) * COUNT, host_input, CL_MEM_COPY_HOST_PTR);
    
    // Memory allocation 2: Create a buffer to store the results
    // of our kernel computation.
    void *device_results = gcl_malloc(sizeof(cl_float) * COUNT, NULL, 0);
    // That's it -- we're ready to send the work to OpenCL.
    // Note that this will execute asynchronously with respect
    // to the host application.
    dispatch_async(dq, ^{
        cl_ndrange range = {
                    1,          // We're using a 1-dimensional execution.
                    {0},        // Start at the beginning of the range.
                    {COUNT},    // Execute 'COUNT' work items.
                    {0}         // Let OpenCL decide how to divide work items
                    // into workgroups.
                };
        
        square_kernel(&range, (cl_float *)device_input, (cl_float *)device_results);
        
        // The computation is done at this point,
        // but the results are still "on" the device.
        // If we want to examine the results on the host,
        // we need to copy them back to the host's memory space.
        // Let's reuse the host-side input buffer.
        gcl_memcpy(host_input, device_results, COUNT * sizeof(cl_float));
        
        // Okay -- signal the dispatch semaphore so the host knows
        // it can continue.
        dispatch_semaphore_signal(dsema);
    });
    
    // Here the host could do other, unrelated work while the OpenCL
    // device works on the kernel-based computation...
    // But now we wait for OpenCL to finish up.
    dispatch_semaphore_wait(dsema, DISPATCH_TIME_FOREVER);
    
    // Test our results:
    int results_ok = 1;
    for (i = 0; i < COUNT; i++)
    {
//        printf("#: %i\tOpr: %f\n", i, host_input[i]);
        
        cl_float truth = (cl_float)i * (cl_float)i;
        if (host_input[i] != truth)
        {
            fprintf(stdout, "Incorrect result @ index %d: Saw %1.4f, expected %1.4f\n\n",
                   i, host_input[i], truth);
            results_ok = 0;
            break;
        }
    }
    
    if (results_ok)
    {
        fprintf(stdout, "Buffer results OK!\n");
    }
    
    // Clean up device-side memory allocations:
    gcl_free(device_input);
    // Clean up host-side memory allocations:
    free(host_input);
}

int mainBufferTest(int argc, const char * argv[])
{
    // Grab a CPU-based dispatch queue.
    dispatch_queue_t dq = gcl_create_dispatch_queue(CL_DEVICE_TYPE_CPU, NULL);
    if (!dq)
    {
        fprintf(stdout, "Unable to create a CPU-based dispatch queue.\n");
        exit(1);
    }
    
    // Display the OpenCL device associated with this dispatch queue.
    display_device(gcl_get_device_id_with_dispatch_queue(dq));
    buffer_test(dq);
    fprintf(stdout, "\nDone.\n\n");
    dispatch_release(dq);
    return 0;
}
