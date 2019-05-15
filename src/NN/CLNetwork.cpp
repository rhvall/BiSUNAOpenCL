//
//  CLNetwork.cpp
//  BiSUNAOpenCL
//
//  Created by RHVT on 14/5/19.
//  Copyright © 2019 R. All rights reserved.
//

#include <stdio.h>
#include "CLNetwork.hpp"
#include "CLNetworkKernel.cl.h"
#include "NNStructures.hpp"
#include "OpenCLUtils.hpp"

#define COUNT 1024

// Use this function to create a pointer that is going to be executed on
// a OpenCL dispatch queue and make the necessary translations of
// void pointers to the neccessary types required on the kernel
void agentKernelFunction(const cl_ndrange *range, void *input, void *output)
{
    agent_kernel(range, (CLNeuron *)input, (CLNeuron *)output);
}

static void clNetworkTest(const dispatch_queue_t dq)
{
    unsigned int i;
    
    size_t numSize = sizeof(CLNeuron) * COUNT;
    
    CLNeuron *host_input = (CLNeuron *)malloc(numSize);
    
    // ... and fill it with some initial data.
    for (i = 0; i < COUNT; i++) {
        host_input[i].nID = (cl_ushort)i;
        host_input[i].firing_rate = (cl_ushort)i;
        host_input[i].type = (cl_ushort)i;
        host_input[i].interface_index = (cl_ushort)i;
    }
    
    // That's it -- we're ready to send the work to OpenCL.
    // Note that this will execute asynchronously with respect
    // to the host application.
    openCLKernelFP (*f) = agentKernelFunction;
    OpenCLUtils::executeOpenCLKernels(dq, numSize, COUNT, host_input, f);
    
    for (i = 0; i < COUNT; i++)
    {
        printf("#: %i\tID: %i\t", i, host_input[i].nID);
        printf("F: %i\tT: %i\tI: %i\n", host_input[i].firing_rate, host_input[i].type, host_input[i].interface_index);
//        cl_float truth = (cl_float)i * (cl_float)i;
//        if (host_input[i] != truth)
//        {
//            fprintf(stdout, "Incorrect result @ index %d: Saw %1.4f, expected %1.4f\n\n",
//                    i, host_input[i], truth);
//            results_ok = 0;
//            break;
//        }
    }
    
    // Clean up host-side memory allocations:
    free(host_input);
}

int mainCLNetwork(int argc, const char * argv[])
{
    // Grab a CPU-based dispatch queue.
    dispatch_queue_t dq = OpenCLUtils::prepareOpenCLQueue();
    if (!dq)
    {
        fprintf(stdout, "Unable to create a CPU-based dispatch queue.\n");
        exit(1);
    }
    
    // Display the OpenCL device associated with this dispatch queue.
    clNetworkTest(dq);
    
    fprintf(stdout, "\nDone.\n\n");
    dispatch_release(dq);
    return 0;
}
