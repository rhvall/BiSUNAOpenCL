//
//  my_kernels_1.cpp
//  TestOpenCL
//
//  Created by RHVT on 6/5/19.
//  Copyright © 2019 R. All rights reserved.
//

kernel void square(global float* input,               //   3
                  global float* output)
{
    size_t i = get_global_id(0);
    output[i] = input[i] * input[i];
}
