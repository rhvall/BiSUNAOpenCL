//
//  CLNetworkKernel.c
//  BiSUNAOpenCL
//
//  Created by RHVT on 14/5/19.
//  Copyright © 2019 R. All rights reserved.
//

#import "CLNetworkKernel.h"

//CLNeuron environment(size_t i, CLNeuron loc)
//{
//    CLNeuron output;
//    output.nID = loc.nID;
//    output.firing_rate = loc.firing_rate;
//    output.type = loc.type;
//    output.interface_index = loc.type * 2;
//    return output;
//}


kernel void agent(global CLNeuron *input,
                 global CLNeuron *output)
{
    size_t i = get_global_id(0);
    size_t j = get_local_id(0);
    size_t k = get_group_id(0);
    
    CLNeuron loc;
    loc.nID = input[i].nID;
    loc.firing_rate = i;
    loc.type = j;
    loc.interface_index = k;
    
    output[i].nID = loc.nID;
    output[i].firing_rate = i;
    output[i].type = j;
    output[i].interface_index = k;
    
//    CLNeuron update = environment(i, loc);
//    (*output).nID = update.nID;
//    (*output).firing_rate = update.firing_rate;
//    (*output).type = update.type;
//    (*output).interface_index = update.interface_index;
}
