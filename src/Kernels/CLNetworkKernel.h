//
//  CLNetwork.h
//  BiSUNAOpenCL
//
//  Created by RHVT on 15/5/19.
//  Copyright © 2019 R. All rights reserved.
//

#ifdef CONTINUOS
typedef float ParameterType;
#else
typedef ushort ParameterType;
#endif

typedef struct CLNeuron {
    ushort nID;
    ushort firing_rate;
    ushort type;
    ushort interface_index;    //only used by input and output neurons to find the respective variable inside the input/output vector
} CLNeuron;

typedef struct CLConnection {
    ushort fromNID;
    ushort toNID;
    ushort neuroMod;    //-1 for inactive, for >0 it is active and represents the id of the neuron whose response is used as weight
    ParameterType weight;
} CLConnectionFlt;
