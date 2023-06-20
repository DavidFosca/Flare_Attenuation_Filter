#pragma once

#include <ap_fixed.h>

#define model_size_input (256)
#define model_size_output (256)
#define model_depth_input (3)
#define model_depth_output (3)

typedef ap_fixed<18, 8> model_type_input;
typedef ap_fixed<18, 8> model_type_output;
typedef ap_fixed<18, 8> model_type_weights;

void FlareNet(model_type_input input_image[196608], model_type_output output_image[196608]);


