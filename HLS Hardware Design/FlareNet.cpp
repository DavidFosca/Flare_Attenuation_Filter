#include <hls_stream.h>
#include <stdio.h>
#include "FlareNet.h"
#include "weights.h"

// ############# Max-Pooling Buffer Initialize Function ############# //
template <int kernel_size, int input_size, int depth>
void init_buffer_and_window_maxpool(hls::stream<model_type_input>& input_stream, model_type_input input_buffer[kernel_size-1][input_size-kernel_size][depth], model_type_input window[kernel_size][kernel_size][depth]) {

	model_type_input pixel_val = 0;
	for (int x = 0; x < kernel_size-1; x++) {
		for (int y = 0; y < input_size; y++) {
			for (int chn = 0; chn < depth; chn++) {
				#pragma HLS PIPELINE
				input_stream >> pixel_val;
				//Store the next values outside the window kernel size (inside the window the values will be shifted left).
				if (y > kernel_size-1 and x < kernel_size-1){
					input_buffer[x][y-kernel_size][chn] = pixel_val;
				}
				//Initialize window
				if (y < kernel_size) {
					window[x][y][chn] = pixel_val;
				}
			}
		}
	}
	//Initialize lower side of window.
	for (int y = 0; y < kernel_size; y++) {
		for (int chn = 0; chn < depth; chn++) {
			#pragma HLS PIPELINE
			input_stream >> pixel_val;
			window[kernel_size-1][y][chn] = pixel_val;
		}
	}
}

// ############# Zero-padding Buffer Initialize Function ############# //
template <int kernel_size, int input_size, int depth>
void init_buffer_and_window_zeropadd(hls::stream<model_type_input>& input_stream, model_type_input input_buffer[kernel_size][input_size-kernel_size+2][depth], model_type_input window[kernel_size][kernel_size][depth]) {

	model_type_input pixel_val = 0;
	//Fill values along column for the kernel size.
	for (int x = 0; x < kernel_size; x++) {
		//Fill values along the kernel width size (+1 because of the padding)
		for (int y = 0; y < input_size+1; y++) {
			//Fill all depth channel values.
			for (int chn = 0; chn < depth; chn++) {
				#pragma HLS PIPELINE
				//If first row or first column, then pixel is 0 (zero padding).
				if (x < 1) {
					pixel_val = 0;
				}
				else if (y < 1) {
					pixel_val = 0;
				}
				else {
					input_stream >> pixel_val;
				}
				//Store the next values outside the window kernel size (inside the window the values will be shifted left).
				if (y > kernel_size-1) {
					if (y == input_size) {
						//Fill zero padding on last column.
						input_buffer[x][y-kernel_size+1][chn] = 0;
					}
					input_buffer[x][y-kernel_size][chn] = pixel_val;
				}
				//Initialize window
				if (y < kernel_size) {
					window[x][y][chn] = pixel_val;
				}
			}
		}
	}
}

// ############# Max-pooling Buffer Update Function ############# //
template <int kernel_size, int input_size, int depth>
void update_buffer_and_window_maxpool(hls::stream<model_type_input>& input_stream, model_type_input input_buffer[kernel_size-1][input_size-kernel_size][depth], model_type_input window[kernel_size][kernel_size][depth]) {

	model_type_input pixel_val = 0;
	//Update window by shifting columns left.
	for (int x = 0; x < kernel_size; x++) {
		for (int y = 0; y < kernel_size; y++) {
			for (int chn = 0; chn < depth; chn++) {
				#pragma HLS PIPELINE
				//First column of window update.
				if (y < kernel_size-1) {
					//Shift window kernel values to the left.
					window[x][y][chn] = window[x][y+1][chn];
				}
				if (y == kernel_size-1) {
					//Upper right corner of window update.
					if (x < kernel_size-1) {
						window[x][y][chn] = input_buffer[0][0][chn];
					}
					//Lower right corner of window update.
					else {
						input_stream >> pixel_val;
						window[x][y][chn] = pixel_val;
					}
				}
			}
		}
	}

	//Update buffer by shifting the buffer to the left and inputing new column value.
	for (int x = 0; x < kernel_size-1; x++) {
		for (int y = 0; y < input_size-kernel_size; y++) {
			for (int chn = 0; chn < depth; chn++) {
				#pragma HLS PIPELINE
				//Shift all row buffer values to the left.
				if (y < input_size-kernel_size-1) {
					input_buffer[x][y][chn] = input_buffer[x][y+1][chn];
				}
			}
		}
	}
}

// ############# Zero-padding Buffer Update Function ############# //
template <int kernel_size, int input_size, int depth>
void update_buffer_and_window_zeropadd(hls::stream<model_type_input>& input_stream, model_type_input input_buffer[kernel_size][input_size-kernel_size+2][depth], model_type_input window[kernel_size][kernel_size][depth], int last_row, int last_col, int first_col, int max_pool_flag) {

	model_type_input temporal[kernel_size-1][depth];

	//Update window by shifting columns left.
	for (int x = 0; x < kernel_size; x++) {
		for (int y = 0; y < kernel_size; y++) {
			for (int chn = 0; chn < depth; chn++) {
				#pragma HLS PIPELINE
				if ((y == 0) and (x > 0) and (max_pool_flag == 0)) {
					//Save first column and last two rows.
					temporal[x-1][chn] = window[x][y][chn];
				}
				if (y < kernel_size-1) {
					//Shift window kernel values to the left.
					window[x][y][chn] = window[x][y+1][chn];
				}
				else {
					//Update last window kernel column with first column value from Buffer.
					window[x][y][chn] = input_buffer[x][0][chn];
				}
			}
		}
	}

	model_type_input pixel_val = 0;
	//Update buffer by shifting the buffer to the left and inputing new column value.
	for (int x = 0; x < kernel_size; x++) {
		for (int y = 0; y < input_size-kernel_size+2; y++) {
			for (int chn = 0; chn < depth; chn++) {
				#pragma HLS PIPELINE
				//Shift all row buffer values to the left.
				if (y < input_size-kernel_size+1) {
					input_buffer[x][y][chn] = input_buffer[x][y+1][chn];
				}
				//If in last buffer column, update column values.
				else {
					if (max_pool_flag == 0) {
						//For all row values except last one (2 out of 3 for the 3x3 kernel for example - upper corner), insert temporal values.
						if (x < kernel_size-1) {
							input_buffer[x][y][chn] = temporal[x][chn]; //x=0
						}
						//For the last column and last row, update with new value from stream.
						else {
							if (last_row != 1) {
								//If not in last row of input tensor.
								if (last_col == 1 or first_col == 1) {
									//If in last column or first column insert zeros.
									pixel_val = 0;
								}
								else {
									input_stream >> pixel_val;
								}
								input_buffer[x][y][chn] = pixel_val;
							}
							//In last row of tensor, input zeros.
							else {
								input_buffer[x][y][chn] = 0;
							}
						}
					}
				}
			}
		}
	}
}


// ############# Transpose Buffer Initialize Function ############# //
template <int kernel_size, int output_size, int output_depth>
void init_tranpose_buffer(model_type_input tran_buff[kernel_size][output_size+1][output_depth], const model_type_weights bias[output_depth] ) {

	//Initialize transpose buffer with bias filter values.
	for (int tran_x = 0; tran_x < kernel_size; tran_x++) {
		for (int tran_y = 0; tran_y < output_size+1; tran_y++) {
			for (int tran_f = 0; tran_f < output_depth; tran_f++) {
				#pragma HLS PIPELINE
				tran_buff[tran_x][tran_y][tran_f] = bias[tran_f];
			}
		}
	}
}

// ############# 1D Buffer Initialize Function ############# //
template <int depth>
void init_1D_window(hls::stream<model_type_input>& input_stream, model_type_input window[depth]) {

	//Fill pixel channel values.
	model_type_input pixel_val = 0;
	for (int chn = 0; chn < depth; chn++) {
		#pragma HLS PIPELINE
		input_stream >> pixel_val;
		window[chn] = pixel_val;
	}
}

//############# 2D Convolutional Layer - RELU #############//
template <int input_size, int kernel_size, int input_depth, int output_depth>
void Conv2D_relu(hls::stream<model_type_input>& input_stream, hls::stream<model_type_output>& output_stream, const model_type_weights weight_filt[kernel_size][kernel_size][input_depth][output_depth], const model_type_weights bias[output_depth]) {

	model_type_input input_buffer[kernel_size][input_size-kernel_size+2][input_depth];
	model_type_input window[kernel_size][kernel_size][input_depth];
	model_type_input window_conv_result = 0;
	int last_row = 0;
	int last_col = 0;
	int first_col = 0;
	int max_pool_flag = 0;

	init_buffer_and_window_zeropadd<kernel_size, input_size, input_depth>(input_stream, input_buffer, window);

	//Iterate through all rows of input tensor.
	for (int x = 0; x < input_size; x++) {
		//Iterate through all columns of input tensor.
		for (int y = 0; y < input_size; y++) {
			//Convolution for every output filter and window.
			for (int filter = 0; filter < output_depth; filter++) {
				#pragma HLS PIPELINE
				//Convolution between window and respective output depth filter.
				window_conv_result = 0;
				for (int win_chn = 0; win_chn < input_depth; win_chn++) {
					for (int win_x = 0; win_x < kernel_size; win_x++) {
						//#pragma HLS PIPELINE //does not generate latency improvements.
						for (int win_y = 0; win_y < kernel_size; win_y++) {
							//#pragma HLS PIPELINE //does not generate latency improvements.
							window_conv_result += weight_filt[win_x][win_y][win_chn][filter] * window[win_x][win_y][win_chn];
						}
					}
				}

				//Add respective bias.
				window_conv_result += bias[filter];

				//Apply ReLU activation function.
				if (window_conv_result < 0) {
					window_conv_result = 0;
				}

				//Write into sequential output_stream.
				output_stream << window_conv_result;

			}
			//Evaluate if border of input tensor.
			if (x == (input_size-kernel_size+1) or x == (input_size-kernel_size+2)) {
				last_row = 1;
			}
			else {
				last_row = 0;
			}
			if (y == 0) {
				first_col = 1;
			}
			else {
				first_col = 0;
			}
			update_buffer_and_window_zeropadd<kernel_size, input_size, input_depth>(input_stream, input_buffer, window, last_row, last_col, first_col, max_pool_flag);
			last_col = 0;
		}
		//In order to shift twice when in border section.
		for (int sec=0; sec<kernel_size-1; sec++){
			if (sec == (kernel_size-2)) {
				last_col = 1;
			}
			else {
				last_col = 0;
			}
			update_buffer_and_window_zeropadd<kernel_size, input_size, input_depth>(input_stream, input_buffer, window, last_row, last_col, first_col, max_pool_flag);
			last_col = 0;
		}
	}
}

//############# 2DConvolutional Layer - RELU #############//
template <int input_size, int kernel_size, int input_depth, int output_depth>
void Conv2D_relu_2streams(hls::stream<model_type_input>& input_stream, hls::stream<model_type_output>& output_stream, hls::stream<model_type_output>& output_stream_2, const model_type_weights weight_filt[kernel_size][kernel_size][input_depth][output_depth], const model_type_weights bias[output_depth]) {

	model_type_input input_buffer[kernel_size][input_size-kernel_size+2][input_depth];
	model_type_input window[kernel_size][kernel_size][input_depth];
	model_type_input window_conv_result = 0;
	int last_row = 0;
	int last_col = 0;
	int first_col = 0;
	int max_pool_flag = 0;

	init_buffer_and_window_zeropadd<kernel_size, input_size, input_depth>(input_stream, input_buffer, window);

	//Iterate through all rows of input tensor.
	for (int x = 0; x < input_size; x++) {
		//Iterate through all columns of input tensor.
		for (int y = 0; y < input_size; y++) {
			//Convolution for every output filter and window.
			for (int filter = 0; filter < output_depth; filter++) {
				#pragma HLS PIPELINE //to produce one output value per clock cycle.
				//Convolution between window and respective output depth filter.
				window_conv_result = 0;
				for (int win_chn = 0; win_chn < input_depth; win_chn++) {
					for (int win_x = 0; win_x < kernel_size; win_x++) {
						//#pragma HLS PIPELINE //does not generate latency improvements.
						for (int win_y = 0; win_y < kernel_size; win_y++) {
							#pragma HLS PIPELINE //does not generate latency improvements.
							window_conv_result += weight_filt[win_x][win_y][win_chn][filter] * window[win_x][win_y][win_chn];
						}
					}
				}
				//Add respective bias.
				window_conv_result += bias[filter];

				//Apply ReLU activation function.
				if (window_conv_result < 0) {
					window_conv_result = 0;
				}

				//Write into sequential output_stream.
				output_stream << window_conv_result;
				//Write into skip-connection output_stream.
				output_stream_2 << window_conv_result;

			}
			//Evaluate if border of input tensor.
			if (x == (input_size-kernel_size+1) or x == (input_size-kernel_size+2)) {
				last_row = 1;
			}
			else {
				last_row = 0;
			}
			if (y == 0) {
				first_col = 1;
			}
			else {
				first_col = 0;
			}
			update_buffer_and_window_zeropadd<kernel_size, input_size, input_depth>(input_stream, input_buffer, window, last_row, last_col, first_col, max_pool_flag);
			last_col = 0;
		}
		//In order to shift twice when in border section.
		for (int sec=0; sec<kernel_size-1; sec++){
			if (sec == (kernel_size-2)) {
				last_col = 1;
			}
			else {
				last_col = 0;
			}
			update_buffer_and_window_zeropadd<kernel_size, input_size, input_depth>(input_stream, input_buffer, window, last_row, last_col, first_col, max_pool_flag);
			last_col = 0;
		}
	}
}

//############# 2DConvolutional Layer - SIGMOID #############//
template <int input_size, int kernel_size, int input_depth, int output_depth>
void Conv2D_sigmoid(hls::stream<model_type_input>& input_stream, hls::stream<model_type_output>& output_stream, const model_type_weights weight_filt[kernel_size][kernel_size][input_depth][output_depth], const model_type_weights bias[output_depth]) {

	model_type_input window[input_depth];
	model_type_input window_conv_result = 0;

	init_1D_window<input_depth>(input_stream, window);

	//Iterate through input tensor.
	for (int x = 0; x < (input_size * input_size); x++) {
		//Convolution for every output filter and window.
		for (int filter = 0; filter < output_depth; filter++) {
			#pragma HLS PIPELINE //to produce one output value per clock cycle.
			//Convolution between window and respective output depth filter.
			window_conv_result = 0;
			for (int win_chn = 0; win_chn < input_depth; win_chn++) {
				for (int win_x = 0; win_x < kernel_size; win_x++) {
					for (int win_y = 0; win_y < kernel_size; win_y++) {
						#pragma HLS PIPELINE //does not generate latency improvements.
						window_conv_result += weight_filt[win_x][win_y][win_chn][filter] * window[win_chn];
					}
				}
			}

			//Add respective bias.
			window_conv_result += bias[filter];

			//window_conv_result = 1 / (1 + exp(-window_conv_result));
			output_stream << window_conv_result;

		}
		if (x != input_size * input_size - 1) {
			init_1D_window<input_depth>(input_stream, window);
		}
	}
}

// ############# Depthwise Separable 2DConvolutional Layer ############# //
template <int input_size, int kernel_size, int input_depth, int output_depth>
void SeparableDW2D_relu(hls::stream<model_type_input>& input_stream, hls::stream<model_type_output>& output_stream, const model_type_weights weight_depth_filt[kernel_size][kernel_size][input_depth], const model_type_weights weight_point_filt[input_depth][output_depth], const model_type_weights bias[output_depth]) {

	model_type_input input_buffer[kernel_size][input_size-kernel_size+2][input_depth];
	model_type_input window[kernel_size][kernel_size][input_depth];
	model_type_input depthwise_vector[input_depth];
	model_type_input depthwise_res = 0;
	model_type_input pointwise_res = 0;
	model_type_input last_row = 0;
	model_type_input last_col = 0;
	model_type_input first_col = 0;
	model_type_input max_pool_flag = 0;

	init_buffer_and_window_zeropadd<kernel_size, input_size, input_depth>(input_stream, input_buffer, window);

	//Iterate through all rows of input tensor.
	for (int x = 0; x < input_size; x++) {
		last_col = 0;
		//Iterate through all columns of input tensor.
		for (int y = 0; y < input_size; y++) {
			//Depth-wise convolution for every input depth filter and window.
			for (int win_chn = 0; win_chn < input_depth; win_chn++) {
				depthwise_res = 0;
				//#pragma HLS PIPELINE (does not improve performance, only increases resource utilization).
				for (int win_x = 0; win_x < kernel_size; win_x++) {
					//#pragma HLS PIPELINE (does not improve performance, only increases resource utilization).
					for (int win_y = 0; win_y < kernel_size; win_y++) {
					#pragma HLS PIPELINE
					depthwise_res += weight_depth_filt[win_x][win_y][win_chn] * window[win_x][win_y][win_chn];
					}
				}
				//Storing results in 1D vector for each input channel.
				depthwise_vector[win_chn] = depthwise_res;
			}
			//Point-wise convolution between depthwise_vector and respective output depth filter.
			for (int filter = 0; filter < output_depth; filter++) {
				pointwise_res = 0;
				//#pragma HLS PIPELINE //(does not improve performance, only increases resource utilization).
				for (int win_chn = 0; win_chn < input_depth; win_chn++) {
					#pragma HLS PIPELINE
					pointwise_res += weight_point_filt[win_chn][filter] * depthwise_vector[win_chn];
				}

				//Add respective bias.
				pointwise_res += bias[filter];

				//Apply ReLU activation function.
				if (pointwise_res < 0){
					pointwise_res = 0;
				}

				//Write into sequential output_stream.
				output_stream << pointwise_res;

			}
			//Evaluate if border of input tensor.
			if (x == (input_size-kernel_size+1) or x == (input_size-kernel_size+2)) {
				last_row = 1;
			}
			else {
				last_row = 0;
			}
			if (y == 0) {
				first_col = 1;
			}
			else {
				first_col = 0;
			}
			update_buffer_and_window_zeropadd<kernel_size, input_size, input_depth>(input_stream, input_buffer, window, last_row, last_col, first_col, max_pool_flag);
		}
		//In order to shift twice when in border section.
		for (int sec=0; sec<kernel_size-1; sec++){
			if (sec == (kernel_size-2)) {
				last_col = 1;
			}
			else {
				last_col = 0;
			}
			update_buffer_and_window_zeropadd<kernel_size, input_size, input_depth>(input_stream, input_buffer, window, last_row, last_col, first_col, max_pool_flag);
		}
	}
}

// ############# Depthwise Separable 2DConvolutional Layer ############# //
template <int input_size, int kernel_size, int input_depth, int output_depth>
void SeparableDW2D_relu_2streams(hls::stream<model_type_input>& input_stream, hls::stream<model_type_output>& output_stream, hls::stream<model_type_output>& output_stream_2, const model_type_weights weight_depth_filt[kernel_size][kernel_size][input_depth], const model_type_weights weight_point_filt[input_depth][output_depth], const model_type_weights bias[output_depth]) {

	model_type_input input_buffer[kernel_size][input_size-kernel_size+2][input_depth];
	model_type_input window[kernel_size][kernel_size][input_depth];
	model_type_input depthwise_vector[input_depth];
	model_type_input depthwise_res = 0;
	model_type_input pointwise_res = 0;
	model_type_input last_row = 0;
	model_type_input last_col = 0;
	model_type_input first_col = 0;
	model_type_input max_pool_flag = 0;

	init_buffer_and_window_zeropadd<kernel_size, input_size, input_depth>(input_stream, input_buffer, window);

	//Iterate through all rows of input tensor.
	for (int x = 0; x < input_size; x++) {
		last_col = 0;
		//Iterate through all columns of input tensor.
		for (int y = 0; y < input_size; y++) {
			//Depth-wise convolution for every input depth filter and window.
			for (int win_chn = 0; win_chn < input_depth; win_chn++) {
				depthwise_res = 0;
				//#pragma HLS PIPELINE (does not improve performance, only increases resource utilization).
				for (int win_x = 0; win_x < kernel_size; win_x++) {
					//#pragma HLS PIPELINE (does not improve performance, only increases resource utilization).
					for (int win_y = 0; win_y < kernel_size; win_y++) {
					#pragma HLS PIPELINE
					depthwise_res += weight_depth_filt[win_x][win_y][win_chn] * window[win_x][win_y][win_chn];
					}
				}
				//Storing results in 1D vector.
				depthwise_vector[win_chn] = depthwise_res;
			}
			//Point-wise convolution between depthwise_vector and respective output depth filter.
			for (int filter = 0; filter < output_depth; filter++) {
				//#pragma HLS PIPELINE //(does not improve performance, only increases resource utilization).
				pointwise_res = 0;
				for (int win_chn = 0; win_chn < input_depth; win_chn++) {
					#pragma HLS PIPELINE
					pointwise_res += weight_point_filt[win_chn][filter] * depthwise_vector[win_chn];
				}

				//Add respective bias.
				pointwise_res += bias[filter];

				//Apply ReLU activation function.
				if (pointwise_res < 0){
					pointwise_res = 0;
				}

				//Write into sequential output_stream.
				output_stream << pointwise_res;
				//Write into skip-connection output_stream.
				output_stream_2 << pointwise_res;

			}
			//Evaluate if border of input tensor.
			if (x == (input_size-kernel_size+1) or x == (input_size-kernel_size+2)) {
				last_row = 1;
			}
			else {
				last_row = 0;
			}
			if (y == 0) {
				first_col = 1;
			}
			else {
				first_col = 0;
			}
			update_buffer_and_window_zeropadd<kernel_size, input_size, input_depth>(input_stream, input_buffer, window, last_row, last_col, first_col, max_pool_flag);
		}
		//In order to shift twice when in border section.
		for (int sec=0; sec<kernel_size-1; sec++){
			if (sec == (kernel_size-2)) {
				last_col = 1;
			}
			else {
				last_col = 0;
			}
			update_buffer_and_window_zeropadd<kernel_size, input_size, input_depth>(input_stream, input_buffer, window, last_row, last_col, first_col, max_pool_flag);
		}
	}
}


// ############# Transposed 2D Convolutional Layer ############# //
template <int output_size, int kernel_size, int stride, int input_depth, int output_depth>
void Conv2D_transposed(hls::stream<model_type_input>& input_stream, hls::stream<model_type_output>& output_stream, const model_type_weights weight_filt[kernel_size][kernel_size][output_depth][input_depth], const model_type_weights bias[output_depth]) {

	model_type_input pixel_vec[input_depth];
	model_type_input tran_buff[kernel_size][output_size+1][output_depth];
	model_type_input conv_res = 0;

	init_tranpose_buffer<kernel_size, output_size, output_depth>(tran_buff, bias);

	//Iterate through all rows of input tensor.
	for (int x = 0; x < output_size; x += stride) {
		//Iterate through all columns of input tensor.
		for (int y = 0; y < output_size; y += stride) {
				//Read pixel channel values.
				for (int chn = 0; chn < input_depth; chn++) {
					#pragma HLS PIPELINE
					input_stream >> pixel_vec[chn];
			}
			//Convolution for every output filter and input pixel values.
			for (int filter = 0; filter < output_depth; filter++) {
				for (int win_x = 0; win_x < kernel_size; win_x++) {
					for (int win_y = 0; win_y < kernel_size; win_y++) {
						conv_res = 0;
						for (int win_chn = 0; win_chn < input_depth; win_chn++) {
							#pragma HLS PIPELINE
							conv_res += weight_filt[win_x][win_y][filter][win_chn] * pixel_vec[win_chn];
						}
						tran_buff[win_x][y+win_y][filter] += conv_res;
					}
				}
			}
		}

		for (int tran_x = 0; tran_x < kernel_size; tran_x++) {
			for (int tran_y = 0; tran_y < output_size; tran_y++) {
				for (int tran_f = 0; tran_f < output_depth; tran_f++) {
					#pragma HLS PIPELINE
					//If first row, then write to output stream and shift last kernel row to first row.
					if (tran_x == 0) {
						//Apply ReLU activation function.
						if (tran_buff[tran_x][tran_y][tran_f] < 0) {
							//Write into sequential output_stream.
							output_stream  << 0;
						}
						else {
							//Write into sequential output_stream.
							output_stream << tran_buff[tran_x][tran_y][tran_f];
						}
						tran_buff[tran_x][tran_y][tran_f] = tran_buff[tran_x+stride][tran_y][tran_f];
					}
					//If middle row, then write to output and change to bias value.
					else if (tran_x == 1) {
						//Apply ReLU activation function.
						if (tran_buff[tran_x][tran_y][tran_f] < 0) {
							output_stream  << 0;
						}
						else {
							output_stream << tran_buff[tran_x][tran_y][tran_f];
						}
						tran_buff[tran_x][tran_y][tran_f] = bias[tran_f];
					}
					//If last row, then change to bias value.
					else {
						tran_buff[tran_x][tran_y][tran_f] = bias[tran_f];
					}
				//Reset last col in buffer to avoid overfloating.
				//tran_buff[tran_x][tran_y+1][tran_f] = 0;
				}
			}
		}
	}
}

// ############# 2D Max Pooling Layer ############# //
template <int input_size, int pool_size, int depth>
void MaxPooling2D(hls::stream<model_type_input>& input_stream, hls::stream<model_type_output>& output_stream) {

	model_type_input input_buffer[pool_size-1][input_size-pool_size][depth];
	model_type_input window[pool_size][pool_size][depth];
	model_type_input maxpool_val = 0;
	int last_col = 0;

	init_buffer_and_window_maxpool<pool_size, input_size, depth>(input_stream, input_buffer, window);

	//Iterate through all rows of input tensor.
	for (int x = 0; x < (input_size/pool_size); x++) {
		//Iterate through all columns of input tensor.
		for (int y = 0; y < input_size-pool_size+1; y++) {
			if (y%pool_size == 0) { // The window has moved by one stride.
				//Calculate max value per channel.
				for (int win_chn = 0; win_chn < depth; win_chn++) {
					#pragma HLS PIPELINE
					maxpool_val = 0;
					for (int win_x = 0; win_x < pool_size; win_x++) {
						#pragma HLS PIPELINE
						for (int win_y = 0; win_y < pool_size; win_y++) {
								if (maxpool_val < window[win_x][win_y][win_chn] ) {
									maxpool_val = window[win_x][win_y][win_chn];
								}
							}
						}
					//Write into sequential output_stream.
					output_stream << maxpool_val;
				}
			}
			if (y != input_size-pool_size) {
				//Shift window and buffer, no new reads.
				update_buffer_and_window_maxpool<pool_size, input_size, depth>(input_stream, input_buffer, window);
			}
		}
		//Load next block of 2 rows.
		if (x != ((input_size/pool_size)-1)) {
			init_buffer_and_window_maxpool<pool_size, input_size, depth>(input_stream, input_buffer, window);
		}
	}
}

// ############# 2D Adding Layer ############# //
template <int input_size, int depth>
void Add(hls::stream<model_type_input>& input_stream_1, hls::stream<model_type_input>& input_stream_2, hls::stream<model_type_output>& output_stream) {

	model_type_input pixel_val_1 = 0;
	model_type_input pixel_val_2 = 0;
	model_type_input adding_result = 0;

	//Iterate through all rows of input tensor.
	for (int x = 0; x < input_size; x++) {
		//Iterate through all columns of input tensor.
		for (int y = 0; y < input_size; y++) {
			for (int win_chn = 0; win_chn < depth; win_chn++) {
				#pragma HLS PIPELINE
				input_stream_1 >> pixel_val_1;
				input_stream_2 >> pixel_val_2;
				adding_result = pixel_val_1 + pixel_val_2;
				//Apply ReLU activation function.
				if (adding_result < 0) {
					adding_result = 0;
				}

				//Write into sequential output_stream.
				output_stream << adding_result;

			}
		}
	}
}

// --------------   Test done to improve performance by optimizing zero-padd buffer functions: -------------- //
/*
template <int kernel_size, int input_size, int depth>
void init_buffer_and_window_maxpool(hls::stream<model_type_input>& input_stream, model_type_input input_buffer[kernel_size][input_size-kernel_size][depth], model_type_input window[kernel_size][kernel_size][depth]) {

	model_type_input pixel_val = 0;
	for (int x = 0; x < kernel_size; x++) {
		for (int y = 0; y < input_size; y++) {
			for (int chn = 0; chn < depth; chn++) {
				#pragma HLS PIPELINE
				 input_stream >> pixel_val;
				//Store the next values outside the window kernel size (inside the window the values will be shifted left).
				if (y > kernel_size-1){
					input_buffer[x][y-kernel_size][chn] = pixel_val;
				}
				//Initialize window
				if (y < kernel_size) {
					window[x][y][chn] = pixel_val;
				}
			}
		}
	}
}

template <int kernel_size, int input_size, int depth>
void update_buffer_and_window_maxpool(hls::stream<model_type_input>& input_stream, model_type_input input_buffer[kernel_size][input_size-kernel_size][depth], model_type_input window[kernel_size][kernel_size][depth]) {

	//Update Window by shifting columns left.
	for (int x = 0; x < kernel_size; x++) {
		for (int y = 0; y < kernel_size; y++) {
			for (int chn = 0; chn < depth; chn++) {
				#pragma HLS PIPELINE
				if (y < kernel_size-1) {
					//Shift window kernel values to the left.
					window[x][y][chn] = window[x][y+1][chn];
				}
				//Update last window kernel column with first column value from Buffer.
				else {
					window[x][y][chn] = input_buffer[x][0][chn];
				}
			}
		}
	}

	//Update Row Buffer by shifting the buffer to the left and inputing new column value.
	for (int x = 0; x < kernel_size; x++) {
		for (int y = 0; y < input_size-kernel_size; y++) {
			for (int chn = 0; chn < depth; chn++) {
				#pragma HLS PIPELINE
				//Shift all row buffer values to the left.
				if (y < input_size-kernel_size-1) {
					input_buffer[x][y][chn] = input_buffer[x][y+1][chn];
				}
			}
		}
	}
}

template <int kernel_size, int input_size, int depth>
void init_buffer_and_window_zeropadd(hls::stream<model_type_input>& input_stream, model_type_input input_buffer[kernel_size-1][input_size-kernel_size+2][depth], model_type_input window[kernel_size][kernel_size][depth]) {

	model_type_input pixel_val = 0;
	//Fill values along the column for the kernel size.
	for (int x = 0; x < kernel_size-1; x++) {
		//Fill values along the kernel width size (+1 because of the padding)
		for (int y = 0; y < input_size+1; y++) {
			//Fill all depth channel values.
			for (int chn = 0; chn < depth; chn++) {
				#pragma HLS PIPELINE
				//If first row or first column, then pixel is 0 (zero padding).
				if (x < 1) {
					pixel_val = 0;
				}
				else if (y < 1) {
					pixel_val = 0;
				}
				else {
					input_stream >> pixel_val;
				}
				//Store the next values outside the window kernel size (inside the window the values will be shifted left).
				if (y > kernel_size-1) {
					if (y == input_size) {
						//Fill zero padding on last column.
						input_buffer[x][y-kernel_size+1][chn] = 0;
					}
					input_buffer[x][y-kernel_size][chn] = pixel_val;
				}
				//Initialize window
				if (y < kernel_size) {
					window[x][y][chn] = pixel_val;
				}
			}
		}
	}
	for (int y = 0; y < kernel_size; y++) {
		for (int chn = 0; chn < depth; chn++) {
			#pragma HLS PIPELINE
			if (y < 1) {
				pixel_val = 0;
			}
			else {
				input_stream >> pixel_val;
			}
			window[kernel_size-1][y][chn] = pixel_val;
		}
	}
}

template <int kernel_size, int input_size, int depth>
void update_buffer_and_window_zeropadd(hls::stream<model_type_input>& input_stream, model_type_input input_buffer[kernel_size-1][input_size-kernel_size+2][depth], model_type_input window[kernel_size][kernel_size][depth], int last_row, int zero_last_col, int first_col, int first) {

	model_type_input temporal[kernel_size-1][depth];
	model_type_input pixel_val = 0;
	int limit = 0;

	//Update Window by shifting columns left.
	for (int x = 0; x < kernel_size; x++) {
		for (int y = 0; y < kernel_size; y++) {
			for (int chn = 0; chn < depth; chn++) {
				#pragma HLS PIPELINE
				if ((y == 0) and (x > 0)) {
					//Save first column and last two rows.
					temporal[x-1][chn] = window[x][y][chn];
				}
				if (y < kernel_size-1) {
					if (first == 0){
						limit = kernel_size-1;
					}
					else {
						limit = kernel_size-2
					}
				}
				if (y < limit) {
					//Shift window kernel values to the left.
					window[x][y][chn] = window[x][y+1][chn];
				}
				//Update last window kernel column with first column value from Buffer.
				else {
					//Update last column first two rows of window.
					if (x < kernel_size-1) {
						window[x][y][chn] = input_buffer[x][0][chn];
					}
					else if (first == 0) {
						//Update lower down corner with new value.
						if (last_row != 1) {
							if (zero_last_col == 1) {
								pixel_val = 0;
							}
							else {
								input_stream >> pixel_val;
							}
							window[x][y][chn] = pixel_val;
						}
						//In last row, append zeros.
						else {
							window[x][y][chn] = 0;
						}
					}
				}
			}
		}
	}

	//Update Row Buffer by shifting the buffer to the left and inputing new column value.
	for (int x = 0; x < kernel_size-1; x++) {
		for (int y = 0; y < input_size-kernel_size+2; y++) {
			for (int chn = 0; chn < depth; chn++) {
				#pragma HLS PIPELINE
				//Shift all row buffer values to the left.
				if (y < input_size-kernel_size+1) {
					input_buffer[x][y][chn] = input_buffer[x][y+1][chn];
				}
				//If in last buffer column, update column values.
				else {
					//Insert temporal values as circular buffer.
					input_buffer[x][y][chn] = temporal[x][chn];
				}
			}
		}
	}
}
*/

void FlareNet(model_type_input input_image[196608], model_type_output output_image[196608]) {

	//Input and Output Streams
	hls::stream<model_type_input> input_stream;
	hls::stream<model_type_output> output_stream;
	//Encoder Internal Streams
	hls::stream<model_type_input> stream_0;
	hls::stream<model_type_input> stream_skip_1;
	hls::stream<model_type_input> stream_1;
	hls::stream<model_type_input> stream_2;
	hls::stream<model_type_input> stream_3;
	hls::stream<model_type_input> stream_4;
	hls::stream<model_type_input> stream_skip_2;
	hls::stream<model_type_input> stream_5;
	hls::stream<model_type_input> stream_6;
	hls::stream<model_type_input> stream_7;
	//Decoder Internal Streams
	hls::stream<model_type_input> stream_8;
	hls::stream<model_type_input> stream_9;
	hls::stream<model_type_input> stream_10;
	hls::stream<model_type_input> stream_11;
	hls::stream<model_type_input> stream_12;
	hls::stream<model_type_input> stream_13;

	#pragma HLS DATAFLOW //enables task-level pipelining, allowing functions and loops to overlap in their operation,
	//increasing the concurrency of the RTL implementation and increasing the overall throughput of the design.

	//Read image vector from TB into the input stream.
	for (int x = 0; x < model_size_input * model_size_input * model_depth_input; x++) {
			#pragma HLS PIPELINE
			input_stream << input_image[x];
	}

	//Instantiate FlareNet-simple architecture.
	//Encoder Layers
	Conv2D_relu_2streams<256, 3, 3, 16>(input_stream, stream_0, stream_skip_1, conv2d_weights_0, conv2d_bias_0);
	MaxPooling2D<256, 2, 16>(stream_0, stream_1);
	SeparableDW2D_relu<128, 3, 16, 32>(stream_1, stream_2, conv2d_depth_weights_1, conv2d_point_weights_1, conv2d_depthwise_bias_1);
	MaxPooling2D<128, 2, 32>(stream_2, stream_3);
	SeparableDW2D_relu_2streams<64, 3, 32, 48>(stream_3, stream_4, stream_skip_2, conv2d_depth_weights_2, conv2d_point_weights_2, conv2d_depthwise_bias_2);
	MaxPooling2D<64, 2, 48>(stream_4, stream_5);
	SeparableDW2D_relu<32, 3, 48, 64>(stream_5, stream_6, conv2d_depth_weights_3, conv2d_point_weights_3, conv2d_depthwise_bias_3);
	MaxPooling2D<32, 2, 64>(stream_6, stream_7);
	//Decoder Layers
	Conv2D_transposed<32, 3, 2, 64, 64> (stream_7, stream_8, conv2d_weights_4, conv2d_bias_4);
	Conv2D_transposed<64, 3, 2, 64, 48> (stream_8, stream_9, conv2d_weights_5, conv2d_bias_5);
	Add<64, 48>(stream_skip_2, stream_9, stream_10);
	Conv2D_transposed<128, 3, 2, 48, 32> (stream_10, stream_11, conv2d_weights_6, conv2d_bias_6);
	Conv2D_transposed<256, 3, 2, 32, 16> (stream_11, stream_12, conv2d_weights_7, conv2d_bias_7);
	Add<256, 16>(stream_skip_1, stream_12, stream_13);
	Conv2D_sigmoid<256, 1, 16, 3> (stream_13, output_stream, conv2d_weights_8, conv2d_bias_8);

	//Write output stream after inference into the output vector to return to TB.
	for (int x = 0; x < model_size_output * model_size_output * model_depth_output; x++) {
			#pragma HLS PIPELINE
			output_stream >> output_image[x];
	}
}
