#include <hls_stream.h>
#include <stdio.h>
#include <stdlib.h>
#include <hls_print.h>
#include "FlareNet.h"
#include <iostream>
#include <string>
#include <fstream>

using namespace std::chrono;

int main() {

 model_type_input input[196608];
 model_type_output output[196608];

  int ret=0;
  int x = 0;
  int int_value = 0;
  model_type_input norm_value = 0;
  ifstream in_fw("C:\\Users\\David\\Desktop\\FlareNet_with_aptfixed_values\\input_2.txt", std::ifstream::in);
  std::string line;

  if (in_fw.is_open()) {
	  while (in_fw) {
	    std::getline (in_fw, line);
	    int_value = std::stoi(line);
	    norm_value = int_value/255.0;
	    input[x] = norm_value;
	    x++;
	  }
  }
  in_fw.close();

  auto start_inference = high_resolution_clock::now();
  for (int i=0; i<1; i++){
	  FlareNet(input, output);
  }

  auto total_execution_time = duration_cast<microseconds>(high_resolution_clock::now() - start_inference);
  std::cout << "duration_inference: " << total_execution_time.count() << '\n';


  ofstream fw("C:\\Users\\David\\Desktop\\FlareNet_with_aptfixed_values\\results_2.txt", std::ofstream::out);
  if (fw.is_open()) {
	  cout << "Opening file";
  }
  for (int x=0; x< model_size_output * model_size_output * model_depth_output; x++) {
		  fw << output[x] << ',' << "\n";
  }
  fw.close();

  ret = system("diff --brief -w C:\\Users\\David\\Desktop\\FlareNet_with_aptfixed_values\\results_2.txt C:\\Users\\David\\Desktop\\FlareNet_with_aptfixed_values\\golden_2.txt");
  if (ret != 0) {
        printf("Test failed  !!!\n");
        ret=1;
  } else {
        printf("Test passed !\n");
  }

  return (ret);
}

