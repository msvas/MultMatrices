
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#include <cstdlib>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <chrono>

using namespace std;

template <typename T> struct mat {
	int rows;
	int cols;
	vector<T> data;
};

cudaError_t multMatrices(mat<float> *c, mat<float> a, mat<float> b, unsigned int threads, unsigned int blocks);

__global__ void multKernel(float *matr, const float *mat1, const float *mat2, const int width)
{
    int tId = threadIdx.x;
	int bId = blockIdx.x;
	int cols = blockDim.x;
    //c[i] = a[i] + b[i];
	//int col = get_global_id(0);
	//int row = get_global_id(1);

	float result = 0;

	for (int i = 0; i < width; ++i){
		float val1 = mat1[bId*width + i];
		float val2 = mat2[i * cols + tId];
		result += val1 * val2;
		//printf("%i - %i - %f, %f, %f\n", bId*width + i, i * cols + tId, val1, val2, result);
	}
	//printf("%i, %i, %f\n", bId, tId, result);
	matr[bId * cols + tId] = result;
}

template <typename T> void print_mat(mat<T> &mt){
	for (int i = 0, l = mt.rows; i<l; ++i){
		for (int j = 0, m = mt.cols; j<m; ++j){
			cout << mt.data[i*m + j] << " ";
		}
		cout << endl;
	}
}

template <typename T> bool read_mat(string filename, mat<T> &mt){
	ifstream file;

	file.open(filename);
	if (!file.is_open())
		return false;

	int rows, cols;
	file >> rows >> cols;
	mt.rows = rows;
	mt.cols = cols;

	for (int i = 0; i<rows; ++i){
		for (int j = 0; j<cols; ++j){
			T value;
			file >> value;
			mt.data.push_back(value);
			if (file.eof())
				return false;
		}
	}

	return true;
}

template <typename T> void init_mat(int rows, int cols, mat<T> &mt){
	mt.rows = rows;
	mt.cols = cols;
	mt.data.assign(rows*cols, 0);
}

int main(int argc, char **argv) {
    //const int arraySize = 5;
    //const int a[arraySize] = { 1, 2, 3, 4, 5 };
    //const int b[arraySize] = { 10, 20, 30, 40, 50 };
    //int c[arraySize] = { 0 };
	mat<float> mat1, mat2, matr;

	if (argc < 3) {
		return (cerr << "Usage: multmat mat1 mat2 [result only:(1|0) supress:2]" << endl), 1;
	}

	auto start = std::chrono::system_clock::now();

	bool notresultonly = true;
	bool noresult = false;

	if (argc > 3){
		notresultonly = (argv[3][0] == '0');
		noresult = (argv[3][0] == '2');
	}

	read_mat(argv[1], mat1);
	read_mat(argv[2], mat2);
	init_mat(mat1.rows, mat2.cols, matr);

    // Add vectors in parallel.
    cudaError_t cudaStatus = multMatrices(&matr, mat1, mat2, mat2.cols, mat1.rows);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        return 1;
    }
    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

	if (!noresult){
		if (notresultonly){
			cout << "Mat 1" << endl;
			print_mat(mat1);
			cout << endl << "Mat 2" << endl;
			print_mat(mat2);
			cout << endl << "Result" << endl;
		}
		else
			cout << matr.rows << " " << matr.cols << endl;
		print_mat(matr);
	}

	auto end = std::chrono::system_clock::now();
	auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
	cerr << endl << "time: " << elapsed.count() << "ms" << endl;

    return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t multMatrices(mat<float> *c, mat<float> a, mat<float> b, unsigned int threads, unsigned int blocks)
{
    float *dev_a = 0;
    float *dev_b = 0;
    float *dev_c = 0;
	float size = threads * blocks;
    cudaError_t cudaStatus;
	
    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

	cudaStatus = cudaMalloc((void**)&dev_a, a.cols * a.rows * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

	cudaStatus = cudaMalloc((void**)&dev_b, b.cols * b.rows * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(dev_a, a.data.data(), a.cols * a.rows * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

	cudaStatus = cudaMemcpy(dev_b, b.data.data(), b.cols * b.rows * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    multKernel<<< threads, blocks >>>(dev_c, dev_a, dev_b, a.cols);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c->data.data(), dev_c, size * sizeof(float), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
    
    return cudaStatus;
}
