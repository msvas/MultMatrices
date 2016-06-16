void kernel multmat(global const float* A, global const float* B, global float* C, global float* C1){
	int i = get_global_id(0);
	int j = get_global_id(1);
	int m = get_global_id(2);

	int width = get_global_size(2);
	int cols = get_global_size(1);
	int size = width*cols;

	C1[i*size + j*width + m] = A[i*width + m]*B[m*cols + j];

	barrier(CLK_GLOBAL_MEM_FENCE);

	int limit = log2((float)width);
	for (int h=1;h<limit;++h)
		if (m < (width/pow(2,(float)h)))
			C1[i*size + j*width + m] = C1[i*size + j*width + (2*m)] + C1[i*size + j*width + (2*m+1)];

	barrier(CLK_GLOBAL_MEM_FENCE);

	if (m==0)
		C[i*width + j] = C1[i*width + j];
}
