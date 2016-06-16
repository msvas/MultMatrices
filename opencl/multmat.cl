
void kernel multmat(global const float* mat1, global const float* mat2, global float* matr, int cols){
	int col = get_global_id(0);
	int row = get_global_id(1); 

	int width = get_global_size(0);

	float result = 0;
	for (int i=0;i<width;++i){
		float val1 = mat1[row*width + i];
		float val2 = mat2[i*cols + col];
		result += val1*val2;
	}

	matr[row*cols + col] = result;
}
