
#include <utility>
//#define __NO_STD_VECTOR
#include <CL/cl.hpp>

#include <cstdlib>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <chrono>

#define CLDEVTYPE CL_DEVICE_TYPE_ALL 

using namespace std;

template <typename T> struct mat {
	int rows;
	int cols;
	vector<T> data;
};

template <typename T> void print_mat(mat<T> &mt){
	for (int i=0,l=mt.rows;i<l;++i){
		for (int j=0,m=mt.cols;j<m;++j){
			cout << mt.data[i*m + j] << " ";
		}
		cout << endl;
	}
}

template <typename T> void init_mat(int rows, int cols, mat<T> &mt){
	mt.rows = rows;
	mt.cols = cols;
	mt.data.assign(rows*cols,0);
}

template <typename T> bool read_mat(string filename, mat<T> &mt){
	ifstream file;

	file.open(filename);
	if (!file.is_open())
		return false;

	int rows,cols;
	file >> rows >> cols;
	mt.rows = rows;
	mt.cols = cols;

	for (int i=0;i<rows;++i){
		for (int j=0;j<cols;++j){
			T value;			
			file >> value;
			mt.data.push_back(value);
			if (file.eof())
				return false;
		}
	}

	return true;
}

template <typename T> mat<T> transpose(mat<T> &mt){
	int rows = mt.rows;
	int cols = mt.cols;	
	vector<T> tr;

	for (int i=0;i<cols;++i){
		for (int j=0;j<rows;++j){
			tr.push_back(mt.data[j*cols + i]);
		}
	}

	return {cols,rows,tr};
}

void cl_check_error(cl_int err, const char * name){
	if (err != CL_SUCCESS)
		exit(((cerr << "ERROR: " << name  << " (" << err << ")" << endl),EXIT_FAILURE));
}

void cl_print_platform(cl::Platform &plat){
	string name,vendor,version;
	plat.getInfo((cl_platform_info)CL_PLATFORM_NAME,&name);
	plat.getInfo((cl_platform_info)CL_PLATFORM_VENDOR,&vendor);
	plat.getInfo((cl_platform_info)CL_PLATFORM_VERSION,&version);

	cerr << "Platform: " << name << endl;
	cerr << "Vendor: " << vendor << endl;
	cerr << "Version: " << version << endl;
}

void cl_print_device(cl::Device &dev){
	string name,vendor,version,kernels;
	dev.getInfo((cl_device_info)CL_DEVICE_NAME,&name);
	dev.getInfo((cl_device_info)CL_DEVICE_VENDOR,&vendor);
	dev.getInfo((cl_device_info)CL_DEVICE_VERSION,&version);
	dev.getInfo((cl_device_info)CL_DEVICE_BUILT_IN_KERNELS,&kernels);

	cerr << "Device: " << name << endl;
	cerr << "Vendor: " << vendor << endl;
	cerr << "Version: " << version << endl;
	cerr << "Kernels: " << kernels << endl;
}

void cl_init_plat_device(cl::Platform &plat, int platform_id, cl::Device device, int device_id){
	vector<cl::Platform> platforms;
	cl::Platform::get(&platforms);
	cl_check_error(platforms.size()!=0 ? CL_SUCCESS : -1, "cl::Platform::get");

	cerr << "OpenCL platforms available: " << platforms.size() << endl;

	for (int i=0,l=platforms.size();i<l;++i){
		cerr << "# Platform ID " << i << endl;  
		cl_print_platform(platforms[i]);
	}

	cerr << "Using platform " << platform_id << endl;
	plat = platforms[platform_id];

	vector<cl::Device> devices;
	plat.getDevices(CLDEVTYPE,&devices);
	cl_check_error(devices.size()!=0 ? CL_SUCCESS : -1, "cl::Platform::getDevices");

	cerr << "OpenCL devices available: " << devices.size() << endl;

	for (int i=0,l=devices.size();i<l;++i){
		cerr << "# Device ID " << i << endl;  
		cl_print_device(devices[i]);
	}

	//cerr << "Using device " << device_id << endl;
	//device = devices[device_id];
}

cl::Context cl_create_context(cl::Device &device, cl::Platform &plat){
	cl_int err;
	cl_context_properties cprops[3] = {CL_CONTEXT_PLATFORM,(cl_context_properties)(plat)(),0};

	cl::Context context(CLDEVTYPE,cprops,NULL,NULL,&err);
	cl_check_error(err, "Context::Context()"); 

	vector<cl::Device> devices;
	context.getInfo((cl_context_info)CL_CONTEXT_DEVICES,&devices);
	device = devices[0];

	string name;
	device.getInfo((cl_device_info)CL_DEVICE_NAME,&name);
	cerr << "Created context with device: " << name << endl;

	return context;
}

cl::Program cl_make_program(string filename, cl::Context context){
	ifstream file;
	file.open(filename);
	cl_check_error(file.is_open() ? CL_SUCCESS:-1,filename.c_str());

	string code;
	file.seekg(0,ios::end);   
	code.reserve(file.tellg());
	file.seekg(0,ios::beg);

	code.assign(istreambuf_iterator<char>(file),istreambuf_iterator<char>());

	cl::Program::Sources source(1,make_pair(code.c_str(),code.length()+1));
	cl::Program program(context, source);

	vector<cl::Device> devices;
	context.getInfo((cl_context_info)CL_CONTEXT_DEVICES,&devices);

	cl_int err = program.build(devices,"");
	if (err != CL_SUCCESS)
		exit(((cerr << "Error building program: " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(devices[0]) << endl),1));

	return program;
}

cl::Kernel cl_create_kernel(cl::Program program){
	cl_int err;	
	cl::Kernel kernel(program,"hello",&err);
	cl_check_error(err,"Kernel::Kernel()");
	return kernel;
}

int main(int argc, char **argv){
	if (argc < 3)
		return (cerr << "Usage: multmat mat1 mat2 [result only:(1|0) supress:2] [cl platform] [cl device]" << endl),1;

	auto start = std::chrono::system_clock::now();

	int platid = 0;
	int deviceid = 0;
	bool notresultonly = true;
	bool noresult = false;

	if (argc > 3){
		notresultonly = (argv[3][0] == '0'); 
		noresult = (argv[3][0] == '2');
		if (argc>4){
			platid = atoi(argv[4]);
			if (argc>5) 
				deviceid = atoi(argv[5]);
		}
	}

	cl::Platform plat;
	cl::Device device;

	cl_init_plat_device(plat,platid,device,deviceid);

	mat<float> mat1,mat2;

	read_mat(string(argv[1]),mat1);
	read_mat(string(argv[2]),mat2);

	if (mat1.cols != mat2.rows)
		return (cerr << "Matrix dimensions invalid for multiplication" << endl),1;

	// tamanho de cada matriz na memória
	int matsize1 = mat1.rows*mat1.cols*sizeof(float);
	int matsize2 = mat2.rows*mat2.cols*sizeof(float);
	int matsizer = mat1.rows*mat2.cols*sizeof(float);
	int matsizem = mat1.rows*mat1.cols*mat2.cols*sizeof(float);

	cl::Context context = cl_create_context(device,plat);
	cl::Program program = cl_make_program(string("multmat.cl"),context);

	cl::Buffer vmat1(context,CL_MEM_READ_ONLY,matsize1);
	cl::Buffer vmat2(context,CL_MEM_READ_ONLY,matsize2);
	cl::Buffer vmatr(context,CL_MEM_WRITE_ONLY,matsizer);
	cl::Buffer vmat3(context,CL_MEM_READ_WRITE|CL_MEM_HOST_READ_ONLY,matsizem);

	cl::CommandQueue queue(context,device);
	queue.enqueueWriteBuffer(vmat1,CL_TRUE,0,matsize1,mat1.data.data());
	queue.enqueueWriteBuffer(vmat2,CL_TRUE,0,matsize2,mat2.data.data());

	cl::Event event;
	cl::Kernel kernel(program,"multmat");
	kernel.setArg(0,vmat1);
	kernel.setArg(1,vmat2);
	kernel.setArg(2,vmatr);
	kernel.setArg(3,vmat3);

	queue.enqueueNDRangeKernel(kernel,cl::NullRange,cl::NDRange(mat1.rows,mat2.cols,mat1.cols),cl::NullRange,NULL,&event);

	mat<float> matr;
	init_mat(mat1.rows,mat2.cols,matr);

	event.wait();

	queue.enqueueReadBuffer(vmatr,CL_TRUE,0,matsizer,matr.data.data());

	if (!noresult){
		if (notresultonly){
			cout << "Mat 1" << endl;
			print_mat(mat1);
			cout << endl << "Mat 2" << endl;
			print_mat(mat2);
			cout << endl << "Result" << endl;
		} else 
			cout << matr.rows << " " << matr.cols << endl;
		print_mat(matr);
	}
	
	auto end = std::chrono::system_clock::now();
	auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
	cerr << endl << "time: " << elapsed.count() << "ms" << endl;
}
