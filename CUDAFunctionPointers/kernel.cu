
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>


__device__ void func1()
{
	printf("Hello from func1\n");
}


// Type definition cannot include __device__ specifier
typedef void(*func_ptr)();

//function pointer defined as *symbols* on the device
//symbol addresses do not exist within the same unified address space like a host accessible device pointer!
__device__ func_ptr ds_f1_ptr = func1;

inline func_ptr get_function(func_ptr d_symbol){
	func_ptr h_ptr;
	
	cudaMemcpyFromSymbol(&h_ptr, d_symbol, sizeof(func_ptr));

	return h_ptr;
}

func_ptr get_function2(func_ptr *d_symbol){
	func_ptr h_ptr;

	cudaMemcpyFromSymbol(&h_ptr, *d_symbol, sizeof(func_ptr));

	return h_ptr;
}




//example kenrel
__global__ void kernel(func_ptr f)
{
	printf("Hello from kernel\n");

	printf("Address of func1 is 0x%08X\n", func1);
	func1();

	printf("Address of ds_f1_ptr is 0x%08X\n", ds_f1_ptr);
	ds_f1_ptr();

	printf("Address of f is 0x%08X\n", f);
	(*f)();

}

void test1()
{
	
	printf("Test1 - works correctly\n");
	//host pointer
	func_ptr h_f1_ptr;

	//get ds_f1_ptr as a device address rather than a device symbol address
	cudaMemcpyFromSymbol(&h_f1_ptr, ds_f1_ptr, sizeof(func_ptr));

	//print symbol address on host (not accessible from host)
	printf("Host address of ds_f1_ptr (symbol) is 0x%08X\n", ds_f1_ptr);

	//print the host copy of the address copied form symbol
	printf("Host address of h_f1_ptr is 0x%08X\n", h_f1_ptr);

	
	//call kernel
	kernel <<<1, 1 >>>(h_f1_ptr);

	//sync to ensure printf outputs
	cudaDeviceSynchronize();

}


void test2()
{
	printf("Test2 - No host copied of device symbol pointers (fails)\n");

	//host pointer
	func_ptr h_f1_ptr;

	const void *temp;

	temp = ds_f1_ptr;

	//get ds_f1_ptr as a device address rather than a device symbol address
	cudaMemcpyFromSymbol(&h_f1_ptr, temp, sizeof(func_ptr));

	//print symbol address on host (not accessible from host)
	printf("Host address of temp (symbol) is 0x%08X\n", temp);

	//print the host copy of the address copied form symbol
	printf("Host address of h_f1_ptr is 0x%08X\n", h_f1_ptr);


	//call kernel
	kernel << <1, 1 >> >(h_f1_ptr);

	//sync to ensure printf outputs
	cudaDeviceSynchronize();

}

void test3()
{
	printf("Test3 - Function to get device symbol on host (must be declared inline)\n");

	//host pointer
	func_ptr h_f1_ptr;

	h_f1_ptr = get_function(ds_f1_ptr);

	//print symbol address on host (not accessible from host)
	printf("Host address of ds_f1_ptr (symbol) is 0x%08X\n", ds_f1_ptr);

	//print the host copy of the address copied form symbol
	printf("Host address of h_f1_ptr is 0x%08X\n", h_f1_ptr);


	//call kernel
	kernel << <1, 1 >> >(h_f1_ptr);

	//sync to ensure printf outputs
	cudaDeviceSynchronize();

}

void test4()
{
	printf("Test4 - String query of device symbol (not supported since CUDA 5)\n");

	//host pointer
	func_ptr h_f1_ptr;

	cudaMemcpyFromSymbol(&h_f1_ptr, "ds_f1_ptr", sizeof(func_ptr));

	//print the host copy of the address copied form symbol
	printf("Host address of h_f1_ptr is 0x%08X\n", h_f1_ptr);


	//call kernel
	kernel << <1, 1 >> >(h_f1_ptr);

	//sync to ensure printf outputs
	cudaDeviceSynchronize();

}

void test5()
{
	printf("Test5 - Pointer to device symbol\n");

	//host pointer
	func_ptr h_f1_ptr;

	//pass device symbol as pointer
	h_f1_ptr = get_function2(&ds_f1_ptr);

	//print symbol address on host (not accessible from host)
	printf("Host address of ds_f1_ptr (symbol) is 0x%08X\n", ds_f1_ptr);

	//print the host copy of the address copied form symbol
	printf("Host address of h_f1_ptr is 0x%08X\n", h_f1_ptr);


	//call kernel
	kernel << <1, 1 >> >(h_f1_ptr);

	//sync to ensure printf outputs
	cudaDeviceSynchronize();

}


int main()
{

	//works
	test1();

	//fails
	//test2();

	//passes in release fails in debug (due to lack on inlining)
	//test3();

	//no longer supported in CUDA
	//test4();

	//
	test5();

	return 0;
}
