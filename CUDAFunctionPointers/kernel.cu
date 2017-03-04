
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

int main()
{
	
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

    return 0;
}
