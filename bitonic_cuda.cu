#include <iostream>
#include <fstream>
#include <stdio.h>
#include <vector>
using namespace std;
bool isIncreased(float *data,int n){
	int is_true = 1;
	for(int i = 0;i < n - 1;i ++){
		if(data[i] > data[i + 1]){
			cout << i << endl;
			return 0;
		}
	}
	return is_true;
}
void get_random(float* data, int n){
    srand(0);
	for (int i = 0; i < n; i++) {
		data[i] = (rand() % 1000)/ 10.0;
	}
}
//Version-1
__global__ void naive_bitonic_sort(float *data,int i,int j){
	int tid = threadIdx.x + blockDim.x * blockIdx.x;
	int neighour_data = tid ^ j;//find the pair data
	if(neighour_data > tid){//exchange data by low thread
		if(((tid / i) % 2) == 0){//sort ascending
			if(data[tid] > data[neighour_data]){
				float temp = data[tid];
				data[tid] = data[neighour_data];
				data[neighour_data] = temp;
			}
		}
		else if(((tid / i) % 2) == 1){//sort decending,exist the same data of same position
			if(data[tid] < data[neighour_data]){
				float temp = data[tid];
				data[tid] = data[neighour_data];
				data[neighour_data] = temp;
			}
		}
	}
}
__host__ void naive_call(int data_size,float *cuda_data,int block_size){
	for(int i = 2;i <= data_size;i = i * 2){//stride_len
		for(int j = i/2;j > 0;j = j/2){//calc for the neighborhood
			naive_bitonic_sort<<<(data_size%block_size == 0) ? data_size/block_size : data_size/block_size + 1, block_size>>>(cuda_data,i,j);
		}
	}
}
//Version-2 :Merge1
__global__ void merge_bitonic_sort(float *data,int data_size){
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    int end = min(data_size,blockDim.x);
    int neighour_data;
    float temp;
    for(int i = 2;i <= end;i = i * 2){
        for(int j = i/2;j > 0;j = j/2){
            neighour_data = tid ^ j;//find the pair data
            if(neighour_data > tid){//exchange data by low thread
                if(((tid / i) % 2) == 0){//sort ascending
                    if(data[tid] > data[neighour_data]){
                        temp = data[tid];
                        data[tid] = data[neighour_data];
                        data[neighour_data] = temp;
                    }
                }
                else if(((tid / i) % 2) == 1){//sort decending,exist the same data of same position
                    if(data[tid] < data[neighour_data]){
                        temp = data[tid];
                        data[tid] = data[neighour_data];
                        data[neighour_data] = temp;
                    }
                }
            }
            __syncthreads();
        }
    }
}
__host__ void less_call(int data_size,float *cuda_data,int block_size){
    merge_bitonic_sort<<<(data_size%block_size == 0) ? data_size/block_size : data_size/block_size + 1,block_size>>>(cuda_data,data_size);
    if(block_size < data_size){
        for(int i = block_size * 2;i <= data_size;i *=2){
            for(int j = i/2;j > 0;j = j/2){//calc for the neighborhood
                naive_bitonic_sort<<<(data_size%1024 == 0) ? data_size/1024 : data_size/1024 + 1, 1024>>>(cuda_data,i,j);
            }
        }
    }
    
}
//Version-3 : Merge the Second Half--Merge2
__global__ void merge2bitonic_sort(float *data,int data_size,int i,int block_size){
	int tid = threadIdx.x + blockDim.x * blockIdx.x;
    int end = min(data_size,blockDim.x);
    int neighour_data;
	float temp;
	for(int j = block_size/2;j > 0;j = j/2){
		neighour_data = tid ^ j;//find the pair data
		if(neighour_data > tid){//exchange data by low thread
			if(((tid / i) % 2) == 0){//sort ascending
				if(data[tid] > data[neighour_data]){
					temp = data[tid];
					data[tid] = data[neighour_data];
					data[neighour_data] = temp;
				}
			}
			else if(((tid / i) % 2) == 1){//sort decending,exist the same data of same position
				if(data[tid] < data[neighour_data]){
					temp = data[tid];
					data[tid] = data[neighour_data];
					data[neighour_data] = temp;
				}
			}
		}
		__syncthreads();
	}
}
__host__ void less_call2(int data_size,float *cuda_data,int block_size){
    merge_bitonic_sort<<<(data_size%block_size == 0) ? data_size/block_size : data_size/block_size + 1,block_size>>>(cuda_data,data_size);
    if(block_size < data_size){
        for(int i = block_size * 2;i <= data_size;i *=2){
            for(int j = i/2;j >= block_size;j = j/2){//calc for the neighborhood
                naive_bitonic_sort<<<(data_size%block_size == 0) ? data_size/block_size : data_size/block_size + 1, block_size>>>(cuda_data,i,j);
			}
			merge2bitonic_sort<<<(data_size%(block_size) == 0) ? data_size/(block_size) : data_size/(block_size) + 1, (block_size)>>>(cuda_data,data_size,i,block_size);
        }
    }
    
}
//Version-4 : Merge1 & shared sort
__global__ void shared_sort(float *data,int data_size){
	extern __shared__ float smem[];
	int tid = threadIdx.x + blockDim.x * blockIdx.x;
	smem[threadIdx.x] = data[tid];
    int end = min(data_size,blockDim.x);
    int neighour_data;
    float temp;
    for(int i = 2;i <= end;i = i * 2){
        for(int j = i/2;j > 0;j = j/2){
            neighour_data = threadIdx.x ^ j;//find the pair data
            if(neighour_data > threadIdx.x){//exchange data by low thread
                if(((tid / i) % 2) == 0){//sort ascending
                    if(smem[threadIdx.x] > smem[neighour_data]){
                        temp = smem[threadIdx.x];
                        smem[threadIdx.x] = smem[neighour_data];
                        smem[neighour_data] = temp;
                    }
                }
                else if(((tid / i) % 2) == 1){//sort decending,exist the same smem of same position
                    if(smem[threadIdx.x] < smem[neighour_data]){
                        temp = smem[threadIdx.x];
                        smem[threadIdx.x] = smem[neighour_data];
                        smem[neighour_data] = temp;
                    }
                }
            }
            __syncthreads();
        }
	}
	data[tid] = smem[threadIdx.x];
}
__host__ void shared_less_call(int data_size,float *cuda_data,int block_size){
    shared_sort<<<(data_size%block_size == 0) ? data_size/block_size : data_size/block_size + 1,block_size,block_size*sizeof(float)>>>(cuda_data,data_size);
    if(block_size < data_size){
        for(int i = block_size*2 ;i <= data_size;i *=2){
            for(int j = i/2;j > 0;j = j/2){//calc for the neighborhood
                naive_bitonic_sort<<<(data_size%block_size == 0) ? data_size/block_size : data_size/block_size + 1, block_size>>>(cuda_data,i,j);
            }
        }
    }
    
}
//Version-5 : merge2 & shared used
__host__ void shared_less_call2(int data_size,float *cuda_data,int block_size){
    shared_sort<<<(data_size%block_size == 0) ? data_size/block_size : data_size/block_size + 1,block_size,block_size*sizeof(float)>>>(cuda_data,data_size);
    if(block_size < data_size){
        for(int i = block_size*2 ;i <= data_size;i *=2){
            for(int j = i/2;j >= block_size;j = j/2){//calc for the neighborhood
                naive_bitonic_sort<<<(data_size%block_size == 0) ? data_size/block_size : data_size/block_size + 1, block_size>>>(cuda_data,i,j);
			}
			merge2bitonic_sort<<<(data_size%(block_size) == 0) ? data_size/(block_size) : data_size/(block_size) + 1, (block_size)>>>(cuda_data,data_size,i,block_size);
        }
    }
    
}
//Version-6: 
__global__ void shared_merge2bitonic_sort(float *data,int data_size,int i,int block_size){
	extern __shared__ float smem[];
	int tid = threadIdx.x + blockDim.x * blockIdx.x;
	smem[threadIdx.x] = data[tid];
	__syncthreads();
    int neighour_data;
	float temp;
	for(int j = block_size/2;j > 0;j = j/2){
		neighour_data = threadIdx.x ^ j;//find the pair data
		if(neighour_data > threadIdx.x){//exchange data by low thread
			if(((tid / i) % 2) == 0){//sort ascending
				if(smem[threadIdx.x] > smem[neighour_data]){
					temp = smem[threadIdx.x];
					smem[threadIdx.x] = smem[neighour_data];
					smem[neighour_data] = temp;
				}
			}
			else if(((tid / i) % 2) == 1){//sort decending,exist the same smem of same position
				if(smem[threadIdx.x] < smem[neighour_data]){
					temp = smem[threadIdx.x];
					smem[threadIdx.x] = smem[neighour_data];
					smem[neighour_data] = temp;
				}
			}
		}
		__syncthreads();
	}
	data[tid] = smem[threadIdx.x];
}

__host__ void shared_less_call3(int data_size,float *cuda_data,int block_size){
    shared_sort<<<(data_size%block_size == 0) ? data_size/block_size : data_size/block_size + 1,block_size,block_size*sizeof(float)>>>(cuda_data,data_size);
    if(block_size < data_size){
        for(int i = block_size*2 ;i <= data_size;i *=2){
            for(int j = i/2;j >= block_size;j = j/2){//calc for the neighborhood
                naive_bitonic_sort<<<(data_size%block_size == 0) ? data_size/block_size : data_size/block_size + 1, block_size>>>(cuda_data,i,j);
			}
			shared_merge2bitonic_sort<<<(data_size%(block_size) == 0)?data_size/(block_size):data_size/(block_size)+1,(block_size),block_size*sizeof(float)>>>(cuda_data,data_size,i,block_size);
        }
    }
    
}
//Version-7:更换判断顺序
__global__ void shared_merge2bitonic_sort2(float *data,int data_size,int i,int block_size){
	extern __shared__ float smem[];
	int tid = threadIdx.x + blockDim.x * blockIdx.x;
	smem[threadIdx.x] = data[tid];
	__syncthreads();
    int neighour_data;
	float temp;
	if(((tid / i) % 2) == 0) {
		for(int j = block_size/2;j > 0;j = j/2){
			neighour_data = threadIdx.x ^ j;//find the pair data
			if(neighour_data > threadIdx.x && smem[threadIdx.x] > smem[neighour_data]){//exchange data by low thread
				temp = smem[threadIdx.x];
				smem[threadIdx.x] = smem[neighour_data];
				smem[neighour_data] = temp;
			}
			__syncthreads();
		}
	}
	if(((tid / i) % 2) == 1) {
		for(int j = block_size/2;j > 0;j = j/2){
			neighour_data = threadIdx.x ^ j;//find the pair data
			if(neighour_data > threadIdx.x && smem[threadIdx.x] < smem[neighour_data]){//exchange data by low thread
				temp = smem[threadIdx.x];
				smem[threadIdx.x] = smem[neighour_data];
				smem[neighour_data] = temp;
			}
			__syncthreads();
		}
	}
	data[tid] = smem[threadIdx.x];
}
__host__ void shared_less_call4(int data_size,float *cuda_data,int block_size){
    shared_sort<<<(data_size%block_size == 0) ? data_size/block_size : data_size/block_size + 1,block_size,block_size*sizeof(float)>>>(cuda_data,data_size);
    if(block_size < data_size){
        for(int i = block_size*2 ;i <= data_size;i *=2){
            for(int j = i/2;j >= block_size;j = j/2){//calc for the neighborhood
                naive_bitonic_sort<<<(data_size%block_size == 0) ? data_size/block_size : data_size/block_size + 1, block_size>>>(cuda_data,i,j);
			}
			shared_merge2bitonic_sort2<<<(data_size%(block_size) == 0)?data_size/(block_size):data_size/(block_size)+1,(block_size),block_size*sizeof(float)>>>(cuda_data,data_size,i,block_size);
        }
    }
    
}
//Version-8
__global__ void unroll_shared_merge2bitonic_sort2(float *data,int data_size,int i,int block_size){
	extern __shared__ float smem[];
	int tid = threadIdx.x + blockDim.x * blockIdx.x;
	smem[threadIdx.x] = data[tid];
	__syncthreads();
    int neighour_data;
	float temp;
	int j;
	if(((tid / i) % 2) == 0) {
		for(j = block_size/2;j > 256;j = j/2){
			neighour_data = threadIdx.x ^ j;//find the pair data
			if(neighour_data > threadIdx.x && smem[threadIdx.x] > smem[neighour_data]){//exchange data by low thread
				temp = smem[threadIdx.x];
				smem[threadIdx.x] = smem[neighour_data];
				smem[neighour_data] = temp;
			}
			__syncthreads();
		}
		j = 256;
		neighour_data = threadIdx.x ^ j;//find the pair data
		if(neighour_data > threadIdx.x && smem[threadIdx.x] > smem[neighour_data]){//exchange data by low thread
			temp = smem[threadIdx.x];
			smem[threadIdx.x] = smem[neighour_data];
			smem[neighour_data] = temp;
		}
		__syncthreads();
		j = 128;
		neighour_data = threadIdx.x ^ j;//find the pair data
		if(neighour_data > threadIdx.x && smem[threadIdx.x] > smem[neighour_data]){//exchange data by low thread
			temp = smem[threadIdx.x];
			smem[threadIdx.x] = smem[neighour_data];
			smem[neighour_data] = temp;
		}
		__syncthreads();
		j = 64;
		neighour_data = threadIdx.x ^ j;//find the pair data
		if(neighour_data > threadIdx.x && smem[threadIdx.x] > smem[neighour_data]){//exchange data by low thread
			temp = smem[threadIdx.x];
			smem[threadIdx.x] = smem[neighour_data];
			smem[neighour_data] = temp;
		}
		__syncthreads();
		j = 32;
		neighour_data = threadIdx.x ^ j;//find the pair data
		if(neighour_data > threadIdx.x && smem[threadIdx.x] > smem[neighour_data]){//exchange data by low thread
			temp = smem[threadIdx.x];
			smem[threadIdx.x] = smem[neighour_data];
			smem[neighour_data] = temp;
		}
		__syncthreads();
		j = 16;
		neighour_data = threadIdx.x ^ j;//find the pair data
		if(neighour_data > threadIdx.x && smem[threadIdx.x] > smem[neighour_data]){//exchange data by low thread
			temp = smem[threadIdx.x];
			smem[threadIdx.x] = smem[neighour_data];
			smem[neighour_data] = temp;
		}
		__syncthreads();
		j = 8;
		neighour_data = threadIdx.x ^ j;//find the pair data
		if(neighour_data > threadIdx.x && smem[threadIdx.x] > smem[neighour_data]){//exchange data by low thread
			temp = smem[threadIdx.x];
			smem[threadIdx.x] = smem[neighour_data];
			smem[neighour_data] = temp;
		}
		__syncthreads();
		j = 4;
		neighour_data = threadIdx.x ^ j;//find the pair data
		if(neighour_data > threadIdx.x && smem[threadIdx.x] > smem[neighour_data]){//exchange data by low thread
			temp = smem[threadIdx.x];
			smem[threadIdx.x] = smem[neighour_data];
			smem[neighour_data] = temp;
		}
		__syncthreads();
		j = 2;
		neighour_data = threadIdx.x ^ j;//find the pair data
		if(neighour_data > threadIdx.x && smem[threadIdx.x] > smem[neighour_data]){//exchange data by low thread
			temp = smem[threadIdx.x];
			smem[threadIdx.x] = smem[neighour_data];
			smem[neighour_data] = temp;
		}
		__syncthreads();
		j = 1;
		neighour_data = threadIdx.x ^ j;//find the pair data
		if(neighour_data > threadIdx.x && smem[threadIdx.x] > smem[neighour_data]){//exchange data by low thread
			temp = smem[threadIdx.x];
			smem[threadIdx.x] = smem[neighour_data];
			smem[neighour_data] = temp;
		}
		__syncthreads();

	}
	if(((tid / i) % 2) == 1) {
		for( j = block_size/2;j > 0;j = j/2){
			neighour_data = threadIdx.x ^ j;//find the pair data
			if(neighour_data > threadIdx.x && smem[threadIdx.x] < smem[neighour_data]){//exchange data by low thread
				temp = smem[threadIdx.x];
				smem[threadIdx.x] = smem[neighour_data];
				smem[neighour_data] = temp;
			}
			__syncthreads();
		}
		j = 256;
		neighour_data = threadIdx.x ^ j;//find the pair data
		if(neighour_data > threadIdx.x && smem[threadIdx.x] < smem[neighour_data]){//exchange data by low thread
			temp = smem[threadIdx.x];
			smem[threadIdx.x] = smem[neighour_data];
			smem[neighour_data] = temp;
		}
		__syncthreads();
		j = 128;
		neighour_data = threadIdx.x ^ j;//find the pair data
		if(neighour_data > threadIdx.x && smem[threadIdx.x] < smem[neighour_data]){//exchange data by low thread
			temp = smem[threadIdx.x];
			smem[threadIdx.x] = smem[neighour_data];
			smem[neighour_data] = temp;
		}
		__syncthreads();
		j = 64;
		neighour_data = threadIdx.x ^ j;//find the pair data
		if(neighour_data > threadIdx.x && smem[threadIdx.x] < smem[neighour_data]){//exchange data by low thread
			temp = smem[threadIdx.x];
			smem[threadIdx.x] = smem[neighour_data];
			smem[neighour_data] = temp;
		}
		__syncthreads();
		j = 32;
		neighour_data = threadIdx.x ^ j;//find the pair data
		if(neighour_data > threadIdx.x && smem[threadIdx.x] < smem[neighour_data]){//exchange data by low thread
			temp = smem[threadIdx.x];
			smem[threadIdx.x] = smem[neighour_data];
			smem[neighour_data] = temp;
		}
		__syncthreads();
		j = 16;
		neighour_data = threadIdx.x ^ j;//find the pair data
		if(neighour_data > threadIdx.x && smem[threadIdx.x] < smem[neighour_data]){//exchange data by low thread
			temp = smem[threadIdx.x];
			smem[threadIdx.x] = smem[neighour_data];
			smem[neighour_data] = temp;
		}
		__syncthreads();
		j = 8;
		neighour_data = threadIdx.x ^ j;//find the pair data
		if(neighour_data > threadIdx.x && smem[threadIdx.x] < smem[neighour_data]){//exchange data by low thread
			temp = smem[threadIdx.x];
			smem[threadIdx.x] = smem[neighour_data];
			smem[neighour_data] = temp;
		}
		__syncthreads();
		j = 4;
		neighour_data = threadIdx.x ^ j;//find the pair data
		if(neighour_data > threadIdx.x && smem[threadIdx.x] < smem[neighour_data]){//exchange data by low thread
			temp = smem[threadIdx.x];
			smem[threadIdx.x] = smem[neighour_data];
			smem[neighour_data] = temp;
		}
		__syncthreads();
		j = 2;
		neighour_data = threadIdx.x ^ j;//find the pair data
		if(neighour_data > threadIdx.x && smem[threadIdx.x] < smem[neighour_data]){//exchange data by low thread
			temp = smem[threadIdx.x];
			smem[threadIdx.x] = smem[neighour_data];
			smem[neighour_data] = temp;
		}
		__syncthreads();
		j = 1;
		neighour_data = threadIdx.x ^ j;//find the pair data
		if(neighour_data > threadIdx.x && smem[threadIdx.x] < smem[neighour_data]){//exchange data by low thread
			temp = smem[threadIdx.x];
			smem[threadIdx.x] = smem[neighour_data];
			smem[neighour_data] = temp;
		}
		__syncthreads();
	}
	data[tid] = smem[threadIdx.x];
}
__host__ void unroll_shared_less_call4(int data_size,float *cuda_data,int block_size){
    shared_sort<<<(data_size%block_size == 0) ? data_size/block_size : data_size/block_size + 1,block_size,block_size*sizeof(float)>>>(cuda_data,data_size);
    if(block_size < data_size){
        for(int i = block_size*2 ;i <= data_size;i *=2){
            for(int j = i/2;j >= block_size;j = j/2){//calc for the neighborhood
                naive_bitonic_sort<<<(data_size%block_size == 0) ? data_size/block_size : data_size/block_size + 1, block_size>>>(cuda_data,i,j);
			}
			unroll_shared_merge2bitonic_sort2<<<(data_size%(block_size) == 0)?data_size/(block_size):data_size/(block_size)+1,(block_size),block_size*sizeof(float)>>>(cuda_data,data_size,i,block_size);
        }
    }
    
}
int main(){
	int block_size = 1024;
	cudaSetDevice(1);
    int whole_size = 1000000;
	float *data_cpu = new float[whole_size];
    get_random(data_cpu,whole_size);
    //padding
	int data_size = 1;
	int n = 0;
    while(data_size < whole_size){
		n ++;
        data_size*=2;
	}
	float *cuda_data,*cuda_data2,*cuda_data3,*cuda_data4,*cuda_data5,*cuda_data6,*cuda_data7,*cuda_data8;
	float *result_data = new float[whole_size];
    cudaMalloc((void **) &cuda_data,data_size*sizeof(float));
	cudaMalloc((void **) &cuda_data2,data_size*sizeof(float));
	cudaMalloc((void **) &cuda_data3,data_size*sizeof(float));
	cudaMalloc((void **) &cuda_data4,data_size*sizeof(float));
	cudaMalloc((void **) &cuda_data5,data_size*sizeof(float));
	cudaMalloc((void **) &cuda_data6,data_size*sizeof(float));
	cudaMalloc((void **) &cuda_data7,data_size*sizeof(float));
	cudaMalloc((void **) &cuda_data8,data_size*sizeof(float));
	cudaMemcpy(cuda_data, data_cpu, whole_size*sizeof(float),cudaMemcpyHostToDevice);
	float start = clock();
    naive_call(data_size,cuda_data,1024);
    cudaDeviceSynchronize();
	float end = clock();
    cout << "data_size : " << data_size <<" GPU naive bitonic sort time :" << float(end-start) * 1000.0/CLOCKS_PER_SEC << " ms" << endl;
	cudaMemcpy(result_data, cuda_data + data_size - whole_size, whole_size*sizeof(float),cudaMemcpyDeviceToHost);
	for(int i = whole_size - 10;i < whole_size;i ++){
		printf("%.1f ", result_data[i]);
	}
	cout << endl;
	cout << "Right or false? " <<isIncreased(result_data,whole_size) << endl;

	cudaMemcpy(cuda_data2, data_cpu, whole_size*sizeof(float),cudaMemcpyHostToDevice);
	start = clock();
	less_call(data_size,cuda_data2,256);
    cudaDeviceSynchronize();
    end = clock();
    cout << "data_size : " << data_size <<" GPU merge1 bitonic sort time :" << float(end-start) * 1000.0/CLOCKS_PER_SEC << " ms" << endl;
    float *result_data2 = new float[whole_size];
    cudaMemcpy(result_data2, cuda_data2 + data_size - whole_size, whole_size*sizeof(float),cudaMemcpyDeviceToHost);
	cout << "Right or false? " <<isIncreased(result_data2,whole_size) << endl;
	for(int i = whole_size - 10;i < whole_size;i ++){
		printf("%.1f ", result_data2[i]);
	}
	cout << endl;

	cudaMemcpy(cuda_data3, data_cpu, whole_size*sizeof(float),cudaMemcpyHostToDevice);
	start = clock();
	less_call2(data_size,cuda_data3,256);
    cudaDeviceSynchronize();
    end = clock();
    cout << "data_size : " << data_size <<" GPU merge2 bitonic sort time :" << float(end-start) * 1000.0/CLOCKS_PER_SEC << " ms" << endl;
    float *result_data3 = new float[whole_size];
    cudaMemcpy(result_data3, cuda_data3 + data_size - whole_size, whole_size*sizeof(float),cudaMemcpyDeviceToHost);
	cout << "Right or false? " <<isIncreased(result_data3,whole_size) << endl;
	for(int i = whole_size - 10;i < whole_size;i ++){
		printf("%.1f ", result_data3[i]);
	}
	cout << endl;

	cudaMemcpy(cuda_data4, data_cpu, whole_size*sizeof(float),cudaMemcpyHostToDevice);
	start = clock();
	shared_less_call(data_size,cuda_data4,256);
    cudaDeviceSynchronize();
    end = clock();
    cout << "data_size : " << data_size <<" GPU shared + merge1 bitonic sort time :" << float(end-start) * 1000.0/CLOCKS_PER_SEC << " ms" << endl;
    float *result_data4 = new float[whole_size];
    cudaMemcpy(result_data4, cuda_data4 + data_size - whole_size, whole_size*sizeof(float),cudaMemcpyDeviceToHost);
	cout << "Right or false? " <<isIncreased(result_data4,whole_size) << endl;
	for(int i = whole_size - 10;i < whole_size;i ++){
		printf("%.1f ", result_data4[i]);
	}
	cout << endl;

	cudaMemcpy(cuda_data5, data_cpu, whole_size*sizeof(float),cudaMemcpyHostToDevice);
	start = clock();
	shared_less_call2(data_size,cuda_data5,256);
    cudaDeviceSynchronize();
    end = clock();
    cout << "data_size : " << data_size <<" GPU shared + merge2 bitonic sort time :" << float(end-start) * 1000.0/CLOCKS_PER_SEC << " ms" << endl;
    float *result_data5 = new float[whole_size];
    cudaMemcpy(result_data5, cuda_data5 + data_size - whole_size, whole_size*sizeof(float),cudaMemcpyDeviceToHost);
	cout << "Right or false? " <<isIncreased(result_data5,whole_size) << endl;
	for(int i = whole_size - 10;i < whole_size;i ++){
		printf("%.1f ", result_data5[i]);
	}
	cout << endl;

	cudaMemcpy(cuda_data6, data_cpu, whole_size*sizeof(float),cudaMemcpyHostToDevice);
	start = clock();
	shared_less_call3(data_size,cuda_data6,256);
    cudaDeviceSynchronize();
    end = clock();
    cout << "data_size : " << data_size <<" GPU shared2 + merge2 bitonic sort time :" << float(end-start) * 1000.0/CLOCKS_PER_SEC << " ms" << endl;
    float *result_data6 = new float[whole_size];
    cudaMemcpy(result_data6, cuda_data6 + data_size - whole_size, whole_size*sizeof(float),cudaMemcpyDeviceToHost);
	cout << "Right or false? " <<isIncreased(result_data6,whole_size) << endl;
	for(int i = whole_size - 10;i < whole_size;i ++){
		printf("%.1f ", result_data6[i]);
	}
	cout << endl;

	cudaMemcpy(cuda_data7, data_cpu, whole_size*sizeof(float),cudaMemcpyHostToDevice);
	start = clock();
	shared_less_call4(data_size,cuda_data7,256);
	cudaDeviceSynchronize();
	end = clock();
	cout << "block : " << 256 <<" GPU change order shared2 + merge2 bitonic sort time :" << float(end-start) * 1000.0/CLOCKS_PER_SEC << " ms" << endl;
	float *result_data7 = new float[whole_size];
    cudaMemcpy(result_data7, cuda_data7 + data_size - whole_size, whole_size*sizeof(float),cudaMemcpyDeviceToHost);
	cout << "Right or false? " <<isIncreased(result_data7,whole_size) << endl;
	for(int i = whole_size - 10;i < whole_size;i ++){
		printf("%.1f ", result_data7[i]);
	}
	cout << endl;

	

	cudaMemcpy(cuda_data8, data_cpu, whole_size*sizeof(float),cudaMemcpyHostToDevice);
	start = clock();
	unroll_shared_less_call4(data_size,cuda_data8,512);
    cudaDeviceSynchronize();
    end = clock();
    cout << "data_size : " << data_size <<" GPU unroll change order shared2 + merge2 bitonic sort time :" << float(end-start) * 1000.0/CLOCKS_PER_SEC << " ms" << endl;
    float *result_data8 = new float[whole_size];
    cudaMemcpy(result_data8, cuda_data8 + data_size - whole_size, whole_size*sizeof(float),cudaMemcpyDeviceToHost);
	cout << "Right or false? " <<isIncreased(result_data8,whole_size) << endl;
	for(int i = whole_size - 10;i < whole_size;i ++){
		printf("%.1f ", result_data8[i]);
	}
	cout << endl;
	cudaFree(cuda_data);
	// cudaFree(cuda_data1);
	// cudaFree(cuda_data2);
	// cudaFree(cuda_data3);
	// cudaFree(cuda_data4);
	// cudaFree(cuda_data5);

}