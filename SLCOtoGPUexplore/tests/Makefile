all:
	nvcc -arch=sm_75 -lcudart -o gpuexplore gpuexplore.cu
debug:
	nvcc -arch=sm_75 -g -G -Xcompiler -rdynamic -lcudart -o gpuexplore gpuexplore.cu
