#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <assert.h>
#include <time.h>
#include <math.h>
#include "example_system_gpuexplore.cuh"

#define MIN(a,b) \
   ({ __typeof__ (a) _a = (a); \
       __typeof__ (b) _b = (b); \
     _a < _b ? _a : _b; })

#define MAX(a,b) \
   ({ __typeof__ (a) _a = (a); \
       __typeof__ (b) _b = (b); \
     _a > _b ? _a : _b; })

/**
 * This macro checks return value of the CUDA runtime call and exits
 * the application if the call failed.
 */
#define CUDA_CHECK_RETURN(value) {											\
	cudaError_t _m_cudaStat = value;										\
	if (_m_cudaStat != cudaSuccess) {										\
		fprintf(stderr, "Error %s at line %d in file %s\n",					\
				cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__);		\
		exit(1);															\
	} }

__inline__ void CUDA_CHECK_FOR_ERROR(int i) {
	//fprintf(stderr, "Call %u:\n", i);
	cudaError_t err = cudaGetLastError();
	CUDA_CHECK_RETURN(err);
}

int vmem = 0;

// Wrapper around cudaMalloc to count allocated memory and check for error while allocating.
int cudaMallocCount ( void ** ptr,int size) {
	cudaError_t err = cudaSuccess;
	vmem += size;
	err = cudaMalloc(ptr,size);
	if (err) {
		printf("Error %s at line %d in file %s\n", cudaGetErrorString(err), __LINE__, __FILE__);
		exit(1);
	}
	fprintf (stdout, "allocated %d\n", size);
	return size;
}

/**
 * CUDA kernel function to initialise the global memory hash table.
 */
__global__ void init_hash_table(compressed_nodetype *d_q, nodetype *d_q_i) {
    for (uint64_t i = GLOBAL_THREAD_ID; i < d_hash_table_size; i += NR_THREADS) {
    	d_q[i] = (compressed_nodetype) EMPTY_COMPRESSED_NODE;
    }
    for (uint64_t i = GLOBAL_THREAD_ID; i < d_internal_hash_table_size; i += NR_THREADS) {
    	d_q_i[i] = (nodetype) EMPTY_NODE;
    }
}

/**
 * CUDA kernel function to count the number of states in the global memory hash table.
 */
__global__ void count_states(compressed_nodetype *d_q, uint64_t *result) {
 	__shared__ uint64_t blockcount;

	if (THREAD_ID == 0) {
		blockcount = 0;
	}
	__syncthreads();
	uint64_t localResult = 0;
	for (uint64_t i = GLOBAL_THREAD_ID; i < d_hash_table_size; i += NR_THREADS) {
		if (d_q[i] != EMPTY_COMPRESSED_NODE) {
			localResult++;
		}
	}
	if (localResult > 0) {
		atomicAdd((unsigned long long *) &blockcount, (unsigned long long) localResult);
	}
	__syncthreads();
	if (THREAD_ID == 0) {
		if (blockcount > 0) {
			atomicAdd((unsigned long long *) result, (unsigned long long) blockcount);
		}
	}
}

/**
 * CUDA kernel function to prepare the cache for a new successor generation iteration.
 */
inline __device__ void PREPARE_CACHE() {
	// Traverse the new state vector trees, and reconstruct the nodes, i.e., put the cache pointers back in place and reconstruct the non-leaf nodes
	// by inspecting their children.
	shared_inttype pointers;
	shared_indextype addr;
	nodetype node;
	bool next_it, is_required;

	// First we mark the root nodes referred to in the worktile for preparation.
	for (shared_indextype i = THREAD_ID; i < OPENTILECOUNT; i += BLOCK_SIZE) {
		pointers = shared[OPENTILEOFFSET+i];
		if (!worktile_element_requires_fetching(pointers)) {
			mark_cached_node_as_next_in_preparation(&shared[CACHEOFFSET+(pointers*3)+2]);
		}
	}

	while (CONTINUE == 1) {
		__syncthreads();
		if (THREAD_ID == 0) {
			CONTINUE = 0;
		}
		__syncthreads();
		#pragma unroll
		for (shared_indextype i = THREAD_ID; (i*3)+2 < d_shared_cache_size - CACHEOFFSET; i += BLOCK_SIZE) {
			if (cached_node_is_next_in_preparation(shared[CACHEOFFSET+(i*3)+2])) {
				node = combine_halfs(shared[CACHEOFFSET+(i*3)], shared[CACHEOFFSET+(i*3)+1]);
				is_required = false;
				if (!is_root(node)) {
					is_required = cached_node_is_required(shared[CACHEOFFSET+(i*3)]);
					// Put the original cache pointers back if the node is new, i.e., it is not yet set as required.
					if (!is_required) {
						shared[CACHEOFFSET+(i*3)+2] = shared[CACHEOFFSET+(i*3)];
					}
				}
				// Mark its children for reconstruction and reconstruct the node, if needed.
				next_it = false;
				addr = sv_step(i, false);
				pointers = shared[CACHEOFFSET+(addr*3)+2];
				if (!cached_node_is_leaf_with_global_address(pointers)) {
					// By definition, a left child stores a global address in its cache pointers.
					// Is the node not a global address stub?
					if (shared[CACHEOFFSET+(addr*3)] != EMPTYVECT32) {
						mark_cached_node_as_next_in_preparation(&shared[CACHEOFFSET+(addr*3)+2]);
						next_it = true;
					}
				}
				if (!is_required) {
					set_left_in_vectortree_node(&node, global_address(pointers));
					// Reset left cache pointer in case the left child is a stub.
					if (shared[CACHEOFFSET+(addr*3)] == EMPTYVECT32) {
						set_left_cache_pointer((shared_inttype *) &shared[CACHEOFFSET+(i*3)+2], EMPTY_CACHE_POINTER);
					}
				}
				addr = sv_step(i, true);
				// Is there actually a right child?
				if (addr != EMPTY_CACHE_POINTER) {
					pointers = shared[CACHEOFFSET+(addr*3)+2];
					if (!cached_node_is_leaf_with_global_address(pointers)) {
						if (!cached_node_contains_global_address(pointers)) {
							mark_cached_node_as_required(&shared[CACHEOFFSET+(addr*3)]);
						}
						mark_cached_node_as_next_in_preparation(&shared[CACHEOFFSET+(addr*3)+2]);
						next_it = true;
					}
				}
				if (!is_required) {
					// Store the node.
					shared[CACHEOFFSET+(i*3)] = get_left(node);
					shared[CACHEOFFSET+(i*3)+1] = get_right(node);
				}
				mark_cached_node_as_required(&shared[CACHEOFFSET+(i*3)]);
				mark_cached_node_as_old(&shared[CACHEOFFSET+(i*3)+2]);
				if (next_it) {
					// A next iteration is required.
					CONTINUE = 1;
				}
			}
		}
		__syncthreads();
	}
	// Scan the cache one more time, remove non-leaf nodes that are no longer required (alternative: keep them with their global memory addresses)
	// and reset the 'required' marks of required non-leaf nodes.
	#pragma unroll
	for (shared_indextype i = THREAD_ID; (i*3)+2 < d_shared_cache_size - CACHEOFFSET; i += BLOCK_SIZE) {
		pointers = shared[CACHEOFFSET+(i*3)+2];
		if (pointers != EMPTYVECT32) {
			if (!cached_node_is_leaf_with_global_address(pointers)) {
				if (cached_node_is_required(shared[CACHEOFFSET+(i*3)])) {
					reset_cached_node_required(&shared[CACHEOFFSET+(i*3)]);
				}
				else {
					// Delete node.
					shared[CACHEOFFSET+(i*3)] = EMPTYVECT32;
					shared[CACHEOFFSET+(i*3)+1] = EMPTYVECT32;
					shared[CACHEOFFSET+(i*3)+2] = EMPTYVECT32;
				}
			}
		}
	}
}

__global__ void __launch_bounds__(512, 2) gather(compressed_nodetype *d_q, nodetype *d_q_i, bool *d_dummy, uint8_t *d_contBFS, uint8_t *d_property_violation, volatile uint8_t *d_newstate_flags, shared_inttype *d_worktiles, const uint8_t scan) {
	uint32_t i;
	indextype l;
	shared_indextype sh_index;
	compressed_nodetype tmp;

	// Reset the shared variables preceding the cache, and reset the cache.
	if (THREAD_ID < SH_OFFSET) {
		shared[THREAD_ID] = 0;
	}
	for (i = THREAD_ID; i < (d_shared_cache_size - SH_OFFSET); i += BLOCK_SIZE) {
		shared[SH_OFFSET+i] = EMPTYVECT32;
	}
	__syncthreads();
	if (scan) {
		// Copy the work tile from global memory.
		if (THREAD_ID < OPENTILELEN + LASTSEARCHLEN) {
			shared[OPENTILEOFFSET + THREAD_ID] = d_worktiles[(OPENTILELEN+LASTSEARCHLEN+1) * BLOCK_ID + THREAD_ID];
		}
		if (THREAD_ID == 0) {
			OPENTILECOUNT = d_worktiles[(OPENTILELEN+LASTSEARCHLEN+1) * BLOCK_ID + OPENTILELEN + LASTSEARCHLEN];
		}
	}
	else if (THREAD_ID < OPENTILELEN+LASTSEARCHLEN) {
		// On first run: initialise the work tile to empty.
		shared[OPENTILEOFFSET+THREAD_ID] = 0;
	}
	__syncthreads();
	while (ITERATIONS < d_kernel_iters) {
		if (ITERATIONS > 0) {
			// Prepare the cache for the next iteration.
			PREPARE_CACHE();
		}
		__syncthreads();
		if (THREAD_ID == 0 && OPENTILECOUNT < OPENTILELEN && d_newstate_flags[BLOCK_ID] == 1) {
			// Indicate that we are scanning.
			d_newstate_flags[BLOCK_ID] = 2;
			SCAN = 1;
		}
		__syncthreads();
		// Scan the open set for work; we use OPENTILECOUNT to count retrieved elements.
		if (SCAN) {
			indextype last_search_location = (indextype) shared[LASTSEARCHOFFSET + WARP_ID];
			// This block should be able to find a new state.
			bool found_new_state = false;
			for (i = GLOBAL_WARP_ID; i < (d_hash_table_size/WARP_SIZE) && OPENTILECOUNT < OPENTILELEN; i += NR_WARPS) {
				indextype loc = last_search_location + i;
				if (loc >= d_hash_table_size/WARP_SIZE) {
					last_search_location = -i + GLOBAL_WARP_ID;
					loc = i + last_search_location;
				}
				if (loc*WARP_SIZE+LANE < d_hash_table_size) {
					tmp = d_q[loc*WARP_SIZE+LANE];
					if (is_new(tmp)) {
						found_new_state = true;
						// Try to increment the OPENTILECOUNT counter. If successful, store a reference to the state.
						l = atomicAdd((shared_inttype *) &OPENTILECOUNT, 1);
						if (l < OPENTILELEN) {
							d_q[loc*WARP_SIZE+LANE] = mark_old(tmp);
							shared[OPENTILEOFFSET+l] = set_worktile_element_to_requiring_fetching(loc*WARP_SIZE+LANE);
						}
					}
				}
			}
			if (i < (d_hash_table_size/WARP_SIZE)) {
				last_search_location = i - GLOBAL_WARP_ID;
			}
			else {
				last_search_location = 0;
			}
			if (LANE == 0) {
				shared[LASTSEARCHOFFSET + WARP_ID] = last_search_location;
			}
			if (found_new_state || i < (d_hash_table_size/WARP_SIZE)) {
				WORKSCANRESULT = 1;
			}
		}
		__syncthreads();
		// If work has been retrieved, indicate this.
		if (THREAD_ID == 0) {
			if (OPENTILECOUNT > 0) {
				(*d_contBFS) = 1;
			}
			if (SCAN && WORKSCANRESULT == 0 && d_newstate_flags[BLOCK_ID] == 2) {
				// Scanning has completed and no new states were found by this block.
				// Save this information to prevent unnecessary scanning later on.
				d_newstate_flags[BLOCK_ID] = 0;
			}
			else {
				WORKSCANRESULT = 0;
			}
		}
		if (OPENTILECOUNT > 0) {
			// Fill the cache with the newly added vector trees referred to in the work tile.
			// Create a cooperative group within a warp in which the thread resides.
			thread_block_tile<VECTOR_GROUP_SIZE> gtile = tiled_partition<VECTOR_GROUP_SIZE>(this_thread_block());

			#pragma unroll
			for (i = VECTOR_GROUP_ID; i < OPENTILECOUNT; i += NR_VECTOR_GROUPS_PER_BLOCK) {
				if (worktile_element_requires_fetching(shared[OPENTILEOFFSET+i])) {
					l = FETCH(gtile, d_q, d_q_i, shared[OPENTILEOFFSET+i]);
					if (l == CACHE_FULL) {
						// PLAN B?
					}
					else {
						sh_index = (shared_indextype) l;
					}
					if (gtile.thread_rank() == 0) {
						// Store the address to the tree in the cache in the work tile.
						shared[OPENTILEOFFSET+i] = sh_index;
					}
				}
			}
		}
		__syncthreads();
		if (GENERATE_SUCCESSORS(d_q, d_q_i, d_dummy, d_newstate_flags) == HASH_TABLE_FULL) {
			CONTINUE = 2;
		}
		bool performed_work = OPENTILECOUNT != 0;
		__syncthreads();
		// Reset the work tile count
		if (THREAD_ID == 0) {
			OPENTILECOUNT = 0;
		}
		// Start scanning the local cache and write results to the global hash table.
		if (performed_work) {
			FINDORPUT_MANY(d_q, d_q_i, d_dummy, d_newstate_flags);
		}
		__syncthreads();
		// Write 'empty' pointers to unused part of the work tile.
		if (THREAD_ID < OPENTILELEN - OPENTILECOUNT) {
			shared[OPENTILEOFFSET+OPENTILECOUNT+THREAD_ID] = EMPTYVECT32;
		}
		// Ready to start next iteration, if error has not occurred.
		if (THREAD_ID == 0) {
			if (CONTINUE == 2) {
				(*d_contBFS) = 2;
				ITERATIONS = d_kernel_iters;
			}
			else {
				ITERATIONS++;
			}
			CONTINUE = 1;
		}
		__syncthreads();
	}
	// Done. Copy the work tile to global memory.
	if (THREAD_ID < OPENTILELEN+LASTSEARCHLEN) {
		d_worktiles[(OPENTILELEN+LASTSEARCHLEN+1)*BLOCK_ID + THREAD_ID] = shared[OPENTILEOFFSET+THREAD_ID];
	}
	if (THREAD_ID == 0) {
		d_worktiles[(OPENTILELEN+LASTSEARCHLEN+1)*BLOCK_ID + OPENTILELEN + LASTSEARCHLEN] = OPENTILECOUNT;
	}
}

/**
 * Host function that prepares data, copies it to the GPU, and handles the control flow of the model checking.
 */
int main(int argc, char** argv) {
	// Size of global hash table.
	uint64_t hash_table_size;
	// Size of the internal hash table.
	uint64_t internal_hash_table_size;
	// Number of search iterations in each kernel launch.
	uint32_t kernel_iters = KERNEL_ITERS;
	// Level of verbosity (1=print level progress)
	int verbosity = 0;
	// Clock to measure time.
	clock_t start, stop;
	double runtime = 0.0;

	// Start timer.
	assert((start = clock()) != -1);

	cudaDeviceProp prop;
	int nDevices;

	// Flag to keep track of the progress and whether hash table errors occurred (value == 2).
	uint8_t contBFS, *d_contBFS;
	// Flags to track which blocks have new states.
	uint8_t *d_newstate_flags;
	// Flag to keep track of property verification outcome.
	uint8_t *d_property_violation;
	// Integer to store the amount of states counted in the hash table.
	uint64_t counted_states, *d_counted_states;
	// Space to temporarily store work tiles.
	shared_inttype *d_worktiles;

	// Global hash table.
	compressed_nodetype *d_q;
	// Internal node global hash table.
	nodetype *d_q_i;
	// Dummy flag to regulate writes to d_q_i.
	bool *d_dummy;

	const char* help_text =
		"Usage: gpuexplore [OPTIONS]\n"
		"Run state-space exploration on preprocessed SLCO model.\n"
		"options:\n"
		"  -k NUM           Run NUM iterations per kernel launch (default 1).\n"
		"  -q NUM           Allocate NUM integers for the global hash table (default fill the memory).\n"
		"  -v NUM           Change the verbosity level:\n"
		"                      0 - minimal output.\n"
		"                      1 - print sequence number of each kernel launch (search step).\n"
		"                      2 - print number of states in the global hash table after each kernel launch.\n"
		"                      3 - print global hash table content after each kernel launch.\n"
		"                      4 - print global hash table content, with pointer info, after each kernel launch.\n"
		"  -h, --help, -?   Show this help message.\n";

	int i = 1;
	while (i < argc) {
		if (!strcmp(argv[i],"--help") || !strcmp(argv[i],"-h") || !strcmp(argv[i],"-?")) {
			fprintf(stdout, "%s", help_text);
			exit(0);
		}
		else if (!strcmp(argv[i],"-k")) {
			kernel_iters = atoi(argv[i+1]);
			i += 2;
		}
		else if (!strcmp(argv[i],"-q")) {
			hash_table_size = atoi(argv[i+1]);
			i += 2;
		}
		else if (!strcmp(argv[i],"-v")) {
			verbosity = atoi(argv[i+1]);
			if (verbosity > 4) {
				verbosity = 4;
			}
			i += 2;
		}
		else {
			fprintf(stderr, "ERROR: unrecognised option %s!\n", argv[i]);
			fprintf(stdout, "%s", help_text);
			exit(1);
		}
	}

	// Set continue flag.
	contBFS = 1;
	
	// Query the device properties and determine the data structure sizes.
	cudaGetDeviceCount(&nDevices);
	if (nDevices == 0) {
		fprintf(stderr, "ERROR: No CUDA compatible GPU detected!\n");
		exit(1);
	}
	cudaGetDeviceProperties(&prop, 0);
	fprintf (stdout, "global mem: %lu\n", (uint64_t) prop.totalGlobalMem);
	fprintf (stdout, "shared mem per block: %d\n", (int) prop.sharedMemPerBlock);
	fprintf (stdout, "shared mem per SM: %d\n", (int) prop.sharedMemPerMultiprocessor);
	fprintf (stdout, "max. threads per block: %d\n", (int) prop.maxThreadsPerBlock);
	fprintf (stdout, "max. grid size: %d\n", (int) prop.maxGridSize[0]);
	fprintf (stdout, "nr. of multiprocessors: %d\n", (int) prop.multiProcessorCount);

	// Determine actual number of blocks.
	uint32_t nblocks = MAX(1,MIN(prop.maxGridSize[0], NR_BLOCKS));

	// Allocate memory on the GPU.
	cudaMallocCount((void **) &d_contBFS, sizeof(uint8_t));
	cudaMallocCount((void **) &d_property_violation, sizeof(uint8_t));
	cudaMallocCount((void **) &d_counted_states, sizeof(uint64_t));
	cudaMallocCount((void **) &d_newstate_flags, nblocks * sizeof(uint8_t));
	cudaMallocCount((void **) &d_worktiles, nblocks * (OPENTILELEN+LASTSEARCHLEN+1) * sizeof(shared_inttype));
	cudaMallocCount((void **) &d_dummy, sizeof(bool));

	// Set data on the GPU to initial values.
	CUDA_CHECK_RETURN(cudaMemset(d_contBFS, 1, sizeof(uint8_t)));
	CUDA_CHECK_RETURN(cudaMemset(d_newstate_flags, 0, nblocks * sizeof(uint8_t)));
	CUDA_CHECK_RETURN(cudaMemset(d_worktiles, 0, nblocks * (OPENTILELEN + LASTSEARCHLEN + 1) * sizeof(shared_inttype)));
	CUDA_CHECK_RETURN(cudaMemset(d_counted_states, 0, sizeof(uint64_t)));

	// We create a global compact hash table for 24 GB. A root table is created that has exactly 2^32 elements, and an internal table is created with 500 million elements.
	hash_table_size = 4294967296;
	cudaMalloc((void **)&d_q, hash_table_size * sizeof(compressed_nodetype));
	internal_hash_table_size = 536870912;
	cudaMalloc((void **)&d_q_i, internal_hash_table_size * sizeof(nodetype));

	fprintf (stdout, "Global mem hash table size: %lu; Number of entries: %lu\n", hash_table_size*sizeof(compressed_nodetype),  hash_table_size);
	fprintf (stdout, "Internal global mem hash table size: %lu; Number of entries: %lu\n", internal_hash_table_size*sizeof(nodetype), internal_hash_table_size);

	shared_inttype shared_cache_size = (shared_inttype) prop.sharedMemPerMultiprocessor / sizeof(shared_inttype) / 2;
	fprintf (stdout, "Shared mem work tile size: 256\n");
	fprintf (stdout, "Shared mem cache size: %u; Number of entries: %u\n", (uint32_t) (shared_cache_size*sizeof(shared_inttype)), (uint32_t) shared_cache_size*3);
	fprintf (stdout, "Nr. of blocks: %d; Block size: 512; Nr. of kernel iterations: %d\n", nblocks, kernel_iters);

	// Copy symbols.
	cudaMemcpyToSymbol(d_shared_cache_size, &shared_cache_size, sizeof(shared_inttype));
	cudaMemcpyToSymbol(d_kernel_iters, &kernel_iters, sizeof(uint32_t));
	cudaMemcpyToSymbol(d_internal_hash_table_size, &internal_hash_table_size, sizeof(uint64_t));
	cudaMemcpyToSymbol(d_hash_table_size, &hash_table_size, sizeof(uint64_t));

	// Initialise the hash table.
	init_hash_table<<<NR_BLOCKS, 512>>>(d_q, d_q_i);
	CUDA_CHECK_RETURN(cudaDeviceSynchronize());
	store_initial_state<<<1, 512, shared_cache_size * sizeof(shared_inttype)>>>(d_q, d_q_i, d_dummy, d_newstate_flags, d_worktiles);
	CUDA_CHECK_RETURN(cudaDeviceSynchronize());

	compressed_nodetype *q_test;
	nodetype *q_i_test;
	if (verbosity >= 3) {
		q_test = (compressed_nodetype*) malloc(sizeof(compressed_nodetype)*hash_table_size);
		q_i_test = (nodetype*) malloc(sizeof(nodetype)*internal_hash_table_size);
	}

	uint32_t iterations_counter = 0;
	uint8_t scan = 1;
	CUDA_CHECK_RETURN(cudaMemset(d_property_violation, 0, sizeof(uint8_t)));
	uint8_t property_violation = 0;

	clock_t exploration_start;
	assert((exploration_start = clock()) != -1);

	while (contBFS == 1) {
		CUDA_CHECK_RETURN(cudaMemset(d_contBFS, 0, sizeof(uint8_t)));
		// TODO: change nr of blocks back to nblocks
		gather<<<1, 512, shared_cache_size * sizeof(shared_inttype)>>>(d_q, d_q_i, d_dummy, d_contBFS, d_property_violation, d_newstate_flags, d_worktiles, scan);

		// Copy progress result.
		CUDA_CHECK_RETURN(cudaDeviceSynchronize());
		CUDA_CHECK_RETURN(cudaMemcpy(&contBFS, d_contBFS, sizeof(uint8_t), cudaMemcpyDeviceToHost));
		// if (check_property > 0) {
		// }
		if (verbosity > 0) {
			if (verbosity == 1) {
				fprintf(stdout, "%d\n", iterations_counter++);
			}
			else if (verbosity == 2) {
				cudaMemset(d_counted_states, 0, sizeof(uint64_t));
				count_states<<<((int) prop.multiProcessorCount)*8, 512>>>(d_q, d_counted_states);
				CUDA_CHECK_RETURN(cudaDeviceSynchronize());
				CUDA_CHECK_RETURN(cudaMemcpy(&counted_states, d_counted_states, sizeof(uint64_t), cudaMemcpyDeviceToHost));
				fprintf(stdout, "Nr. of states in hash table: %lu\n", counted_states);
			}
			else if (verbosity == 3) {
				cudaMemcpy(q_test, d_q, hash_table_size * sizeof(compressed_nodetype), cudaMemcpyDeviceToHost);
				cudaMemcpy(q_i_test, d_q_i, internal_hash_table_size * sizeof(nodetype), cudaMemcpyDeviceToHost);
				print_content_hash_table(stdout, q_test, q_i_test, hash_table_size, internal_hash_table_size, false);
			}
			else if (verbosity == 4) {
				cudaMemcpy(q_test, d_q, hash_table_size * sizeof(compressed_nodetype), cudaMemcpyDeviceToHost);
				cudaMemcpy(q_i_test, d_q_i, internal_hash_table_size * sizeof(nodetype), cudaMemcpyDeviceToHost);
				print_content_hash_table(stdout, q_test, q_i_test, hash_table_size, internal_hash_table_size, true);
			}
		}
		scan = 1;
	}

	// Determine runtime.
	stop = clock();
	runtime = (double) (stop-start)/CLOCKS_PER_SEC;
	fprintf(stdout, "Run time: %f\n", runtime);
	runtime = (double) (stop-exploration_start)/CLOCKS_PER_SEC;
	fprintf(stdout, "Exploration time: %f\n", runtime);

	// TODO: Property violation report

	// Report hash table error if required.
	if (contBFS == 2) {
		fprintf(stderr, "ERROR: problem with hash table!\n");
	}

	CUDA_CHECK_RETURN(cudaMemset(d_counted_states, 0, sizeof(uint64_t)));
	count_states<<<((int) prop.multiProcessorCount)*8, 512>>>(d_q, d_counted_states);
	CUDA_CHECK_RETURN(cudaDeviceSynchronize());
	CUDA_CHECK_RETURN(cudaMemcpy(&counted_states, d_counted_states, sizeof(uint64_t), cudaMemcpyDeviceToHost));
	fprintf(stdout, "Nr. of states in hash table: %lu\n", counted_states);

	return 0;
}