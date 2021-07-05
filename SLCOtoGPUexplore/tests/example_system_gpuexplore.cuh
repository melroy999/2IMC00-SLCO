#include <stdbool.h>
#include <cooperative_groups.h>
#include <vector>
#include <algorithm>
using namespace cooperative_groups;

// test macros
#define PRINTTHREADID()						{printf("Hello thread %d\n", (blockIdx.x*blockDim.x)+threadIdx.x);}
#define PRINTTHREAD(j, i)					{printf("%d: Seen by thread %d: %d\n", (j), (blockIdx.x*blockDim.x)+threadIdx.x, (i));}

// Structure of the state vector:
// [ one bit reserved, state globalObject'P_0: 3 bit(s), state globalObject'P_4: 3 bit(s), state globalObject'P_2: 3 bit(s), state globalObject'P_3: 3 bit(s), 
//   state globalObject'P_1: 3 bit(s), variable globalObject'P_3'myplace: 8 bit(s), variable globalObject'P_0'myplace: 8 bit(s), variable globalObject'next: 8 bit(s), variable globalObject'Slot[0]: 8 bit(s), 
//   variable globalObject'Slot[1]: 8 bit(s), variable globalObject'Slot[2]: 8 bit(s) ],
// [ one bit reserved, variable globalObject'Slot[3]: 8 bit(s), variable globalObject'Slot[4]: 8 bit(s), variable globalObject'P_1'myplace: 8 bit(s), variable globalObject'P_4'myplace: 8 bit(s), 
//   variable globalObject'P_2'myplace: 8 bit(s) ]

// type of vectortree nodes used.
#define nodetype uint64_t
#define compressed_nodetype uint32_t
// type of global memory indices used.
#define indextype uint32_t
// type of shared memory elements used.
#define shared_inttype uint32_t
// type for shared memory cache indices.
#define shared_indextype uint16_t
// type of state machine state.
#define statetype uint8_t
// types of data elements.
#define elem_inttype int32_t
#define elem_chartype uint8_t
#define elem_booltype bool
// type for array and channel buffer indexing.
#define array_indextype int8_t
// type for indexing in variable buffers.
#define buffer_indextype int8_t
// type for vector node IDs.
#define vectornode_indextype uint8_t

// *******************************************************************************

// GPU constants.
static const int WARP_SIZE = 32;
__constant__ uint32_t d_kernel_iters;
__constant__ shared_inttype d_shared_size;
__constant__ uint64_t d_hash_table_size;
__constant__ uint64_t d_internal_hash_table_size;

// GPU configuraton.
static const int KERNEL_ITERS = 1;
static const int NR_BLOCKS = 1;

// Thread ids and dimensions.
#define GRID_SIZE 					gridDim.x
#define BLOCK_SIZE					blockDim.x
#define NR_THREADS					(GRID_SIZE * BLOCK_SIZE)

#define BLOCK_ID					blockIdx.x
#define THREAD_ID 					threadIdx.x
#define GLOBAL_THREAD_ID			((BLOCK_ID * BLOCK_SIZE) + THREAD_ID)
#define WARP_ID						(THREAD_ID / WARP_SIZE)
#define GLOBAL_WARP_ID				(((BLOCK_SIZE / WARP_SIZE) * BLOCK_ID) + WARP_ID)
#define NR_WARPS_PER_BLOCK			(BLOCK_SIZE / WARP_SIZE)
#define NR_WARPS					(NR_WARPS_PER_BLOCK * GRID_SIZE)
#define LANE						(THREAD_ID & 0x0000001F)
#define VECTOR_GROUP_SIZE			2
#define VECTOR_GROUP_ID				(THREAD_ID / VECTOR_GROUP_SIZE)
#define NR_VECTOR_GROUPS_PER_BLOCK	(BLOCK_SIZE / VECTOR_GROUP_SIZE)

// Constant representing empty array index entry.
#define EMPTY_INDEX -1
// Constant used to initialise state variables.
#define NO_STATE 5
#define EMPTYVECT32					0x3FFFFFFF
#define EMPTYVECT16					0xFFFF
#define CACHE_POINTERS_NEW_LEAF		0x1FFFFFFF
// A cache never contains 2^16-1 or more elements, hence 2^16-1 is available for the constant.
#define EMPTY_CACHE_POINTER			0x7FFF
#define EMPTY_HASH_POINTER 			0xFFFFFFFF

// Evict constant to determine number of allowed evictions for element insertion.
#define EVICTFREQ 10
// Linear probing parameter: number of probing attempts before the hash table is considered full.
#define LINEARPROBINGFREQ 20
#define NR_HASH_FUNCTIONS 32
// Number of retries in local cache.
#define CACHERETRYFREQ 40

// Enum type to indicate node storage mode (see the 'explore_' functions).
enum Storage_mode {TO_CACHE, TO_GLOBAL, STORED, GLOBAL_STORED, HASH_TABLE_FULL};

// CONSTANTS FOR SHARED MEMORY CACHES
// Offsets calculations for shared memory arrays
#define OPENTILELEN					102
#define LASTSEARCHLEN				16 // (GATHER KERNEL BLOCK_SIZE/WARP_SIZE)

// Offsets in shared memory from which loaded data can be read.
#define SH_OFFSET 4
#define OPENTILEOFFSET 				(SH_OFFSET)
#define LASTSEARCHOFFSET			(OPENTILEOFFSET+OPENTILELEN+OPENTILELEN)
#define CACHEOFFSET 				(LASTSEARCHOFFSET+LASTSEARCHLEN)

// Shared memory work tile size in nr. of warps
#define OPENTILE_WARP_WIDTH			4

// Error value to indicate a full global hash table.
#define HASHTABLE_FULL 				0xFFFFFFFF
// Error value to indicate that a shared memory cache is full.
#define CACHE_FULL 				0xFFFFFFFF

// Shared memory local progress flags
#define ITERATIONS					(shared[0])
#define CONTINUE					(shared[1])
#define OPENTILECOUNT				(shared[2])
#define SCAN						(shared[3])

// The number of state machines in the model.
#define NR_SMS						5

// CONSTANTS FOR GLOBAL MEMORY HASH TABLE
// Empty root hash table element
#define EMPTY_COMPRESSED_NODE		0xFFFFFFFF
// Empty internal hash table element (exploits that an uncompressed internal vectornode always has its highest bit set to 0)
#define EMPTY_NODE					0xFFFFFFFFFFFFFFFF

// GPU shared memory array.
extern __shared__ volatile shared_inttype shared[];

// Bitmask to identify parts of the state vector that contain state machine states.
#define VECTOR_SMPARTS			0x80000000

// *** START BIT OPERATIONS ***

// Bit right shift function.
inline __host__ __device__ uint64_t rshft(const uint64_t x, uint8_t i) {
	return (x >> i);
}

// Bit left shift function for 58 bits.
inline __host__ __device__ uint64_t lshft_58(const uint64_t x, uint8_t i) {
	uint64_t y = (x << i);
	return y & 0x3ffffffffffffff;
}

// Multiplication modulo 2^58.
inline __host__ __device__ uint64_t mult_58(const uint64_t x, uint64_t a) {
	return ((x * a) & 0x3ffffffffffffff);
}

// XOR two times bit shift function for 58 bits.
inline __host__ __device__ uint64_t xor_shft2_58(const uint64_t x, uint8_t a, uint8_t b) {
	uint64_t y = (x ^ lshft_58(x,a));
	y = (y ^ rshft(y,b));
	return y;
}

// Bit left shift function for 64 bits.
inline __host__ __device__ uint64_t lshft_64(const uint64_t x, uint8_t i) {
	return (x << i);
}

// XOR two times bit shift function for 64 bits.
inline __host__ __device__ uint64_t xor_shft2_64(uint64_t x, uint8_t a, uint8_t b) {
	uint64_t y = (x ^ lshft_64(x,a));
	y = (y ^ rshft(x,b));
	return y;
}

// *** END BIT OPERATIONS ***

// *** START FUNCTIONS FOR VECTOR TREE NODE MANIPULATION AND STORAGE TO THE SHARED MEMORY CACHE ***

// The highest bit of a state encodes that the state is new.

// Mark state as new or old.
inline __device__ compressed_nodetype mark_new(compressed_nodetype node) {
	return node | 0x80000000;
}

inline __device__ compressed_nodetype mark_old(compressed_nodetype node) {
	return node & 0x7FFFFFFF;
}

// Check whether state is new.
// This is the case if the highest bit is set, and not all bits are set.
inline __host__ __device__ bool is_new(compressed_nodetype node) {
	return (node != EMPTY_COMPRESSED_NODE && (node & 0x80000000) == 0x80000000);
}

// Mark a node as root.
inline __device__ void mark_root(volatile nodetype *node) {
	*node = *node | 0x4000000000000000;
}

// Check whether node is root.
inline __host__ __device__ bool is_root(nodetype node) {
	return (node & 0x4000000000000000) != 0;
}

// Mark a node in the cache as old.
inline __device__ void mark_cached_node_as_old(volatile shared_inttype *pointers) {
	*pointers = *pointers & 0x3FFFFFFF;
}

// A cached node is marked new non-leaf (non-root) by setting the highest two bits
// of the cache pointers to '10'.
inline __device__ void mark_cached_node_new_nonleaf(volatile shared_inttype *pointers) {
	*pointers = (*pointers & 0x3FFFFFFF) | 0x80000000;
}

inline __device__ bool cached_node_is_new_nonleaf(shared_inttype pointers) {
	return (pointers & 0xC0000000) == 0x80000000;
}

inline __device__ bool cached_node_is_new_leaf(shared_inttype pointers) {
	return pointers == CACHE_POINTERS_NEW_LEAF;
}

// The cache pointers of a node are assigned a global hash table address by setting the highest two bits
// of the cache pointers either to '01' or '11', and setting the remaining bits to the address.
// The use of both '01' and '11' is to be able to distinguish a leaf with a global address ('01') from a
// non-leaf with a global address ('11'). After this function is called,
// when storing non-leaf nodes in the global hash table, the original cache pointers are stored in the first half of the
// node itself, to allow efficient follow-up iterations in the successor generation procedure. Reconstructing the node
// by means of cache pointers (see PREPARE_CACHE()) is faster than having to reconstruct cache pointers.
inline __device__ void set_cache_pointers_to_global_address(volatile shared_inttype *pointers, indextype addr, bool is_leaf) {
	// The highest two bits in pointers are set to indicate that it now stores a global memory address.
	if (is_leaf) {
		*pointers = (addr & 0x3FFFFFFF) | 0x40000000;
	}
	else {
		*pointers = addr | 0xC0000000;
	}
}

inline __device__ bool cached_node_is_leaf_with_global_address(shared_inttype pointers) {
	return (pointers & 0xC0000000) == 0x40000000;
}

inline __device__ bool cached_node_is_nonleaf_with_global_address(shared_inttype pointers) {
	return (pointers & 0xC0000000) == 0xC0000000;
}

inline __device__ bool cached_node_contains_global_address(shared_inttype pointers) {
	return (cached_node_is_leaf_with_global_address(pointers) || cached_node_is_nonleaf_with_global_address(pointers));
}

// Extract the global address from cache pointers.
inline __device__ nodetype global_address(shared_inttype pointers) {
	return (pointers & 0x3FFFFFFF);
}

inline __device__ bool cached_node_is_new(shared_inttype pointers) {
	return (cached_node_is_new_nonleaf(pointers) || cached_node_is_new_leaf(pointers));
}

// To prepare the cache for another successor generation iteration, we use the highest two bits of the cache pointers of nodes
// again. The two bits set to 1 again indicate a global hash table address is stored in the remaining bits. '00' again indicates
// an old non-leaf element. '10' is now used to indicate that the node can update, i.e., reconstruct itself by checking its children,
// and '01' indicates that a node has reconstructed itself. Note that the latter code coincides with a leaf having a global address.
// This is intentional, as such leaves do not require reconstruction.

inline __device__ bool cached_node_is_next_in_preparation(shared_inttype pointers) {
	return (pointers & 0xC0000000) == 0x80000000;
}

inline __device__ void mark_cached_node_as_next_in_preparation(volatile shared_inttype *pointers) {
	*pointers = (*pointers & 0x3FFFFFFF) | 0x80000000;
}

inline __device__ void mark_cached_node_as_prepared(volatile shared_inttype *pointers) {
	*pointers = (*pointers & 0x3FFFFFFF) | 0x40000000;
}

// In part 1 of a cached node, we set the highest bit in case the (non-leaf) node needs to be kept in the cache for future successor generation.
// (This is used when preparing the cache to distinguish nodes that are still needed from nodes that are not).
inline __device__ void mark_cached_node_as_required(volatile shared_inttype *part1) {
	*part1 = (*part1) | 0x80000000;
}

inline __device__ void reset_cached_node_required(volatile shared_inttype *part1) {
	*part1 = (*part1) & 0x7FFFFFFF;
}

inline __device__ bool cached_node_is_required(shared_inttype pointers) {
	return (pointers & 0x80000000) == 0x80000000;
}

// Filter the bookkeeping bit values from the given node.
inline __device__ __host__ nodetype filter_bookkeeping(nodetype node) {
	return node & 0x3FFFFFFFFFFFFFFF;
}

// Filter the bookkeeping bit values from the given compressed node.
inline __device__ compressed_nodetype filter_compressed_bookkeeping(compressed_nodetype node) {
	return node & 0x7FFFFFFF;
}

// Function to traverse one step in state vector tree (stored in shared memory).
inline __device__ shared_indextype sv_step(shared_indextype node_index, bool goright) {
	shared_indextype index = 0;
	shared_inttype tmp = shared[CACHEOFFSET+(node_index*3)+2];
	if (!goright) {
		asm("{\n\t"
			" .reg .u32 t1;\n\t"
			" bfe.u32 t1, %1, 15, 15;\n\t"
			" cvt.u16.u32 %0, t1;\n\t"
			"}" : "=h"(index) : "r"(tmp));
	}
	else {
		asm("{\n\t"
			" .reg .u32 t1;\n\t"
			" bfe.u32 t1, %1, 0, 15;\n\t"
			" cvt.u16.u32 %0, t1;\n\t"
			"}" : "=h"(index) : "r"(tmp));
	}
	return index;
}

// Get left or right half of a 64-bit integer
inline __device__ uint32_t get_left(uint64_t node) {
	uint32_t result;
	asm("{\n\t"
		" .reg .u64 t1;\n\t"
		" bfe.u64 t1, %1, 32, 32;\n\t"
		" cvt.u32.u64 %0, t1;\n\t"
		"}" : "=r"(result) : "l"(node));
	return result;
}

inline __device__ uint32_t get_right(uint64_t node) {
	uint32_t result;
	asm("{\n\t"
		" cvt.u32.u64 %0, %1;\n\t"
		"}" : "=r"(result) : "l"(node));
	return result;
}

// Combine two halfs of a 64-bit integer
inline __device__ uint64_t combine_halfs(uint32_t n1, uint32_t n2) {
	uint64_t node = (uint64_t) n2;
	uint64_t node2 = (uint64_t) n1;
	asm("{\n\t"
		" bfi.b64 %0, %1, %0, 32, 32;\n\t"
		"}" : "+l"(node) : "l"(node2));
	return node;
}

// Host (CPU) version.
inline __host__ uint64_t host_combine_halfs(uint32_t n1, uint32_t n2) {
	uint64_t node = (uint64_t) n1;
	node = node << 32;
	node = node | n2;
	return node;
}

inline __device__ nodetype get_vectorpart_0(shared_indextype node_index) {
	shared_indextype index = node_index;
	index = sv_step(index, false);
	nodetype part;
	asm("{\n\t"
		" mov.b64 %0,{ %1, %2 };\n\t"
		"}" : "=l"(part) : "r"(shared[CACHEOFFSET+(index*3)+1]), "r"(shared[CACHEOFFSET+(index*3)]));
	return part;
}

inline __device__ nodetype get_vectorpart_1(shared_indextype node_index) {
	shared_indextype index = node_index;
	index = sv_step(index, true);
	nodetype part;
	asm("{\n\t"
		" mov.b64 %0,{ %1, %2 };\n\t"
		"}" : "=l"(part) : "r"(shared[CACHEOFFSET+(index*3)+1]), "r"(shared[CACHEOFFSET+(index*3)]));
	return part;
}

// Retrieval functions for vector parts from shared memory.
// We ignore shared memory node pointers (cache pointers).
inline __device__ nodetype get_vectorpart(shared_indextype node_index, vectornode_indextype part_id) {
	switch (part_id) {
	  case 0:
	  	return get_vectorpart_0(node_index);
	  case 1:
	  	return get_vectorpart_1(node_index);
	  default:
	  	return 0;
	}
}

inline __device__ void get_vectortree_node_0(nodetype *node, shared_inttype *d_cachepointers, shared_indextype node_index) {
	asm("{\n\t"
		" mov.b64 %0,{ %1, %2 };\n\t"
		"}" : "=l"(*node) : "r"(shared[CACHEOFFSET+(node_index*3)+1]), "r"(shared[CACHEOFFSET+(node_index*3)]));
	*d_cachepointers = shared[CACHEOFFSET+(node_index*3)+2];
}

inline __device__ void get_vectortree_node_1(nodetype *node, shared_inttype *d_cachepointers, shared_indextype node_index) {
	shared_indextype index = node_index;
	index = sv_step(index, false);
	asm("{\n\t"
		" mov.b64 %0,{ %1, %2 };\n\t"
		"}" : "=l"(*node) : "r"(shared[CACHEOFFSET+(index*3)+1]), "r"(shared[CACHEOFFSET+(index*3)]));
	*d_cachepointers = shared[CACHEOFFSET+(index*3)+2];
}

inline __device__ void get_vectortree_node_2(nodetype *node, shared_inttype *d_cachepointers, shared_indextype node_index) {
	shared_indextype index = node_index;
	index = sv_step(index, true);
	asm("{\n\t"
		" mov.b64 %0,{ %1, %2 };\n\t"
		"}" : "=l"(*node) : "r"(shared[CACHEOFFSET+(index*3)+1]), "r"(shared[CACHEOFFSET+(index*3)]));
	*d_cachepointers = shared[CACHEOFFSET+(index*3)+2];
}

// Retrieval functions for vector tree nodes from shared memory, including shared memory node pointers (cache pointers).
inline __device__ void get_vectortree_node(nodetype *node, shared_inttype *d_cachepointers, shared_indextype node_index, vectornode_indextype i) {
	switch (i) {
	  case 0:
	  	get_vectortree_node_0(node, d_cachepointers, node_index);
	  	break;
	  case 1:
	  	get_vectortree_node_1(node, d_cachepointers, node_index);
	  	break;
	  case 2:
	  	get_vectortree_node_2(node, d_cachepointers, node_index);
	  	break;
	  default:
	  	return;
	}
}

// Cache pointers set functions.
inline __device__ void set_left_cache_pointer(shared_inttype *pointers, shared_indextype new_pointer) {
	shared_inttype t1 = (shared_inttype) new_pointer;
	asm("{\n\t"
		" bfi.b32 %0, %1, %0, 15, 15;\n\t"
		"}" : "+r"(*pointers) : "r"(t1));
}

inline __device__ void set_right_cache_pointer(shared_inttype *pointers, shared_indextype new_pointer) {
	shared_inttype t1 = (shared_inttype) new_pointer;
	asm("{\n\t"
		" bfi.b32 %0, %1, %0, 0, 15;\n\t"
		"}" : "+r"(*pointers) : "r"(t1));
}

// Vectornode reset functions.
inline __device__ void reset_left_in_vectortree_node(nodetype *node) {
	asm("{\n\t"
		" bfi.b64 %0, 0xFFFFFFFFFFFFFFFF, %0, 31, 31;\n\t"
		"}" : "+l"(*node));
}

inline __device__ void reset_right_in_vectortree_node(nodetype *node) {
	asm("{\n\t"
		" bfi.b64 %0, 0xFFFFFFFFFFFFFFFF, %0, 0, 31;\n\t"
		"}" : "+l"(*node));
}

// Vectornode set functions.
inline __device__ void set_left_in_vectortree_node(nodetype *node, indextype address) {
	nodetype t1 = (nodetype) global_address(address);
	asm("{\n\t"
		" bfi.b64 %0, %1, %0, 31, 31;\n\t"
		"}" : "+l"(*node) : "l"(t1));
}

inline __device__ void set_right_in_vectortree_node(nodetype *node, indextype address) {
	nodetype t1 = (nodetype) global_address(address);
	asm("{\n\t"
		" bfi.b64 %0, %1, %0, 0, 31;\n\t"
		"}" : "+l"(*node) : "l"(t1));
}

// Vectornode get functions.
inline __device__ indextype get_pointer_from_vectortree_node(nodetype node, bool choice) {
	indextype result;
	asm("{\n\t"
		" .reg .u64 t1;\n\t"
		" bfe.u64 t1, %1, %2, 31;\n\t"
		" cvt.u32.u64 %0, t1;\n\t"
		"}" : "=r"(result) : "l"(node), "r"((1-(choice ? 1 : 0))*31));
	return result;
}

// Functions to collapse and expand 64-bit nodes to/from 58-bit nodes.

// Create a node in which the two pointers in the given node are stored each in 29 bits.
// Precondition: the two pointers fit in 29 bits each, and are stored in the input each in 31 bits.
inline __device__ nodetype collapse(nodetype node) {
	nodetype newnode = filter_bookkeeping(node);
	newnode = (newnode >> 31) << 29;
	newnode = newnode | (node & 0x1fffffff);
	return newnode;
}

// Create a node in which the two pointers in the given node are stored each in 31 bits.
// Precondition: the two pointers in the input are stored in 29 bits each.
inline __device__ __host__ nodetype expand(nodetype node) {
	nodetype newnode = (node >> 29) << 31;
	newnode = newnode | (node & 0x1fffffff);
	return newnode;
}

// Function to traverse one step in state vector tree (stored in global memory).
inline __device__ nodetype direct_sv_step(compressed_nodetype *d_q, nodetype *d_q_i, nodetype node, bool goright) {
	indextype index = get_pointer_from_vectortree_node(node, goright);
	return d_q_i[index];
}

// Host (CPU) version of get_pointer_from_vectortree_node.
inline __host__ indextype host_get_pointer_from_vectortree_node(nodetype node, bool choice) {
	if (choice) {
		return (indextype) (node & 0x7fffffff);
	}
	else {
		return (indextype) ((node & 0x3fffffff80000000) >> 31);
	}
}

// Host (CPU) version of direct_sv_step.
inline __host__ nodetype host_direct_sv_step(compressed_nodetype *q, nodetype *q_i, nodetype node, bool goright, FILE* stream, bool print_pointers) {
	indextype index = host_get_pointer_from_vectortree_node(expand(node), goright);
	if (print_pointers) {
		fprintf(stream, "Navigating node with value %lu.\n", node);
		if (goright) {
			fprintf(stream, "Navigating from root to right child in internal hash table located at %u.\n", index);
		}
		else {
			fprintf(stream, "Navigating from root to left child in internal hash table located at %u.\n", index);			
		}
	}
	return q_i[index];
}

// Functions to retrieve vector parts from global memory.
inline __device__ nodetype direct_get_vectorpart_0(compressed_nodetype *d_q, nodetype *d_q_i, nodetype node) {
	nodetype tmp = node;
	tmp = direct_sv_step(d_q, d_q_i, tmp, false);
	return tmp;
}

inline __device__ nodetype direct_get_vectorpart_1(compressed_nodetype *d_q, nodetype *d_q_i, nodetype node) {
	nodetype tmp = node;
	tmp = direct_sv_step(d_q, d_q_i, tmp, true);
	return tmp;
}

// Functions to retrieve vector parts from host memory.
inline __host__ nodetype host_direct_get_vectorpart_0(compressed_nodetype *q, nodetype *q_i, nodetype node, FILE* stream, bool print_pointers) {
	nodetype tmp = node;
	tmp = host_direct_sv_step(q, q_i, tmp, false, stream, print_pointers);
	return tmp;
}

inline __host__ nodetype host_direct_get_vectorpart_1(compressed_nodetype *q, nodetype *q_i, nodetype node, FILE* stream, bool print_pointers) {
	nodetype tmp = node;
	tmp = host_direct_sv_step(q, q_i, tmp, true, stream, print_pointers);
	return tmp;
}

// Vectornode check for a left or right pointer gap.
inline __device__ bool vectortree_node_contains_left_gap(nodetype node) {
	return (node & 0x3fffffff80000000) == 0x3fffffff80000000;
}

inline __device__ bool vectortree_node_contains_right_gap(nodetype node) {
	return (node & 0x7fffffff) == 0x7fffffff;
}

// Cache hash function.
inline __device__ shared_indextype CACHE_HASH(nodetype node) {
	uint64_t node1 = xor_shft2_64((uint64_t) node, 38, 14);
	node1 ^= 0xD1B54A32D192ED03L;
	node1 *= 0xAEF17502108EF2D9L;
	node1 = xor_shft2_64(node1, 12, 52);
	node1 = xor_shft2_64(node1, 37, 27);
	node1 *= 0xd1b549a75a913001L;
	node1 ^= rshft(node1, 43);
	node1 ^= rshft(node1, 31);
	node1 ^= rshft(node1, 23);
	node1 *= 0xdb4f0ad2012a3801L;
	node1 ^= rshft(node1, 28);
	// Make sure the resulting address is within the range of the cache.
	return (shared_indextype) (node1 & 0x000000000000FFFF) % 4021;
}

// Store a vectortree node in the cache.
// Return address if successful, HASHTABLE_FULL if cache is full.
inline __device__ indextype STOREINCACHE(nodetype node, shared_inttype cache_pointers, bool is_leaf) {
	uint8_t i = 0;
	shared_indextype addr;
	shared_inttype element;
	shared_inttype part1, part2;

	// Split the node in two.
	part1 = get_left(node);
	part2 = get_right(node);
	// As non-leafs often have the same initial node value (if two non-leafs both have two new child trees, their node values are identical),
	// we hash on the cache_pointers for non-leafs, which usually are different, as they are related to the reachable leafs.
	if (is_leaf) {
		addr = CACHE_HASH(node);
	}
	else {
		addr = CACHE_HASH((nodetype) cache_pointers);
	}
	while (i < CACHERETRYFREQ) {
		element = atomicCAS((shared_inttype *) &(shared[CACHEOFFSET+(addr*3)]), EMPTYVECT32, part1);
		if (element == EMPTYVECT32 || element == part1) {
			// Successful storage.
			element = atomicCAS((shared_inttype *) &(shared[CACHEOFFSET+(addr*3)+1]), EMPTYVECT32, part2);
			if (element == EMPTYVECT32 || element == part2) {
				// Successful storage.
				element = atomicCAS((shared_inttype *) &(shared[CACHEOFFSET+(addr*3)+2]), EMPTYVECT32, cache_pointers);
				if (element == EMPTYVECT32 || element == cache_pointers || (cache_pointers == CACHE_POINTERS_NEW_LEAF && cached_node_contains_global_address(element))) {
					// Storage of node successful.
					return addr;
				}
				else {
					// Storage of node after all not successful. Try another address.
					addr++;
					if ((addr*3)+2 >= (d_shared_size - CACHEOFFSET)) {
						addr = 0;
					}
					i++;
					continue;
				}
			}
			else {
				// Storage of node after all not successful. Try another address.
				addr++;
				if ((addr*3)+2 >= (d_shared_size - CACHEOFFSET)) {
					addr = 0;
				}
				i++;
				continue;
			}
		}
		else {
			// Storage of node after all not successful. Try another address.
			addr++;
			if ((addr*3)+2 >= (d_shared_size - CACHEOFFSET)) {
				addr = 0;
			}
			i++;		
		}
	}
	// Storage of node not successful. We conclude that the cache is full.
	return CACHE_FULL;
}

// Create a 'stub' for storage of a global address, and store it in the cache.
inline __device__ shared_indextype store_global_address_stub(nodetype node, shared_indextype i) {
	// Construct cache pointers to store the global address.
	shared_inttype n = 0;
	set_cache_pointers_to_global_address(&n, get_pointer_from_vectortree_node(node, false), false);
	// Create a node for storage of the stub. Only the global address, stored as cache pointers, is
	// actually relevant. The node is created to be able to dinstinguish a stub from a non-leaf. To do so, the highest bits of the node
	// are set to EMPTYVECT32, which is a value those bits can never have when the node is a non-leaf. The lowest bits are set to the
	// global address, to achieve variation in values. This helps the hashing.
	shared_indextype addr = STOREINCACHE(combine_halfs(EMPTYVECT32, n), n, true);
	return addr;
}

// *** END FUNCTIONS FOR VECTOR TREE NODE MANIPULATION AND STORAGE TO THE SHARED MEMORY CACHE ***

// *** START FUNCTIONS FOR MODEL DATA RETRIEVAL AND STORAGE ***

// Auxiliary functions to check for and obtain/store an array element with an index equal to the given expression e.
// There are functions for the various buffer sizes required to interpret the model.

// Store the given value v under index e. Check for presence of e in the index buffer. If not present, store e and v.
// Precondition: if e is not already present, there is space in the buffer to store it.
template<class T>
inline __device__ void A_STR_1(array_indextype *idx_0, T *v_0, array_indextype e, T v) {
	if (((array_indextype) e) == *idx_0) {
		*v_0 = v;
		return;
	}
	else if (*idx_0 == EMPTY_INDEX) {
		*idx_0 = (array_indextype) e;
		*v_0 = v;
		return;
	}
}

// Return the value stored at index e.
// Precondition: provided array contains the requested element.
template<class T>
inline __device__ T A_LD_1(array_indextype idx_0, T v_0, array_indextype e) {
	if (((array_indextype) e) == idx_0) {
		return v_0;
	}
	return T();
}

// Check whether the given array index e is stored in the given array index buffer.
inline __device__ bool A_IEX_1(array_indextype idx_0, array_indextype e) {
	if (((array_indextype) e) == idx_0) {
		return true;
	}
	return false;
}

// GPU data retrieval functions. Retrieve particular state info from the given state vector part(s).
// Precondition: the given parts indeed contain the requested info.
inline __device__ void get_globalObject_P_0(statetype *b, nodetype part1, nodetype part2) {
	uint16_t t2;
	asm("{\n\t"
		" .reg .u64 t1;\n\t"
		" bfe.u64 t1, %1, 60, 3;\n\t"
		" cvt.u16.u64 %0, t1;\n\t"
	    "}" : "=h"(t2) : "l"(part1), "l"(part2));
	*b = (statetype) t2;
}

inline __device__ void get_globalObject_P_4(statetype *b, nodetype part1, nodetype part2) {
	uint16_t t2;
	asm("{\n\t"
		" .reg .u64 t1;\n\t"
		" bfe.u64 t1, %1, 57, 3;\n\t"
		" cvt.u16.u64 %0, t1;\n\t"
	    "}" : "=h"(t2) : "l"(part1), "l"(part2));
	*b = (statetype) t2;
}

inline __device__ void get_globalObject_P_2(statetype *b, nodetype part1, nodetype part2) {
	uint16_t t2;
	asm("{\n\t"
		" .reg .u64 t1;\n\t"
		" bfe.u64 t1, %1, 54, 3;\n\t"
		" cvt.u16.u64 %0, t1;\n\t"
	    "}" : "=h"(t2) : "l"(part1), "l"(part2));
	*b = (statetype) t2;
}

inline __device__ void get_globalObject_P_3(statetype *b, nodetype part1, nodetype part2) {
	uint16_t t2;
	asm("{\n\t"
		" .reg .u64 t1;\n\t"
		" bfe.u64 t1, %1, 51, 3;\n\t"
		" cvt.u16.u64 %0, t1;\n\t"
	    "}" : "=h"(t2) : "l"(part1), "l"(part2));
	*b = (statetype) t2;
}

inline __device__ void get_globalObject_P_1(statetype *b, nodetype part1, nodetype part2) {
	uint16_t t2;
	asm("{\n\t"
		" .reg .u64 t1;\n\t"
		" bfe.u64 t1, %1, 48, 3;\n\t"
		" cvt.u16.u64 %0, t1;\n\t"
	    "}" : "=h"(t2) : "l"(part1), "l"(part2));
	*b = (statetype) t2;
}

inline __device__ void get_globalObject_P_3_myplace(elem_chartype *b, nodetype part1, nodetype part2) {
	uint16_t t2;
	asm("{\n\t"
		" .reg .u64 t1;\n\t"
		" bfe.u64 t1, %1, 40, 8;\n\t"
		" cvt.u16.u64 %0, t1;\n\t"
	    "}" : "=h"(t2) : "l"(part1), "l"(part2));
	*b = (elem_chartype) t2;
}

inline __device__ void get_globalObject_P_0_myplace(elem_chartype *b, nodetype part1, nodetype part2) {
	uint16_t t2;
	asm("{\n\t"
		" .reg .u64 t1;\n\t"
		" bfe.u64 t1, %1, 32, 8;\n\t"
		" cvt.u16.u64 %0, t1;\n\t"
	    "}" : "=h"(t2) : "l"(part1), "l"(part2));
	*b = (elem_chartype) t2;
}

inline __device__ void get_globalObject_next(elem_chartype *b, nodetype part1, nodetype part2) {
	uint16_t t2;
	asm("{\n\t"
		" .reg .u64 t1;\n\t"
		" bfe.u64 t1, %1, 24, 8;\n\t"
		" cvt.u16.u64 %0, t1;\n\t"
	    "}" : "=h"(t2) : "l"(part1), "l"(part2));
	*b = (elem_chartype) t2;
}

inline __device__ void get_globalObject_P_1_myplace(elem_chartype *b, nodetype part1, nodetype part2) {
	uint16_t t2;
	asm("{\n\t"
		" .reg .u64 t1;\n\t"
		" bfe.u64 t1, %1, 39, 8;\n\t"
		" cvt.u16.u64 %0, t1;\n\t"
	    "}" : "=h"(t2) : "l"(part1), "l"(part2));
	*b = (elem_chartype) t2;
}

inline __device__ void get_globalObject_P_4_myplace(elem_chartype *b, nodetype part1, nodetype part2) {
	uint16_t t2;
	asm("{\n\t"
		" .reg .u64 t1;\n\t"
		" bfe.u64 t1, %1, 31, 8;\n\t"
		" cvt.u16.u64 %0, t1;\n\t"
	    "}" : "=h"(t2) : "l"(part1), "l"(part2));
	*b = (elem_chartype) t2;
}

inline __device__ void get_globalObject_P_2_myplace(elem_chartype *b, nodetype part1, nodetype part2) {
	uint16_t t2;
	asm("{\n\t"
		" .reg .u64 t1;\n\t"
		" bfe.u64 t1, %1, 23, 8;\n\t"
		" cvt.u16.u64 %0, t1;\n\t"
	    "}" : "=h"(t2) : "l"(part1), "l"(part2));
	*b = (elem_chartype) t2;
}

// Data retrieval functions for array elements, including the fetching of required vector parts.
inline __device__ void get_globalObject_Slot(shared_indextype node_index, elem_chartype *b, array_indextype index) {
	nodetype part;
	uint16_t t2;
	if (index <= 2) {
		part = get_vectorpart_0(node_index);

		asm("{\n\t"
			" .reg .u64 t1;\n\t"
			" bfe.u64 t1, %1, %2, %3;\n\t"
			" cvt.u16.u64 %0, t1;\n\t"
	    	"}" : "=h"(t2) : "l"(part), "r"(16-(index-0)*8), "r"(8));
		*b = (elem_chartype) t2;
	}
	else if (index <= 4) {
		part = get_vectorpart_1(node_index);

		asm("{\n\t"
			" .reg .u64 t1;\n\t"
			" bfe.u64 t1, %1, %2, %3;\n\t"
			" cvt.u16.u64 %0, t1;\n\t"
	    	"}" : "=h"(t2) : "l"(part), "r"(55-(index-3)*8), "r"(8));
		*b = (elem_chartype) t2;
	}
}

// Data retrieval functions to support dynamic array accessing. Retrieve the requested element from a local array buffer, if present, and if not, fetch it and store it in the array buffer.
// Precondition: if the requested element is not in the buffer, there is still space in the buffer to store it.
inline __device__ elem_chartype globalObject_Slot(shared_indextype node_index, array_indextype *idx_0, elem_chartype *v_0, array_indextype e) {
	if (!A_IEX_1(*idx_0, e)) {
		// Fetch and store value.
		get_globalObject_Slot(node_index, v_0, e);
		A_STR_1(idx_0, v_0, (array_indextype) e, *v_0);
		return *v_0;
	}
	else {
		return A_LD_1(*idx_0, *v_0, e);
	}
}

// Retrieval of current state of state machine at position i in state vector.
inline __device__ void get_current_state(statetype *b, shared_indextype node_index, uint8_t i) {
	nodetype part1 = 0;
	nodetype part2 = 0;
	switch (i) {
		case 0:
			part1 = get_vectorpart_0(node_index);
			part2 = part1;
			get_globalObject_P_0(b, part1, part2);
			break;
		case 1:
			part1 = get_vectorpart_0(node_index);
			part2 = part1;
			get_globalObject_P_4(b, part1, part2);
			break;
		case 2:
			part1 = get_vectorpart_0(node_index);
			part2 = part1;
			get_globalObject_P_2(b, part1, part2);
			break;
		case 3:
			part1 = get_vectorpart_0(node_index);
			part2 = part1;
			get_globalObject_P_3(b, part1, part2);
			break;
		case 4:
			part1 = get_vectorpart_0(node_index);
			part2 = part1;
			get_globalObject_P_1(b, part1, part2);
			break;
		default:
			break;
	}
}

// CPU data retrieval functions. Retrieve particular state info from the given state vector part(s).
// Precondition: the given parts indeed contain the requested info.
inline void host_get_globalObject_P_0(statetype *b, nodetype part1, nodetype part2) {
	nodetype t1 = part1;
	// Strip away data beyond the requested data.
	t1 = t1 & 0x7fffffffffffffff;
	// Right shift to isolate requested data.
	t1 = t1 >> 60;
	*b = (statetype) t1;
}

inline void host_get_globalObject_P_4(statetype *b, nodetype part1, nodetype part2) {
	nodetype t1 = part1;
	// Strip away data beyond the requested data.
	t1 = t1 & 0xfffffffffffffff;
	// Right shift to isolate requested data.
	t1 = t1 >> 57;
	*b = (statetype) t1;
}

inline void host_get_globalObject_P_2(statetype *b, nodetype part1, nodetype part2) {
	nodetype t1 = part1;
	// Strip away data beyond the requested data.
	t1 = t1 & 0x1ffffffffffffff;
	// Right shift to isolate requested data.
	t1 = t1 >> 54;
	*b = (statetype) t1;
}

inline void host_get_globalObject_P_3(statetype *b, nodetype part1, nodetype part2) {
	nodetype t1 = part1;
	// Strip away data beyond the requested data.
	t1 = t1 & 0x3fffffffffffff;
	// Right shift to isolate requested data.
	t1 = t1 >> 51;
	*b = (statetype) t1;
}

inline void host_get_globalObject_P_1(statetype *b, nodetype part1, nodetype part2) {
	nodetype t1 = part1;
	// Strip away data beyond the requested data.
	t1 = t1 & 0x7ffffffffffff;
	// Right shift to isolate requested data.
	t1 = t1 >> 48;
	*b = (statetype) t1;
}

inline void host_get_globalObject_P_3_myplace(elem_chartype *b, nodetype part1, nodetype part2) {
	nodetype t1 = part1;
	// Strip away data beyond the requested data.
	t1 = t1 & 0xffffffffffff;
	// Right shift to isolate requested data.
	t1 = t1 >> 40;
	*b = (elem_chartype) t1;
}

inline void host_get_globalObject_P_0_myplace(elem_chartype *b, nodetype part1, nodetype part2) {
	nodetype t1 = part1;
	// Strip away data beyond the requested data.
	t1 = t1 & 0xffffffffff;
	// Right shift to isolate requested data.
	t1 = t1 >> 32;
	*b = (elem_chartype) t1;
}

inline void host_get_globalObject_next(elem_chartype *b, nodetype part1, nodetype part2) {
	nodetype t1 = part1;
	// Strip away data beyond the requested data.
	t1 = t1 & 0xffffffff;
	// Right shift to isolate requested data.
	t1 = t1 >> 24;
	*b = (elem_chartype) t1;
}

inline void host_get_globalObject_P_1_myplace(elem_chartype *b, nodetype part1, nodetype part2) {
	nodetype t1 = part1;
	// Strip away data beyond the requested data.
	t1 = t1 & 0x7fffffffffff;
	// Right shift to isolate requested data.
	t1 = t1 >> 39;
	*b = (elem_chartype) t1;
}

inline void host_get_globalObject_P_4_myplace(elem_chartype *b, nodetype part1, nodetype part2) {
	nodetype t1 = part1;
	// Strip away data beyond the requested data.
	t1 = t1 & 0x7fffffffff;
	// Right shift to isolate requested data.
	t1 = t1 >> 31;
	*b = (elem_chartype) t1;
}

inline void host_get_globalObject_P_2_myplace(elem_chartype *b, nodetype part1, nodetype part2) {
	nodetype t1 = part1;
	// Strip away data beyond the requested data.
	t1 = t1 & 0x7fffffff;
	// Right shift to isolate requested data.
	t1 = t1 >> 23;
	*b = (elem_chartype) t1;
}

// CPU data retrieval functions for arrays.
inline void host_get_globalObject_Slot(elem_chartype *b, nodetype part1, nodetype part2, array_indextype index) {
	nodetype t1 = part1;
	if (index <= 2) {
		// Right shift to isolate requested data.
		t1 = t1 >> (16 - ((index - 0)*8));
		// Strip away data beyond the requested data.
		t1 = t1 & 0xff;
		*b = (elem_chartype) t1;
	}
	else if (index <= 4) {
		// Right shift to isolate requested data.
		t1 = t1 >> (55 - ((index - 3)*8));
		// Strip away data beyond the requested data.
		t1 = t1 & 0xff;
		*b = (elem_chartype) t1;
	}
}

// GPU data update functions. Update particular state info in the given state vector part(s).
// Precondition: the given part indeed needs to contain the indicated fragment (left or right in case the info is split over two parts) of the updated info.
inline __device__ void set_left_globalObject_P_0(nodetype *part, elem_chartype x) {
	nodetype t1 = (nodetype) x;
	asm("{\n\t"
		" bfi.b64 %0, %1, %0, 60, 3;\n\t"
		"}" : "+l"(*part) : "l"(t1));
}

inline __device__ void set_left_globalObject_P_4(nodetype *part, elem_chartype x) {
	nodetype t1 = (nodetype) x;
	asm("{\n\t"
		" bfi.b64 %0, %1, %0, 57, 3;\n\t"
		"}" : "+l"(*part) : "l"(t1));
}

inline __device__ void set_left_globalObject_P_2(nodetype *part, elem_chartype x) {
	nodetype t1 = (nodetype) x;
	asm("{\n\t"
		" bfi.b64 %0, %1, %0, 54, 3;\n\t"
		"}" : "+l"(*part) : "l"(t1));
}

inline __device__ void set_left_globalObject_P_3(nodetype *part, elem_chartype x) {
	nodetype t1 = (nodetype) x;
	asm("{\n\t"
		" bfi.b64 %0, %1, %0, 51, 3;\n\t"
		"}" : "+l"(*part) : "l"(t1));
}

inline __device__ void set_left_globalObject_P_1(nodetype *part, elem_chartype x) {
	nodetype t1 = (nodetype) x;
	asm("{\n\t"
		" bfi.b64 %0, %1, %0, 48, 3;\n\t"
		"}" : "+l"(*part) : "l"(t1));
}

inline __device__ void set_left_globalObject_P_3_myplace(nodetype *part, elem_chartype x) {
	nodetype t1 = (nodetype) x;
	asm("{\n\t"
		" bfi.b64 %0, %1, %0, 40, 8;\n\t"
		"}" : "+l"(*part) : "l"(t1));
}

inline __device__ void set_left_globalObject_P_0_myplace(nodetype *part, elem_chartype x) {
	nodetype t1 = (nodetype) x;
	asm("{\n\t"
		" bfi.b64 %0, %1, %0, 32, 8;\n\t"
		"}" : "+l"(*part) : "l"(t1));
}

inline __device__ void set_left_globalObject_next(nodetype *part, elem_chartype x) {
	nodetype t1 = (nodetype) x;
	asm("{\n\t"
		" bfi.b64 %0, %1, %0, 24, 8;\n\t"
		"}" : "+l"(*part) : "l"(t1));
}

inline __device__ void set_left_globalObject_P_1_myplace(nodetype *part, elem_chartype x) {
	nodetype t1 = (nodetype) x;
	asm("{\n\t"
		" bfi.b64 %0, %1, %0, 39, 8;\n\t"
		"}" : "+l"(*part) : "l"(t1));
}

inline __device__ void set_left_globalObject_P_4_myplace(nodetype *part, elem_chartype x) {
	nodetype t1 = (nodetype) x;
	asm("{\n\t"
		" bfi.b64 %0, %1, %0, 31, 8;\n\t"
		"}" : "+l"(*part) : "l"(t1));
}

inline __device__ void set_left_globalObject_P_2_myplace(nodetype *part, elem_chartype x) {
	nodetype t1 = (nodetype) x;
	asm("{\n\t"
		" bfi.b64 %0, %1, %0, 23, 8;\n\t"
		"}" : "+l"(*part) : "l"(t1));
}

// Data update functions for arrays with dynamic indexing, focussed on one specific vector part.
// Auxiliary functions for globalObject'Slot.
inline __device__ bool array_element_is_in_vectorpart_globalObject_Slot(array_indextype i, vectornode_indextype pid) {
	switch (pid) {
		case 0:
			return (i >= 0 && i <= 2);
		case 1:
			return (i >= 3 && i <= 4);
		default:
			return false;
	}
}

// Precondition: array element i is (partially) stored in vector part pid.
inline __device__ bool is_left_vectorpart_for_array_element_globalObject_Slot(array_indextype i, vectornode_indextype pid) {
	switch (pid) {
		case 0:
			return (i >= 0 && i <= 2);
		case 1:
			return (i > 2 && i <= 4);
		default:
			return false;
	}	
}

// Left data update function for array globalObject'Slot.
// Precondition: the left part of the array element at the given index is stored in the vector part with the given ID pid
inline __device__ void set_left_globalObject_Slot(nodetype *part, array_indextype index, elem_chartype buf, uint8_t pid) {
	nodetype t1 = (nodetype) buf;
	switch (pid) {
		case 0:
			asm("{\n\t"
			" bfi.b64 %0, %1, %0, %2, %3;\n\t"
			"}" : "+l"(*part) : "l"(t1), "r"(16-(index-0)*8), "r"(8));
			break;
		case 1:
			asm("{\n\t"
			" bfi.b64 %0, %1, %0, %2, %3;\n\t"
			"}" : "+l"(*part) : "l"(t1), "r"(55-(index-3)*8), "r"(8));
			break;
		default:
			break;
	}
}

// Right data update function for array globalObject'Slot.
// Precondition: the right part of the array element at the given index is stored in the vector part with the given ID pid
inline __device__ void set_right_globalObject_Slot(nodetype *part, array_indextype index, elem_chartype buf, uint8_t pid) {
	nodetype t1 = (nodetype) buf;
	switch (pid) {
		default:
			break;
	}
}

// *** END FUNCTIONS FOR MODEL DATA RETRIEVAL AND STORAGE ***

// *** START KERNELS AND FUNCTIONS FOR VECTOR TREE NODE STORAGE AND RETRIEVAL TO/FROM THE GLOBAL MEMORY HASH TABLE ***

// Initial bitmixer function.
inline __host__ __device__ uint64_t UHASH_INIT(nodetype node) {
	uint64_t node1 = xor_shft2_64((uint64_t) node, 38, 14);
	node1 ^= 0xD1B54A32D192ED03L;
	node1 *= 0xAEF17502108EF2D9L;
	return node1;
}

inline __host__ __device__ nodetype UHASH(uint8_t id, uint64_t node) {
	uint64_t node1 = node;
	switch (id) {
		case 0:
			node1 = xor_shft2_64(node1, 12, 52);
			node1 = xor_shft2_64(node1, 37, 27);
			node1 *= 0xd1b549a75a913001L;
			node1 ^= rshft(node1, 43);
			node1 ^= rshft(node1, 31);
			node1 ^= rshft(node1, 23);
			node1 *= 0xdb4f0ad2012a3801L;
			node1 ^= rshft(node1, 28);
			break;
		case 1:
			node1 = xor_shft2_64(node1, 10, 54);
			node1 = xor_shft2_64(node1, 35, 29);
			node1 *= 0xe19b01a4b9790801L;
			node1 ^= rshft(node1, 40);
			node1 ^= rshft(node1, 34);
			node1 ^= rshft(node1, 19);
			node1 *= 0xe60e2b7534e39801L;
			node1 ^= rshft(node1, 26);
			break;
		case 2:
			node1 = xor_shft2_64(node1, 14, 50);
			node1 = xor_shft2_64(node1, 39, 25);
			node1 *= 0xe95e1e450c4a0001L;
			node1 ^= rshft(node1, 44);
			node1 ^= rshft(node1, 30);
			node1 ^= rshft(node1, 15);
			node1 *= 0xebedeec09311a001L;
			node1 ^= rshft(node1, 30);
			break;
		case 3:
			node1 = xor_shft2_64(node1, 13, 51);
			node1 = xor_shft2_64(node1, 34, 30);
			node1 *= 0xedf84e880b907001L;
			node1 ^= rshft(node1, 39);
			node1 ^= rshft(node1, 33);
			node1 ^= rshft(node1, 25);
			node1 *= 0xefa23a45cfedf001L;
			node1 ^= rshft(node1, 28);
			break;
		case 4:
			node1 = xor_shft2_64(node1, 11, 53);
			node1 = xor_shft2_64(node1, 35, 29);
			node1 *= 0xf10426bc3c049001L;
			node1 ^= rshft(node1, 23);
			node1 ^= rshft(node1, 30);
			node1 ^= rshft(node1, 26);
			node1 *= 0xf22eeccd9c67f001L;
			node1 ^= rshft(node1, 20);
			break;
		case 5:
			node1 = xor_shft2_64(node1, 15, 49);
			node1 = xor_shft2_64(node1, 33, 31);
			node1 *= 0xf32e82653debe001L;
			node1 ^= rshft(node1, 41);
			node1 ^= rshft(node1, 36);
			node1 ^= rshft(node1, 20);
			node1 *= 0xf40ba356b4275801L;
			node1 ^= rshft(node1, 23);
			break;
		case 6:
			node1 = xor_shft2_64(node1, 34, 30);
			node1 = xor_shft2_64(node1, 12, 52);
			node1 *= 0xf4ccd6e6806ab001L;
			node1 ^= rshft(node1, 38);
			node1 ^= rshft(node1, 42);
			node1 ^= rshft(node1, 24);
			node1 *= 0xf57716cb7624c001L;
			node1 ^= rshft(node1, 29);
			break;
		case 7:
			node1 = xor_shft2_64(node1, 23, 41);
			node1 = xor_shft2_64(node1, 35, 29);
			node1 *= 0xf60e40dc166cc801L;
			node1 ^= rshft(node1, 26);
			node1 ^= rshft(node1, 46);
			node1 ^= rshft(node1, 23);
			node1 *= 0xf6955e9e8d598801L;
			node1 ^= rshft(node1, 26);
			break;
		case 8:
			node1 = xor_shft2_64(node1, 31, 33);
			node1 = xor_shft2_64(node1, 34, 30);
			node1 *= 0xf70edc3d744ed001L;
			node1 ^= rshft(node1, 29);
			node1 ^= rshft(node1, 40);
			node1 ^= rshft(node1, 20);
			node1 *= 0xf77cb25898912801L;
			node1 ^= rshft(node1, 27);
			break;
		case 9:
			node1 = xor_shft2_64(node1, 24, 40);
			node1 = xor_shft2_64(node1, 20, 44);
			node1 *= 0xf7e077f74f578001L;
			node1 ^= rshft(node1, 40);
			node1 ^= rshft(node1, 33);
			node1 ^= rshft(node1, 25);
			node1 *= 0xf83b82f6f35ba801L;
			node1 ^= rshft(node1, 30);
			break;
		case 10:
			node1 = xor_shft2_64(node1, 10, 54);
			node1 = xor_shft2_64(node1, 19, 45);
			node1 *= 0xf88ee9a3839ce001L;
			node1 ^= rshft(node1, 26);
			node1 ^= rshft(node1, 46);
			node1 ^= rshft(node1, 23);
			node1 *= 0xf8db99901f3c8001L;
			node1 ^= rshft(node1, 28);
			break;
		case 11:
			node1 = xor_shft2_64(node1, 14, 50);
			node1 = xor_shft2_64(node1, 30, 34);
			node1 *= 0xf922585980506801L;
			node1 &= 0xffffffffffffffff;
			node1 ^= rshft(node1, 43);
			node1 ^= rshft(node1, 12);
			node1 ^= rshft(node1, 26);
			node1 *= 0xf963d3cf7e2a1801L;
			node1 &= 0xffffffffffffffff;
			node1 ^= rshft(node1, 31);
			break;
		case 12:
			node1 = xor_shft2_64(node1, 19, 45);
			node1 = xor_shft2_64(node1, 41, 23);
			node1 *= 0xf9a099bfb3022801L;
			node1 &= 0xffffffffffffffff;
			node1 ^= rshft(node1, 24);
			node1 ^= rshft(node1, 39);
			node1 ^= rshft(node1, 21);
			node1 *= 0xf9d92a6f4ae93001L;
			node1 &= 0xffffffffffffffff;
			node1 ^= rshft(node1, 25);
			break;
		case 13:
			node1 = xor_shft2_64(node1, 28, 36);
			node1 = xor_shft2_64(node1, 15, 49);
			node1 *= 0xfa0deebd73767801L;
			node1 &= 0xffffffffffffffff;
			node1 ^= rshft(node1, 32);
			node1 ^= rshft(node1, 40);
			node1 ^= rshft(node1, 24);
			node1 *= 0xfa3f47134b674001L;
			node1 &= 0xffffffffffffffff;
			node1 ^= rshft(node1, 27);
			break;
		case 14:
			node1 = xor_shft2_64(node1, 43, 21);
			node1 = xor_shft2_64(node1, 32, 32);
			node1 *= 0xfa6d875fe549e801L;
			node1 &= 0xffffffffffffffff;
			node1 ^= rshft(node1, 31);
			node1 ^= rshft(node1, 46);
			node1 ^= rshft(node1, 28);
			node1 *= 0xfa98f6e7e6aa4001L;
			node1 &= 0xffffffffffffffff;
			node1 ^= rshft(node1, 26);
			break;
		case 15:
			node1 = xor_shft2_64(node1, 35, 29);
			node1 = xor_shft2_64(node1, 42, 22);
			node1 *= 0xfac1d3f253805801L;
			node1 &= 0xffffffffffffffff;
			node1 ^= rshft(node1, 25);
			node1 ^= rshft(node1, 44);
			node1 ^= rshft(node1, 29);
			node1 *= 0xfae8596df8e0f801L;
			node1 &= 0xffffffffffffffff;
			node1 ^= rshft(node1, 28);
			break;
		case 16:
			node1 = xor_shft2_64(node1, 35, 29);
			node1 = xor_shft2_64(node1, 49, 15);
			node1 *= 0xfb0cb72e09912001L;
			node1 &= 0xffffffffffffffff;
			node1 ^= rshft(node1, 20);
			node1 ^= rshft(node1, 29);
			node1 ^= rshft(node1, 44);
			node1 *= 0xfb2f1d5d45068801L;
			node1 &= 0xffffffffffffffff;
			node1 ^= rshft(node1, 35);
			break;
		case 17:
			node1 = xor_shft2_64(node1, 45, 19);
			node1 = xor_shft2_64(node1, 29, 35);
			node1 *= 0xfb4fb0e58fd21001L;
			node1 &= 0xffffffffffffffff;
			node1 ^= rshft(node1, 15);
			node1 ^= rshft(node1, 44);
			node1 ^= rshft(node1, 33);
			node1 *= 0xfb6e96edd75cd001L;
			node1 &= 0xffffffffffffffff;
			node1 ^= rshft(node1, 41);
			break;
		case 18:
			node1 = xor_shft2_64(node1, 20, 44);
			node1 = xor_shft2_64(node1, 36, 28);
			node1 *= 0xfb8bf0f836eb0801L;
			node1 &= 0xffffffffffffffff;
			node1 ^= rshft(node1, 21);
			node1 ^= rshft(node1, 41);
			node1 ^= rshft(node1, 13);
			node1 *= 0xfba7daea24511001L;
			node1 &= 0xffffffffffffffff;
			node1 ^= rshft(node1, 34);
			break;
		case 19:
			node1 = xor_shft2_64(node1, 10, 54);
			node1 = xor_shft2_64(node1, 8, 56);
			node1 *= 0xfbc26ee0b36f8801L;
			node1 &= 0xffffffffffffffff;
			node1 ^= rshft(node1, 49);
			node1 ^= rshft(node1, 11);
			node1 ^= rshft(node1, 34);
			node1 *= 0xfbdbc52b8ed14001L;
			node1 &= 0xffffffffffffffff;
			node1 ^= rshft(node1, 15);
			break;
		case 20:
			node1 = xor_shft2_64(node1, 29, 35);
			node1 = xor_shft2_64(node1, 50, 14);
			node1 *= 0xfbf3f4486e688001L;
			node1 &= 0xffffffffffffffff;
			node1 ^= rshft(node1, 43);
			node1 ^= rshft(node1, 39);
			node1 ^= rshft(node1, 49);
			node1 *= 0xfc0b0cfe69507001L;
			node1 &= 0xffffffffffffffff;
			node1 ^= rshft(node1, 24);
			break;
		case 21:
			node1 = xor_shft2_64(node1, 38, 26);
			node1 = xor_shft2_64(node1, 31, 33);
			node1 *= 0xfc2125faae868001L;
			node1 &= 0xffffffffffffffff;
			node1 ^= rshft(node1, 50);
			node1 ^= rshft(node1, 34);
			node1 ^= rshft(node1, 48);
			node1 *= 0xfc364c4c4e6e4001L;
			node1 &= 0xffffffffffffffff;
			node1 ^= rshft(node1, 44);
			break;
		case 22:
			node1 = xor_shft2_64(node1, 14, 50);
			node1 = xor_shft2_64(node1, 49, 15);
			node1 *= 0xfc4a90f39b00c001L;
			node1 &= 0xffffffffffffffff;
			node1 ^= rshft(node1, 11);
			node1 ^= rshft(node1, 23);
			node1 ^= rshft(node1, 11);
			node1 *= 0xfc5e011de66eb801L;
			node1 &= 0xffffffffffffffff;
			node1 ^= rshft(node1, 8);
			break;
		case 23:
			node1 = xor_shft2_64(node1, 20, 44);
			node1 = xor_shft2_64(node1, 19, 45);
			node1 *= 0xfc70aa0535529001L;
			node1 &= 0xffffffffffffffff;
			node1 ^= rshft(node1, 50);
			node1 ^= rshft(node1, 29);
			node1 ^= rshft(node1, 28);
			node1 *= 0xfc8296fd443dd001L;
			node1 &= 0xffffffffffffffff;
			node1 ^= rshft(node1, 43);
			break;
		case 24:
			node1 = xor_shft2_64(node1, 31, 33);
			node1 = xor_shft2_64(node1, 18, 46);
			node1 *= 0xfc93d17169271001L;
			node1 &= 0xffffffffffffffff;
			node1 ^= rshft(node1, 39);
			node1 ^= rshft(node1, 14);
			node1 ^= rshft(node1, 32);
			node1 *= 0xfca466baa09f2001L;
			node1 &= 0xffffffffffffffff;
			node1 ^= rshft(node1, 20);
			break;
		case 25:
			node1 = xor_shft2_64(node1, 26, 38);
			node1 = xor_shft2_64(node1, 35, 29);
			node1 *= 0xfcb45e62fe09e801L;
			node1 &= 0xffffffffffffffff;
			node1 ^= rshft(node1, 18);
			node1 ^= rshft(node1, 24);
			node1 ^= rshft(node1, 50);
			node1 *= 0xfcc3bffadf4d6801L;
			node1 &= 0xffffffffffffffff;
			node1 ^= rshft(node1, 25);
			break;
		case 26:
			node1 = xor_shft2_64(node1, 48, 16);
			node1 = xor_shft2_64(node1, 49, 15);
			node1 *= 0xfcd2950bef268801L;
			node1 &= 0xffffffffffffffff;
			node1 ^= rshft(node1, 28);
			node1 ^= rshft(node1, 33);
			node1 ^= rshft(node1, 41);
			node1 *= 0xfce0e33f22ce8001L;
			node1 &= 0xffffffffffffffff;
			node1 ^= rshft(node1, 17);
			break;
		case 27:
			node1 = xor_shft2_64(node1, 26, 38);
			node1 = xor_shft2_64(node1, 20, 44);
			node1 *= 0xfceeb42967ca2001L;
			node1 &= 0xffffffffffffffff;
			node1 ^= rshft(node1, 46);
			node1 ^= rshft(node1, 43);
			node1 ^= rshft(node1, 23);
			node1 *= 0xfcfc0b896bf43001L;
			node1 &= 0xffffffffffffffff;
			node1 ^= rshft(node1, 18);
			break;
		case 28:
			node1 = xor_shft2_64(node1, 32, 32);
			node1 = xor_shft2_64(node1, 12, 52);
			node1 *= 0xfd08f2fd84bba801L;
			node1 &= 0xffffffffffffffff;
			node1 ^= rshft(node1, 10);
			node1 ^= rshft(node1, 15);
			node1 ^= rshft(node1, 49);
			node1 *= 0xfd156c57a01b1001L;
			node1 &= 0xffffffffffffffff;
			node1 ^= rshft(node1, 29);
			break;
		case 29:
			node1 = xor_shft2_64(node1, 30, 34);
			node1 = xor_shft2_64(node1, 49, 15);
			node1 *= 0xfd217f49540da801L;
			node1 &= 0xffffffffffffffff;
			node1 ^= rshft(node1, 27);
			node1 ^= rshft(node1, 45);
			node1 ^= rshft(node1, 36);
			node1 *= 0xfd2d2da9e732b801L;
			node1 &= 0xffffffffffffffff;
			node1 ^= rshft(node1, 49);
			break;
		case 30:
			node1 = xor_shft2_64(node1, 28, 36);
			node1 = xor_shft2_64(node1, 17, 47);
			node1 *= 0xfd3881260ec2f001L;
			node1 &= 0xffffffffffffffff;
			node1 ^= rshft(node1, 12);
			node1 ^= rshft(node1, 39);
			node1 ^= rshft(node1, 34);
			node1 *= 0xfd4379a53f94a801L;
			node1 &= 0xffffffffffffffff;
			node1 ^= rshft(node1, 43);
			break;
		case 31:
			node1 = xor_shft2_64(node1, 47, 17);
			node1 = xor_shft2_64(node1, 10, 54);
			node1 *= 0xfd4e1cef854fc801L;
			node1 &= 0xffffffffffffffff;
			node1 ^= rshft(node1, 28);
			node1 ^= rshft(node1, 39);
			node1 ^= rshft(node1, 50);
			node1 *= 0xfd586eda2734f801L;
			node1 &= 0xffffffffffffffff;
			node1 ^= rshft(node1, 7);
			break;
	}
	return node1;
}

// Initial bitmixer functions.
inline __host__ __device__ nodetype RHASH_INIT(nodetype node) {
	nodetype node1 = xor_shft2_58(node, 38, 20);
	node1 = xor_shft2_58(node1, 14, 44);
	node1 ^= 0x01b54a32d192ed03L;
	node1 = mult_58(node1, 0x02f17502108ef2d9L);
	return node1;
}

inline __host__ __device__ nodetype RHASH_INIT_INVERSE(nodetype node) {
	nodetype node1 = node;
	nodetype node2;
	node1 = mult_58(node1, 0x4ca582acb86d69L);
	node1 ^= 0x01b54a32d192ed03L;
	node2 = node1;
	node2 = (node1 ^ rshft(node2, 44));
	node2 = (node1 ^ rshft(node2, 44));
	node1 = node2;
	node1 = (node2 ^ lshft_58(node1, 14));
	node1 = (node2 ^ lshft_58(node1, 14));
	node1 = (node2 ^ lshft_58(node1, 14));
	node1 = (node2 ^ lshft_58(node1, 14));
	node1 = (node2 ^ lshft_58(node1, 14));
	node1 &= 0x3ffffffffffffff;;
	node2 = node1;
	node2 = (node1 ^ rshft(node2, 20));
	node2 = (node1 ^ rshft(node2, 20));
	node2 = (node1 ^ rshft(node2, 20));
	node1 = node2;
	node1 = (node2 ^ lshft_58(node1, 38));
	node1 = (node2 ^ lshft_58(node1, 38));
	node1 &= 0x3ffffffffffffff;;
	return node1;
}

inline __host__ __device__ nodetype RHASH(uint8_t id, nodetype node) {
	nodetype node1 = node;
	switch (id) {
		case 0:
			node1 = xor_shft2_58(node1, 12, 46);
			node1 = xor_shft2_58(node1, 37, 21);
			node1 = mult_58(node1, 0x346d5269d6a44c1L);
			node1 ^= rshft(node1, 43);
			node1 ^= rshft(node1, 31);
			node1 ^= rshft(node1, 23);
			node1 = mult_58(node1, 0x36d3c2b4804a8e1L);
			node1 ^= rshft(node1, 28);
			break;
		case 1:
			node1 = xor_shft2_58(node1, 10, 48);
			node1 = xor_shft2_58(node1, 35, 23);
			node1 = mult_58(node1, 0x3866c0692e5e421L);
			node1 ^= rshft(node1, 40);
			node1 ^= rshft(node1, 34);
			node1 ^= rshft(node1, 19);
			node1 = mult_58(node1, 0x39838add4d38e61L);
			node1 ^= rshft(node1, 26);
			break;
		case 2:
			node1 = xor_shft2_58(node1, 14, 44);
			node1 = xor_shft2_58(node1, 39, 19);
			node1 = mult_58(node1, 0x3a5787914312801L);
			node1 ^= rshft(node1, 44);
			node1 ^= rshft(node1, 30);
			node1 ^= rshft(node1, 15);
			node1 = mult_58(node1, 0x3afb7bb024c4681L);
			node1 ^= rshft(node1, 30);
			break;
		case 3:
			node1 = xor_shft2_58(node1, 13, 45);
			node1 = xor_shft2_58(node1, 34, 24);
			node1 = mult_58(node1, 0x3b7e13a202e41c1L);
			node1 ^= rshft(node1, 39);
			node1 ^= rshft(node1, 33);
			node1 ^= rshft(node1, 25);
			node1 = mult_58(node1, 0x3be88e9173fb7c1L);
			node1 ^= rshft(node1, 28);
			break;
		case 4:
			node1 = xor_shft2_58(node1, 11, 47);
			node1 = xor_shft2_58(node1, 35, 23);
			node1 = mult_58(node1, 0x3c4109af0f01241L);
			node1 ^= rshft(node1, 23);
			node1 ^= rshft(node1, 30);
			node1 ^= rshft(node1, 26);
			node1 = mult_58(node1, 0x3c8bbb336719fc1L);
			node1 ^= rshft(node1, 20);
			break;
		case 5:
			node1 = xor_shft2_58(node1, 15, 43);
			node1 = xor_shft2_58(node1, 33, 25);
			node1 = mult_58(node1, 0x3ccba0994f7af81L);
			node1 ^= rshft(node1, 41);
			node1 ^= rshft(node1, 36);
			node1 ^= rshft(node1, 20);
			node1 = mult_58(node1, 0x3d02e8d5ad09d61L);
			node1 ^= rshft(node1, 23);
			break;
		case 6:
			node1 = xor_shft2_58(node1, 34, 24);
			node1 = xor_shft2_58(node1, 12, 46);
			node1 = mult_58(node1, 0x3d3335b9a01aac1L);
			node1 ^= rshft(node1, 38);
			node1 ^= rshft(node1, 42);
			node1 ^= rshft(node1, 24);
			node1 = mult_58(node1, 0x3d5dc5b2dd89301L);
			node1 ^= rshft(node1, 29);
			break;
		case 7:
			node1 = xor_shft2_58(node1, 23, 35);
			node1 = xor_shft2_58(node1, 35, 23);
			node1 = mult_58(node1, 0x3d839037059b321L);
			node1 ^= rshft(node1, 26);
			node1 ^= rshft(node1, 46);
			node1 ^= rshft(node1, 23);
			node1 = mult_58(node1, 0x3da557a7a356621L);
			node1 ^= rshft(node1, 26);
			break;
		case 8:
			node1 = xor_shft2_58(node1, 31, 27);
			node1 = xor_shft2_58(node1, 34, 24);
			node1 = mult_58(node1, 0x3dc3b70f5d13b41L);
			node1 ^= rshft(node1, 29);
			node1 ^= rshft(node1, 40);
			node1 ^= rshft(node1, 20);
			node1 = mult_58(node1, 0x3ddf2c9626244a1L);
			node1 ^= rshft(node1, 27);
			break;
		case 9:
			node1 = xor_shft2_58(node1, 24, 34);
			node1 = xor_shft2_58(node1, 20, 38);
			node1 = mult_58(node1, 0x3df81dfdd3d5e01L);
			node1 ^= rshft(node1, 40);
			node1 ^= rshft(node1, 33);
			node1 ^= rshft(node1, 25);
			node1 = mult_58(node1, 0x3e0ee0bdbcd6ea1L);
			node1 ^= rshft(node1, 30);
			break;
		case 10:
			node1 = xor_shft2_58(node1, 10, 48);
			node1 = xor_shft2_58(node1, 19, 39);
			node1 = mult_58(node1, 0x3e23ba68e0e7381L);
			node1 ^= rshft(node1, 26);
			node1 ^= rshft(node1, 46);
			node1 ^= rshft(node1, 23);
			node1 = mult_58(node1, 0x3e36e66407cf201L);
			node1 ^= rshft(node1, 28);
			break;
		case 11:
			node1 = xor_shft2_58(node1, 14, 44);
			node1 = xor_shft2_58(node1, 30, 28);
			node1 = mult_58(node1, 0x3e48961660141a1L);
			node1 ^= rshft(node1, 43);
			node1 ^= rshft(node1, 12);
			node1 ^= rshft(node1, 26);
			node1 = mult_58(node1, 0x3e58f4f3df8a861L);
			node1 ^= rshft(node1, 31);
			break;
		case 12:
			node1 = xor_shft2_58(node1, 19, 39);
			node1 = xor_shft2_58(node1, 41, 17);
			node1 = mult_58(node1, 0x3e68266fecc08a1L);
			node1 ^= rshft(node1, 24);
			node1 ^= rshft(node1, 39);
			node1 ^= rshft(node1, 21);
			node1 = mult_58(node1, 0x3e764a9bd2ba4c1L);
			node1 ^= rshft(node1, 25);
			break;
		case 13:
			node1 = xor_shft2_58(node1, 28, 30);
			node1 = xor_shft2_58(node1, 15, 43);
			node1 = mult_58(node1, 0x3e837baf5cdd9e1L);
			node1 ^= rshft(node1, 32);
			node1 ^= rshft(node1, 40);
			node1 ^= rshft(node1, 24);
			node1 = mult_58(node1, 0x3e8fd1c4d2d9d01L);
			node1 ^= rshft(node1, 27);
			break;
		case 14:
			node1 = xor_shft2_58(node1, 43, 15);
			node1 = xor_shft2_58(node1, 32, 26);
			node1 = mult_58(node1, 0x3e9b61d7f9527a1L);
			node1 ^= rshft(node1, 31);
			node1 ^= rshft(node1, 46);
			node1 ^= rshft(node1, 28);
			node1 = mult_58(node1, 0x3ea63db9f9aa901L);
			node1 ^= rshft(node1, 26);
			break;
		case 15:
			node1 = xor_shft2_58(node1, 35, 23);
			node1 = xor_shft2_58(node1, 42, 16);
			node1 = mult_58(node1, 0x3eb074fc94e0161L);
			node1 ^= rshft(node1, 25);
			node1 ^= rshft(node1, 44);
			node1 ^= rshft(node1, 29);
			node1 = mult_58(node1, 0x3eba165b7e383e1L);
			node1 ^= rshft(node1, 28);
			break;
		case 16:
			node1 = xor_shft2_58(node1, 35, 23);
			node1 = xor_shft2_58(node1, 49, 9);
			node1 = mult_58(node1, 0x3ec32dcb8264481L);
			node1 ^= rshft(node1, 20);
			node1 ^= rshft(node1, 29);
			node1 ^= rshft(node1, 44);
			node1 = mult_58(node1, 0x3ecbc7575141a21L);
			node1 ^= rshft(node1, 35);
			break;
		case 17:
			node1 = xor_shft2_58(node1, 45, 13);
			node1 = xor_shft2_58(node1, 29, 29);
			node1 = mult_58(node1, 0x3ed3ec3963f4841L);
			node1 ^= rshft(node1, 15);
			node1 ^= rshft(node1, 44);
			node1 ^= rshft(node1, 33);
			node1 = mult_58(node1, 0x3edba5bb75d7341L);
			node1 ^= rshft(node1, 41);
			break;
		case 18:
			node1 = xor_shft2_58(node1, 20, 38);
			node1 = xor_shft2_58(node1, 36, 22);
			node1 = mult_58(node1, 0x3ee2fc3e0dbac21L);
			node1 ^= rshft(node1, 21);
			node1 ^= rshft(node1, 41);
			node1 ^= rshft(node1, 13);
			node1 = mult_58(node1, 0x3ee9f6ba8914441L);
			node1 ^= rshft(node1, 34);
			break;
		case 19:
			node1 = xor_shft2_58(node1, 10, 48);
			node1 = xor_shft2_58(node1, 8, 50);
			node1 = mult_58(node1, 0x3ef09bb82cdbe21L);
			node1 ^= rshft(node1, 49);
			node1 ^= rshft(node1, 11);
			node1 ^= rshft(node1, 34);
			node1 = mult_58(node1, 0x3ef6f14ae3b4501L);
			node1 ^= rshft(node1, 15);
			break;
		case 20:
			node1 = xor_shft2_58(node1, 29, 29);
			node1 = xor_shft2_58(node1, 50, 8);
			node1 = mult_58(node1, 0x3efcfd121b9a201L);
			node1 ^= rshft(node1, 43);
			node1 ^= rshft(node1, 39);
			node1 ^= rshft(node1, 49);
			node1 = mult_58(node1, 0x3f02c33f9a541c1L);
			node1 ^= rshft(node1, 24);
			break;
		case 21:
			node1 = xor_shft2_58(node1, 38, 20);
			node1 = xor_shft2_58(node1, 31, 27);
			node1 = mult_58(node1, 0x3f08497eaba1a01L);
			node1 ^= rshft(node1, 50);
			node1 ^= rshft(node1, 34);
			node1 ^= rshft(node1, 48);
			node1 = mult_58(node1, 0x3f0d9313139b901L);
			node1 ^= rshft(node1, 44);
			break;
		case 22:
			node1 = xor_shft2_58(node1, 14, 44);
			node1 = xor_shft2_58(node1, 49, 9);
			node1 = mult_58(node1, 0x3f12a43ce6c0301L);
			node1 ^= rshft(node1, 11);
			node1 ^= rshft(node1, 23);
			node1 ^= rshft(node1, 11);
			node1 = mult_58(node1, 0x3f178047799bae1L);
			node1 ^= rshft(node1, 8);
			break;
		case 23:
			node1 = xor_shft2_58(node1, 20, 38);
			node1 = xor_shft2_58(node1, 19, 39);
			node1 = mult_58(node1, 0x3f1c2a814d54a41L);
			node1 ^= rshft(node1, 50);
			node1 ^= rshft(node1, 29);
			node1 ^= rshft(node1, 28);
			node1 = mult_58(node1, 0x3f20a5bf510f741L);
			node1 ^= rshft(node1, 43);
			break;
		case 24:
			node1 = xor_shft2_58(node1, 31, 27);
			node1 = xor_shft2_58(node1, 18, 40);
			node1 = mult_58(node1, 0x3f24f45c5a49c41L);
			node1 ^= rshft(node1, 39);
			node1 ^= rshft(node1, 14);
			node1 ^= rshft(node1, 32);
			node1 = mult_58(node1, 0x3f2919aea827c81L);
			node1 ^= rshft(node1, 20);
			break;
		case 25:
			node1 = xor_shft2_58(node1, 26, 32);
			node1 = xor_shft2_58(node1, 35, 23);
			node1 = mult_58(node1, 0x3f2d1798bf827a1L);
			node1 ^= rshft(node1, 18);
			node1 ^= rshft(node1, 24);
			node1 ^= rshft(node1, 50);
			node1 = mult_58(node1, 0x3f30effeb7d35a1L);
			node1 ^= rshft(node1, 25);
			break;
		case 26:
			node1 = xor_shft2_58(node1, 48, 10);
			node1 = xor_shft2_58(node1, 49, 9);
			node1 = mult_58(node1, 0x3f34a542fbc9a21L);
			node1 ^= rshft(node1, 28);
			node1 ^= rshft(node1, 33);
			node1 ^= rshft(node1, 41);
			node1 = mult_58(node1, 0x3f3838cfc8b3a01L);
			node1 ^= rshft(node1, 17);
			break;
		case 27:
			node1 = xor_shft2_58(node1, 26, 32);
			node1 = xor_shft2_58(node1, 20, 38);
			node1 = mult_58(node1, 0x3f3bad0a59f2881L);
			node1 ^= rshft(node1, 46);
			node1 ^= rshft(node1, 43);
			node1 ^= rshft(node1, 23);
			node1 = mult_58(node1, 0x3f3f02e25afd0c1L);
			node1 ^= rshft(node1, 18);
			break;
		case 28:
			node1 = xor_shft2_58(node1, 32, 26);
			node1 = xor_shft2_58(node1, 12, 46);
			node1 = mult_58(node1, 0x3f423cbf612eea1L);
			node1 ^= rshft(node1, 10);
			node1 ^= rshft(node1, 15);
			node1 ^= rshft(node1, 49);
			node1 = mult_58(node1, 0x3f455b15e806c41L);
			node1 ^= rshft(node1, 29);
			break;
		case 29:
			node1 = xor_shft2_58(node1, 30, 28);
			node1 = xor_shft2_58(node1, 49, 9);
			node1 = mult_58(node1, 0x3f485fd255036a1L);
			node1 ^= rshft(node1, 27);
			node1 ^= rshft(node1, 45);
			node1 ^= rshft(node1, 36);
			node1 = mult_58(node1, 0x3f4b4b6a79ccae1L);
			node1 ^= rshft(node1, 49);
			break;
		case 30:
			node1 = xor_shft2_58(node1, 28, 30);
			node1 = xor_shft2_58(node1, 17, 41);
			node1 = mult_58(node1, 0x3f4e204983b0bc1L);
			node1 ^= rshft(node1, 12);
			node1 ^= rshft(node1, 39);
			node1 ^= rshft(node1, 34);
			node1 = mult_58(node1, 0x3f50de694fe52a1L);
			node1 ^= rshft(node1, 43);
			break;
		case 31:
			node1 = xor_shft2_58(node1, 47, 11);
			node1 = xor_shft2_58(node1, 10, 48);
			node1 = mult_58(node1, 0x3f53873be153f21L);
			node1 ^= rshft(node1, 28);
			node1 ^= rshft(node1, 39);
			node1 ^= rshft(node1, 50);
			node1 = mult_58(node1, 0x3f561bb689cd3e1L);
			node1 ^= rshft(node1, 7);
			break;
	}
	return node1;
}

// Inverse hash functions.
inline __host__ __device__ nodetype RHASH_INVERSE(uint8_t id, nodetype node) {
	nodetype node1 = node;
	nodetype node2;
	switch (id) {
		case 0:
			node2 = node1;
			node2 = (node1 ^ rshft(node2, 28));
			node2 = (node1 ^ rshft(node2, 28));
			node2 = (node1 ^ rshft(node2, 28));
			node1 = node2;
			node1 = mult_58(node1, 0x3e340cf8be69b21L);
			node2 = node1;
			node2 = (node1 ^ rshft(node2, 43));
			node2 = (node1 ^ rshft(node2, 43));
			node1 = node2;
			node1 = (node2 ^ rshft(node1, 31));
			node1 = (node2 ^ rshft(node1, 31));
			node2 = node1;
			node2 = (node1 ^ rshft(node2, 23));
			node2 = (node1 ^ rshft(node2, 23));
			node2 = (node1 ^ rshft(node2, 23));
			node1 = node2;
			node1 = mult_58(node1, 0x16e2e65fde04b41L);
			node2 = node1;
			node2 = (node1 ^ rshft(node2, 21));
			node2 = (node1 ^ rshft(node2, 21));
			node2 = (node1 ^ rshft(node2, 21));
			node1 = node2;
			node1 = (node2 ^ lshft_58(node1, 37));
			node1 = (node2 ^ lshft_58(node1, 37));
			node1 &= 0x3ffffffffffffff;
			node2 = node1;
			node2 = (node1 ^ rshft(node2, 46));
			node2 = (node1 ^ rshft(node2, 46));
			node1 = node2;
			node1 = (node2 ^ lshft_58(node1, 12));
			node1 = (node2 ^ lshft_58(node1, 12));
			node1 = (node2 ^ lshft_58(node1, 12));
			node1 = (node2 ^ lshft_58(node1, 12));
			node1 = (node2 ^ lshft_58(node1, 12));
			node1 &= 0x3ffffffffffffff;
			break;
		case 1:
			node2 = node1;
			node2 = (node1 ^ rshft(node2, 26));
			node2 = (node1 ^ rshft(node2, 26));
			node2 = (node1 ^ rshft(node2, 26));
			node1 = node2;
			node1 = mult_58(node1, 0x27e8bdf6b3595a1L);
			node2 = node1;
			node2 = (node1 ^ rshft(node2, 40));
			node2 = (node1 ^ rshft(node2, 40));
			node1 = node2;
			node1 = (node2 ^ rshft(node1, 34));
			node1 = (node2 ^ rshft(node1, 34));
			node2 = node1;
			node2 = (node1 ^ rshft(node2, 19));
			node2 = (node1 ^ rshft(node2, 19));
			node2 = (node1 ^ rshft(node2, 19));
			node2 = (node1 ^ rshft(node2, 19));
			node1 = node2;
			node1 = mult_58(node1, 0x3234705f3029fe1L);
			node2 = node1;
			node2 = (node1 ^ rshft(node2, 23));
			node2 = (node1 ^ rshft(node2, 23));
			node2 = (node1 ^ rshft(node2, 23));
			node1 = node2;
			node1 = (node2 ^ lshft_58(node1, 35));
			node1 = (node2 ^ lshft_58(node1, 35));
			node1 &= 0x3ffffffffffffff;
			node2 = node1;
			node2 = (node1 ^ rshft(node2, 48));
			node2 = (node1 ^ rshft(node2, 48));
			node1 = node2;
			node1 = (node2 ^ lshft_58(node1, 10));
			node1 = (node2 ^ lshft_58(node1, 10));
			node1 = (node2 ^ lshft_58(node1, 10));
			node1 = (node2 ^ lshft_58(node1, 10));
			node1 = (node2 ^ lshft_58(node1, 10));
			node1 = (node2 ^ lshft_58(node1, 10));
			node1 &= 0x3ffffffffffffff;
			break;
		case 2:
			node2 = node1;
			node2 = (node1 ^ rshft(node2, 30));
			node2 = (node1 ^ rshft(node2, 30));
			node1 = node2;
			node1 = mult_58(node1, 0x38516503a7df981L);
			node2 = node1;
			node2 = (node1 ^ rshft(node2, 44));
			node2 = (node1 ^ rshft(node2, 44));
			node1 = node2;
			node1 = (node2 ^ rshft(node1, 30));
			node1 = (node2 ^ rshft(node1, 30));
			node2 = node1;
			node2 = (node1 ^ rshft(node2, 15));
			node2 = (node1 ^ rshft(node2, 15));
			node2 = (node1 ^ rshft(node2, 15));
			node2 = (node1 ^ rshft(node2, 15));
			node1 = node2;
			node1 = mult_58(node1, 0x3880f7d420ed801L);
			node2 = node1;
			node2 = (node1 ^ rshft(node2, 19));
			node2 = (node1 ^ rshft(node2, 19));
			node2 = (node1 ^ rshft(node2, 19));
			node2 = (node1 ^ rshft(node2, 19));
			node1 = node2;
			node1 = (node2 ^ lshft_58(node1, 39));
			node1 = (node2 ^ lshft_58(node1, 39));
			node1 &= 0x3ffffffffffffff;
			node2 = node1;
			node2 = (node1 ^ rshft(node2, 44));
			node2 = (node1 ^ rshft(node2, 44));
			node1 = node2;
			node1 = (node2 ^ lshft_58(node1, 14));
			node1 = (node2 ^ lshft_58(node1, 14));
			node1 = (node2 ^ lshft_58(node1, 14));
			node1 = (node2 ^ lshft_58(node1, 14));
			node1 = (node2 ^ lshft_58(node1, 14));
			node1 &= 0x3ffffffffffffff;
			break;
		case 3:
			node2 = node1;
			node2 = (node1 ^ rshft(node2, 28));
			node2 = (node1 ^ rshft(node2, 28));
			node2 = (node1 ^ rshft(node2, 28));
			node1 = node2;
			node1 = mult_58(node1, 0x3a638f15ba85841L);
			node2 = node1;
			node2 = (node1 ^ rshft(node2, 39));
			node2 = (node1 ^ rshft(node2, 39));
			node1 = node2;
			node1 = (node2 ^ rshft(node1, 33));
			node1 = (node2 ^ rshft(node1, 33));
			node2 = node1;
			node2 = (node1 ^ rshft(node2, 25));
			node2 = (node1 ^ rshft(node2, 25));
			node2 = (node1 ^ rshft(node2, 25));
			node1 = node2;
			node1 = mult_58(node1, 0x20fabc04158ce41L);
			node2 = node1;
			node2 = (node1 ^ rshft(node2, 24));
			node2 = (node1 ^ rshft(node2, 24));
			node2 = (node1 ^ rshft(node2, 24));
			node1 = node2;
			node1 = (node2 ^ lshft_58(node1, 34));
			node1 = (node2 ^ lshft_58(node1, 34));
			node1 &= 0x3ffffffffffffff;
			node2 = node1;
			node2 = (node1 ^ rshft(node2, 45));
			node2 = (node1 ^ rshft(node2, 45));
			node1 = node2;
			node1 = (node2 ^ lshft_58(node1, 13));
			node1 = (node2 ^ lshft_58(node1, 13));
			node1 = (node2 ^ lshft_58(node1, 13));
			node1 = (node2 ^ lshft_58(node1, 13));
			node1 = (node2 ^ lshft_58(node1, 13));
			node1 &= 0x3ffffffffffffff;
			break;
		case 4:
			node2 = node1;
			node2 = (node1 ^ rshft(node2, 20));
			node2 = (node1 ^ rshft(node2, 20));
			node2 = (node1 ^ rshft(node2, 20));
			node1 = node2;
			node1 = mult_58(node1, 0x3592d3c27c27041L);
			node2 = node1;
			node2 = (node1 ^ rshft(node2, 23));
			node2 = (node1 ^ rshft(node2, 23));
			node2 = (node1 ^ rshft(node2, 23));
			node1 = node2;
			node1 = (node2 ^ rshft(node1, 30));
			node1 = (node2 ^ rshft(node1, 30));
			node2 = node1;
			node2 = (node1 ^ rshft(node2, 26));
			node2 = (node1 ^ rshft(node2, 26));
			node2 = (node1 ^ rshft(node2, 26));
			node1 = node2;
			node1 = mult_58(node1, 0x4f6952eaf8fdc1L);
			node2 = node1;
			node2 = (node1 ^ rshft(node2, 23));
			node2 = (node1 ^ rshft(node2, 23));
			node2 = (node1 ^ rshft(node2, 23));
			node1 = node2;
			node1 = (node2 ^ lshft_58(node1, 35));
			node1 = (node2 ^ lshft_58(node1, 35));
			node1 &= 0x3ffffffffffffff;
			node2 = node1;
			node2 = (node1 ^ rshft(node2, 47));
			node2 = (node1 ^ rshft(node2, 47));
			node1 = node2;
			node1 = (node2 ^ lshft_58(node1, 11));
			node1 = (node2 ^ lshft_58(node1, 11));
			node1 = (node2 ^ lshft_58(node1, 11));
			node1 = (node2 ^ lshft_58(node1, 11));
			node1 = (node2 ^ lshft_58(node1, 11));
			node1 = (node2 ^ lshft_58(node1, 11));
			node1 &= 0x3ffffffffffffff;
			break;
		case 5:
			node2 = node1;
			node2 = (node1 ^ rshft(node2, 23));
			node2 = (node1 ^ rshft(node2, 23));
			node2 = (node1 ^ rshft(node2, 23));
			node1 = node2;
			node1 = mult_58(node1, 0x1a0c447ad94c6a1L);
			node2 = node1;
			node2 = (node1 ^ rshft(node2, 41));
			node2 = (node1 ^ rshft(node2, 41));
			node1 = node2;
			node1 = (node2 ^ rshft(node1, 36));
			node1 = (node2 ^ rshft(node1, 36));
			node2 = node1;
			node2 = (node1 ^ rshft(node2, 20));
			node2 = (node1 ^ rshft(node2, 20));
			node2 = (node1 ^ rshft(node2, 20));
			node1 = node2;
			node1 = mult_58(node1, 0x2a87df258789081L);
			node2 = node1;
			node2 = (node1 ^ rshft(node2, 25));
			node2 = (node1 ^ rshft(node2, 25));
			node2 = (node1 ^ rshft(node2, 25));
			node1 = node2;
			node1 = (node2 ^ lshft_58(node1, 33));
			node1 = (node2 ^ lshft_58(node1, 33));
			node1 &= 0x3ffffffffffffff;
			node2 = node1;
			node2 = (node1 ^ rshft(node2, 43));
			node2 = (node1 ^ rshft(node2, 43));
			node1 = node2;
			node1 = (node2 ^ lshft_58(node1, 15));
			node1 = (node2 ^ lshft_58(node1, 15));
			node1 = (node2 ^ lshft_58(node1, 15));
			node1 = (node2 ^ lshft_58(node1, 15));
			node1 &= 0x3ffffffffffffff;
			break;
		case 6:
			node2 = node1;
			node2 = (node1 ^ rshft(node2, 29));
			node2 = (node1 ^ rshft(node2, 29));
			node1 = node2;
			node1 = mult_58(node1, 0xa6775beb906d01L);
			node2 = node1;
			node2 = (node1 ^ rshft(node2, 38));
			node2 = (node1 ^ rshft(node2, 38));
			node1 = node2;
			node1 = (node2 ^ rshft(node1, 42));
			node1 = (node2 ^ rshft(node1, 42));
			node2 = node1;
			node2 = (node1 ^ rshft(node2, 24));
			node2 = (node1 ^ rshft(node2, 24));
			node2 = (node1 ^ rshft(node2, 24));
			node1 = node2;
			node1 = mult_58(node1, 0x37e24eee615e541L);
			node2 = node1;
			node2 = (node1 ^ rshft(node2, 46));
			node2 = (node1 ^ rshft(node2, 46));
			node1 = node2;
			node1 = (node2 ^ lshft_58(node1, 12));
			node1 = (node2 ^ lshft_58(node1, 12));
			node1 = (node2 ^ lshft_58(node1, 12));
			node1 = (node2 ^ lshft_58(node1, 12));
			node1 = (node2 ^ lshft_58(node1, 12));
			node1 &= 0x3ffffffffffffff;
			node2 = node1;
			node2 = (node1 ^ rshft(node2, 24));
			node2 = (node1 ^ rshft(node2, 24));
			node2 = (node1 ^ rshft(node2, 24));
			node1 = node2;
			node1 = (node2 ^ lshft_58(node1, 34));
			node1 = (node2 ^ lshft_58(node1, 34));
			node1 &= 0x3ffffffffffffff;
			break;
		case 7:
			node2 = node1;
			node2 = (node1 ^ rshft(node2, 26));
			node2 = (node1 ^ rshft(node2, 26));
			node2 = (node1 ^ rshft(node2, 26));
			node1 = node2;
			node1 = mult_58(node1, 0x33b48646b8f9de1L);
			node2 = node1;
			node2 = (node1 ^ rshft(node2, 26));
			node2 = (node1 ^ rshft(node2, 26));
			node2 = (node1 ^ rshft(node2, 26));
			node1 = node2;
			node1 = (node2 ^ rshft(node1, 46));
			node1 = (node2 ^ rshft(node1, 46));
			node2 = node1;
			node2 = (node1 ^ rshft(node2, 23));
			node2 = (node1 ^ rshft(node2, 23));
			node2 = (node1 ^ rshft(node2, 23));
			node1 = node2;
			node1 = mult_58(node1, 0x14eaae5968790e1L);
			node2 = node1;
			node2 = (node1 ^ rshft(node2, 23));
			node2 = (node1 ^ rshft(node2, 23));
			node2 = (node1 ^ rshft(node2, 23));
			node1 = node2;
			node1 = (node2 ^ lshft_58(node1, 35));
			node1 = (node2 ^ lshft_58(node1, 35));
			node1 &= 0x3ffffffffffffff;
			node2 = node1;
			node2 = (node1 ^ rshft(node2, 35));
			node2 = (node1 ^ rshft(node2, 35));
			node1 = node2;
			node1 = (node2 ^ lshft_58(node1, 23));
			node1 = (node2 ^ lshft_58(node1, 23));
			node1 = (node2 ^ lshft_58(node1, 23));
			node1 &= 0x3ffffffffffffff;
			break;
		case 8:
			node2 = node1;
			node2 = (node1 ^ rshft(node2, 27));
			node2 = (node1 ^ rshft(node2, 27));
			node2 = (node1 ^ rshft(node2, 27));
			node1 = node2;
			node1 = mult_58(node1, 0x1fdf88f19a49f61L);
			node2 = node1;
			node2 = (node1 ^ rshft(node2, 29));
			node2 = (node1 ^ rshft(node2, 29));
			node1 = node2;
			node1 = (node2 ^ rshft(node1, 40));
			node1 = (node2 ^ rshft(node1, 40));
			node2 = node1;
			node2 = (node1 ^ rshft(node2, 20));
			node2 = (node1 ^ rshft(node2, 20));
			node2 = (node1 ^ rshft(node2, 20));
			node1 = node2;
			node1 = mult_58(node1, 0xdbbcfbf69154c1L);
			node2 = node1;
			node2 = (node1 ^ rshft(node2, 24));
			node2 = (node1 ^ rshft(node2, 24));
			node2 = (node1 ^ rshft(node2, 24));
			node1 = node2;
			node1 = (node2 ^ lshft_58(node1, 34));
			node1 = (node2 ^ lshft_58(node1, 34));
			node1 &= 0x3ffffffffffffff;
			node2 = node1;
			node2 = (node1 ^ rshft(node2, 27));
			node2 = (node1 ^ rshft(node2, 27));
			node2 = (node1 ^ rshft(node2, 27));
			node1 = node2;
			node1 = (node2 ^ lshft_58(node1, 31));
			node1 = (node2 ^ lshft_58(node1, 31));
			node1 &= 0x3ffffffffffffff;
			break;
		case 9:
			node2 = node1;
			node2 = (node1 ^ rshft(node2, 30));
			node2 = (node1 ^ rshft(node2, 30));
			node1 = node2;
			node1 = mult_58(node1, 0x2565bb394a9f561L);
			node2 = node1;
			node2 = (node1 ^ rshft(node2, 40));
			node2 = (node1 ^ rshft(node2, 40));
			node1 = node2;
			node1 = (node2 ^ rshft(node1, 33));
			node1 = (node2 ^ rshft(node1, 33));
			node2 = node1;
			node2 = (node1 ^ rshft(node2, 25));
			node2 = (node1 ^ rshft(node2, 25));
			node2 = (node1 ^ rshft(node2, 25));
			node1 = node2;
			node1 = mult_58(node1, 0x209a299946a201L);
			node2 = node1;
			node2 = (node1 ^ rshft(node2, 38));
			node2 = (node1 ^ rshft(node2, 38));
			node1 = node2;
			node1 = (node2 ^ lshft_58(node1, 20));
			node1 = (node2 ^ lshft_58(node1, 20));
			node1 = (node2 ^ lshft_58(node1, 20));
			node1 &= 0x3ffffffffffffff;
			node2 = node1;
			node2 = (node1 ^ rshft(node2, 34));
			node2 = (node1 ^ rshft(node2, 34));
			node1 = node2;
			node1 = (node2 ^ lshft_58(node1, 24));
			node1 = (node2 ^ lshft_58(node1, 24));
			node1 = (node2 ^ lshft_58(node1, 24));
			node1 &= 0x3ffffffffffffff;
			break;
		case 10:
			node2 = node1;
			node2 = (node1 ^ rshft(node2, 28));
			node2 = (node1 ^ rshft(node2, 28));
			node2 = (node1 ^ rshft(node2, 28));
			node1 = node2;
			node1 = mult_58(node1, 0x2749093cc470e01L);
			node2 = node1;
			node2 = (node1 ^ rshft(node2, 26));
			node2 = (node1 ^ rshft(node2, 26));
			node2 = (node1 ^ rshft(node2, 26));
			node1 = node2;
			node1 = (node2 ^ rshft(node1, 46));
			node1 = (node2 ^ rshft(node1, 46));
			node2 = node1;
			node2 = (node1 ^ rshft(node2, 23));
			node2 = (node1 ^ rshft(node2, 23));
			node2 = (node1 ^ rshft(node2, 23));
			node1 = node2;
			node1 = mult_58(node1, 0x397388d192dcc81L);
			node2 = node1;
			node2 = (node1 ^ rshft(node2, 39));
			node2 = (node1 ^ rshft(node2, 39));
			node1 = node2;
			node1 = (node2 ^ lshft_58(node1, 19));
			node1 = (node2 ^ lshft_58(node1, 19));
			node1 = (node2 ^ lshft_58(node1, 19));
			node1 = (node2 ^ lshft_58(node1, 19));
			node1 &= 0x3ffffffffffffff;
			node2 = node1;
			node2 = (node1 ^ rshft(node2, 48));
			node2 = (node1 ^ rshft(node2, 48));
			node1 = node2;
			node1 = (node2 ^ lshft_58(node1, 10));
			node1 = (node2 ^ lshft_58(node1, 10));
			node1 = (node2 ^ lshft_58(node1, 10));
			node1 = (node2 ^ lshft_58(node1, 10));
			node1 = (node2 ^ lshft_58(node1, 10));
			node1 = (node2 ^ lshft_58(node1, 10));
			node1 &= 0x3ffffffffffffff;
			break;
		case 11:
			node2 = node1;
			node2 = (node1 ^ rshft(node2, 31));
			node2 = (node1 ^ rshft(node2, 31));
			node1 = node2;
			node1 = mult_58(node1, 0x31b72238ae7fba1L);
			node2 = node1;
			node2 = (node1 ^ rshft(node2, 43));
			node2 = (node1 ^ rshft(node2, 43));
			node1 = node2;
			node1 = (node2 ^ rshft(node1, 12));
			node1 = (node2 ^ rshft(node1, 12));
			node1 = (node2 ^ rshft(node1, 12));
			node1 = (node2 ^ rshft(node1, 12));
			node1 = (node2 ^ rshft(node1, 12));
			node2 = node1;
			node2 = (node1 ^ rshft(node2, 26));
			node2 = (node1 ^ rshft(node2, 26));
			node2 = (node1 ^ rshft(node2, 26));
			node1 = node2;
			node1 = mult_58(node1, 0x17db804c1d6e261L);
			node2 = node1;
			node2 = (node1 ^ rshft(node2, 28));
			node2 = (node1 ^ rshft(node2, 28));
			node2 = (node1 ^ rshft(node2, 28));
			node1 = node2;
			node1 = (node2 ^ lshft_58(node1, 30));
			node1 = (node2 ^ lshft_58(node1, 30));
			node1 &= 0x3ffffffffffffff;
			node2 = node1;
			node2 = (node1 ^ rshft(node2, 44));
			node2 = (node1 ^ rshft(node2, 44));
			node1 = node2;
			node1 = (node2 ^ lshft_58(node1, 14));
			node1 = (node2 ^ lshft_58(node1, 14));
			node1 = (node2 ^ lshft_58(node1, 14));
			node1 = (node2 ^ lshft_58(node1, 14));
			node1 = (node2 ^ lshft_58(node1, 14));
			node1 &= 0x3ffffffffffffff;
			break;
		case 12:
			node2 = node1;
			node2 = (node1 ^ rshft(node2, 25));
			node2 = (node1 ^ rshft(node2, 25));
			node2 = (node1 ^ rshft(node2, 25));
			node1 = node2;
			node1 = mult_58(node1, 0x17cd12a8d2eeb41L);
			node2 = node1;
			node2 = (node1 ^ rshft(node2, 24));
			node2 = (node1 ^ rshft(node2, 24));
			node2 = (node1 ^ rshft(node2, 24));
			node1 = node2;
			node1 = (node2 ^ rshft(node1, 39));
			node1 = (node2 ^ rshft(node1, 39));
			node2 = node1;
			node2 = (node1 ^ rshft(node2, 21));
			node2 = (node1 ^ rshft(node2, 21));
			node2 = (node1 ^ rshft(node2, 21));
			node1 = node2;
			node1 = mult_58(node1, 0xdda8affbefdb61L);
			node2 = node1;
			node2 = (node1 ^ rshft(node2, 17));
			node2 = (node1 ^ rshft(node2, 17));
			node2 = (node1 ^ rshft(node2, 17));
			node2 = (node1 ^ rshft(node2, 17));
			node1 = node2;
			node1 = (node2 ^ lshft_58(node1, 41));
			node1 = (node2 ^ lshft_58(node1, 41));
			node1 &= 0x3ffffffffffffff;
			node2 = node1;
			node2 = (node1 ^ rshft(node2, 39));
			node2 = (node1 ^ rshft(node2, 39));
			node1 = node2;
			node1 = (node2 ^ lshft_58(node1, 19));
			node1 = (node2 ^ lshft_58(node1, 19));
			node1 = (node2 ^ lshft_58(node1, 19));
			node1 = (node2 ^ lshft_58(node1, 19));
			node1 &= 0x3ffffffffffffff;
			break;
		case 13:
			node2 = node1;
			node2 = (node1 ^ rshft(node2, 27));
			node2 = (node1 ^ rshft(node2, 27));
			node2 = (node1 ^ rshft(node2, 27));
			node1 = node2;
			node1 = mult_58(node1, 0x13d6fbb801b6301L);
			node2 = node1;
			node2 = (node1 ^ rshft(node2, 32));
			node2 = (node1 ^ rshft(node2, 32));
			node1 = node2;
			node1 = (node2 ^ rshft(node1, 40));
			node1 = (node2 ^ rshft(node1, 40));
			node2 = node1;
			node2 = (node1 ^ rshft(node2, 24));
			node2 = (node1 ^ rshft(node2, 24));
			node2 = (node1 ^ rshft(node2, 24));
			node1 = node2;
			node1 = mult_58(node1, 0x34aade611b82a21L);
			node2 = node1;
			node2 = (node1 ^ rshft(node2, 43));
			node2 = (node1 ^ rshft(node2, 43));
			node1 = node2;
			node1 = (node2 ^ lshft_58(node1, 15));
			node1 = (node2 ^ lshft_58(node1, 15));
			node1 = (node2 ^ lshft_58(node1, 15));
			node1 = (node2 ^ lshft_58(node1, 15));
			node1 &= 0x3ffffffffffffff;
			node2 = node1;
			node2 = (node1 ^ rshft(node2, 30));
			node2 = (node1 ^ rshft(node2, 30));
			node1 = node2;
			node1 = (node2 ^ lshft_58(node1, 28));
			node1 = (node2 ^ lshft_58(node1, 28));
			node1 = (node2 ^ lshft_58(node1, 28));
			node1 &= 0x3ffffffffffffff;
			break;
		case 14:
			node2 = node1;
			node2 = (node1 ^ rshft(node2, 26));
			node2 = (node1 ^ rshft(node2, 26));
			node2 = (node1 ^ rshft(node2, 26));
			node1 = node2;
			node1 = mult_58(node1, 0x303124e6af65701L);
			node2 = node1;
			node2 = (node1 ^ rshft(node2, 31));
			node2 = (node1 ^ rshft(node2, 31));
			node1 = node2;
			node1 = (node2 ^ rshft(node1, 46));
			node1 = (node2 ^ rshft(node1, 46));
			node2 = node1;
			node2 = (node1 ^ rshft(node2, 28));
			node2 = (node1 ^ rshft(node2, 28));
			node2 = (node1 ^ rshft(node2, 28));
			node1 = node2;
			node1 = mult_58(node1, 0x2b47151bd0a7c61L);
			node2 = node1;
			node2 = (node1 ^ rshft(node2, 26));
			node2 = (node1 ^ rshft(node2, 26));
			node2 = (node1 ^ rshft(node2, 26));
			node1 = node2;
			node1 = (node2 ^ lshft_58(node1, 32));
			node1 = (node2 ^ lshft_58(node1, 32));
			node1 &= 0x3ffffffffffffff;
			node2 = node1;
			node2 = (node1 ^ rshft(node2, 15));
			node2 = (node1 ^ rshft(node2, 15));
			node2 = (node1 ^ rshft(node2, 15));
			node2 = (node1 ^ rshft(node2, 15));
			node1 = node2;
			node1 = (node2 ^ lshft_58(node1, 43));
			node1 = (node2 ^ lshft_58(node1, 43));
			node1 &= 0x3ffffffffffffff;
			break;
		case 15:
			node2 = node1;
			node2 = (node1 ^ rshft(node2, 28));
			node2 = (node1 ^ rshft(node2, 28));
			node2 = (node1 ^ rshft(node2, 28));
			node1 = node2;
			node1 = mult_58(node1, 0x1941bdcc12c0021L);
			node2 = node1;
			node2 = (node1 ^ rshft(node2, 25));
			node2 = (node1 ^ rshft(node2, 25));
			node2 = (node1 ^ rshft(node2, 25));
			node1 = node2;
			node1 = (node2 ^ rshft(node1, 44));
			node1 = (node2 ^ rshft(node1, 44));
			node2 = node1;
			node2 = (node1 ^ rshft(node2, 29));
			node2 = (node1 ^ rshft(node2, 29));
			node1 = node2;
			node1 = mult_58(node1, 0x998456ffaa62a1L);
			node2 = node1;
			node2 = (node1 ^ rshft(node2, 16));
			node2 = (node1 ^ rshft(node2, 16));
			node2 = (node1 ^ rshft(node2, 16));
			node2 = (node1 ^ rshft(node2, 16));
			node1 = node2;
			node1 = (node2 ^ lshft_58(node1, 42));
			node1 = (node2 ^ lshft_58(node1, 42));
			node1 &= 0x3ffffffffffffff;
			node2 = node1;
			node2 = (node1 ^ rshft(node2, 23));
			node2 = (node1 ^ rshft(node2, 23));
			node2 = (node1 ^ rshft(node2, 23));
			node1 = node2;
			node1 = (node2 ^ lshft_58(node1, 35));
			node1 = (node2 ^ lshft_58(node1, 35));
			node1 &= 0x3ffffffffffffff;
			break;
		case 16:
			node2 = node1;
			node2 = (node1 ^ rshft(node2, 35));
			node2 = (node1 ^ rshft(node2, 35));
			node1 = node2;
			node1 = mult_58(node1, 0x2efb706fdede9e1L);
			node2 = node1;
			node2 = (node1 ^ rshft(node2, 20));
			node2 = (node1 ^ rshft(node2, 20));
			node2 = (node1 ^ rshft(node2, 20));
			node1 = node2;
			node1 = (node2 ^ rshft(node1, 29));
			node1 = (node2 ^ rshft(node1, 29));
			node2 = node1;
			node2 = (node1 ^ rshft(node2, 44));
			node2 = (node1 ^ rshft(node2, 44));
			node1 = node2;
			node1 = mult_58(node1, 0x3933d59b50dfb81L);
			node2 = node1;
			node2 = (node1 ^ rshft(node2, 9));
			node2 = (node1 ^ rshft(node2, 9));
			node2 = (node1 ^ rshft(node2, 9));
			node2 = (node1 ^ rshft(node2, 9));
			node2 = (node1 ^ rshft(node2, 9));
			node2 = (node1 ^ rshft(node2, 9));
			node2 = (node1 ^ rshft(node2, 9));
			node1 = node2;
			node1 = (node2 ^ lshft_58(node1, 49));
			node1 = (node2 ^ lshft_58(node1, 49));
			node1 &= 0x3ffffffffffffff;
			node2 = node1;
			node2 = (node1 ^ rshft(node2, 23));
			node2 = (node1 ^ rshft(node2, 23));
			node2 = (node1 ^ rshft(node2, 23));
			node1 = node2;
			node1 = (node2 ^ lshft_58(node1, 35));
			node1 = (node2 ^ lshft_58(node1, 35));
			node1 &= 0x3ffffffffffffff;
			break;
		case 17:
			node2 = node1;
			node2 = (node1 ^ rshft(node2, 41));
			node2 = (node1 ^ rshft(node2, 41));
			node1 = node2;
			node1 = mult_58(node1, 0x21a8aeb5ab11cc1L);
			node2 = node1;
			node2 = (node1 ^ rshft(node2, 15));
			node2 = (node1 ^ rshft(node2, 15));
			node2 = (node1 ^ rshft(node2, 15));
			node2 = (node1 ^ rshft(node2, 15));
			node1 = node2;
			node1 = (node2 ^ rshft(node1, 44));
			node1 = (node2 ^ rshft(node1, 44));
			node2 = node1;
			node2 = (node1 ^ rshft(node2, 33));
			node2 = (node1 ^ rshft(node2, 33));
			node1 = node2;
			node1 = mult_58(node1, 0x26923bf4120c7c1L);
			node2 = node1;
			node2 = (node1 ^ rshft(node2, 29));
			node2 = (node1 ^ rshft(node2, 29));
			node1 = node2;
			node1 = (node2 ^ lshft_58(node1, 29));
			node1 = (node2 ^ lshft_58(node1, 29));
			node1 &= 0x3ffffffffffffff;
			node2 = node1;
			node2 = (node1 ^ rshft(node2, 13));
			node2 = (node1 ^ rshft(node2, 13));
			node2 = (node1 ^ rshft(node2, 13));
			node2 = (node1 ^ rshft(node2, 13));
			node2 = (node1 ^ rshft(node2, 13));
			node1 = node2;
			node1 = (node2 ^ lshft_58(node1, 45));
			node1 = (node2 ^ lshft_58(node1, 45));
			node1 &= 0x3ffffffffffffff;
			break;
		case 18:
			node2 = node1;
			node2 = (node1 ^ rshft(node2, 34));
			node2 = (node1 ^ rshft(node2, 34));
			node1 = node2;
			node1 = mult_58(node1, 0x399c741b25ccbc1L);
			node2 = node1;
			node2 = (node1 ^ rshft(node2, 21));
			node2 = (node1 ^ rshft(node2, 21));
			node2 = (node1 ^ rshft(node2, 21));
			node1 = node2;
			node1 = (node2 ^ rshft(node1, 41));
			node1 = (node2 ^ rshft(node1, 41));
			node2 = node1;
			node2 = (node1 ^ rshft(node2, 13));
			node2 = (node1 ^ rshft(node2, 13));
			node2 = (node1 ^ rshft(node2, 13));
			node2 = (node1 ^ rshft(node2, 13));
			node2 = (node1 ^ rshft(node2, 13));
			node1 = node2;
			node1 = mult_58(node1, 0x31b1f4e059ed7e1L);
			node2 = node1;
			node2 = (node1 ^ rshft(node2, 22));
			node2 = (node1 ^ rshft(node2, 22));
			node2 = (node1 ^ rshft(node2, 22));
			node1 = node2;
			node1 = (node2 ^ lshft_58(node1, 36));
			node1 = (node2 ^ lshft_58(node1, 36));
			node1 &= 0x3ffffffffffffff;
			node2 = node1;
			node2 = (node1 ^ rshft(node2, 38));
			node2 = (node1 ^ rshft(node2, 38));
			node1 = node2;
			node1 = (node2 ^ lshft_58(node1, 20));
			node1 = (node2 ^ lshft_58(node1, 20));
			node1 = (node2 ^ lshft_58(node1, 20));
			node1 &= 0x3ffffffffffffff;
			break;
		case 19:
			node2 = node1;
			node2 = (node1 ^ rshft(node2, 15));
			node2 = (node1 ^ rshft(node2, 15));
			node2 = (node1 ^ rshft(node2, 15));
			node2 = (node1 ^ rshft(node2, 15));
			node1 = node2;
			node1 = mult_58(node1, 0x1880d14f55dbb01L);
			node2 = node1;
			node2 = (node1 ^ rshft(node2, 49));
			node2 = (node1 ^ rshft(node2, 49));
			node1 = node2;
			node1 = (node2 ^ rshft(node1, 11));
			node1 = (node2 ^ rshft(node1, 11));
			node1 = (node2 ^ rshft(node1, 11));
			node1 = (node2 ^ rshft(node1, 11));
			node1 = (node2 ^ rshft(node1, 11));
			node1 = (node2 ^ rshft(node1, 11));
			node2 = node1;
			node2 = (node1 ^ rshft(node2, 34));
			node2 = (node1 ^ rshft(node2, 34));
			node1 = node2;
			node1 = mult_58(node1, 0x201dedfc54d45e1L);
			node2 = node1;
			node2 = (node1 ^ rshft(node2, 50));
			node2 = (node1 ^ rshft(node2, 50));
			node1 = node2;
			node1 = (node2 ^ lshft_58(node1, 8));
			node1 = (node2 ^ lshft_58(node1, 8));
			node1 = (node2 ^ lshft_58(node1, 8));
			node1 = (node2 ^ lshft_58(node1, 8));
			node1 = (node2 ^ lshft_58(node1, 8));
			node1 = (node2 ^ lshft_58(node1, 8));
			node1 = (node2 ^ lshft_58(node1, 8));
			node1 = (node2 ^ lshft_58(node1, 8));
			node1 &= 0x3ffffffffffffff;
			node2 = node1;
			node2 = (node1 ^ rshft(node2, 48));
			node2 = (node1 ^ rshft(node2, 48));
			node1 = node2;
			node1 = (node2 ^ lshft_58(node1, 10));
			node1 = (node2 ^ lshft_58(node1, 10));
			node1 = (node2 ^ lshft_58(node1, 10));
			node1 = (node2 ^ lshft_58(node1, 10));
			node1 = (node2 ^ lshft_58(node1, 10));
			node1 = (node2 ^ lshft_58(node1, 10));
			node1 &= 0x3ffffffffffffff;
			break;
		case 20:
			node2 = node1;
			node2 = (node1 ^ rshft(node2, 24));
			node2 = (node1 ^ rshft(node2, 24));
			node2 = (node1 ^ rshft(node2, 24));
			node1 = node2;
			node1 = mult_58(node1, 0x2afe38ab861ce41L);
			node2 = node1;
			node2 = (node1 ^ rshft(node2, 43));
			node2 = (node1 ^ rshft(node2, 43));
			node1 = node2;
			node1 = (node2 ^ rshft(node1, 39));
			node1 = (node2 ^ rshft(node1, 39));
			node2 = node1;
			node2 = (node1 ^ rshft(node2, 49));
			node2 = (node1 ^ rshft(node2, 49));
			node1 = node2;
			node1 = mult_58(node1, 0xdf792e0ca5e01L);
			node2 = node1;
			node2 = (node1 ^ rshft(node2, 8));
			node2 = (node1 ^ rshft(node2, 8));
			node2 = (node1 ^ rshft(node2, 8));
			node2 = (node1 ^ rshft(node2, 8));
			node2 = (node1 ^ rshft(node2, 8));
			node2 = (node1 ^ rshft(node2, 8));
			node2 = (node1 ^ rshft(node2, 8));
			node2 = (node1 ^ rshft(node2, 8));
			node1 = node2;
			node1 = (node2 ^ lshft_58(node1, 50));
			node1 = (node2 ^ lshft_58(node1, 50));
			node1 &= 0x3ffffffffffffff;
			node2 = node1;
			node2 = (node1 ^ rshft(node2, 29));
			node2 = (node1 ^ rshft(node2, 29));
			node1 = node2;
			node1 = (node2 ^ lshft_58(node1, 29));
			node1 = (node2 ^ lshft_58(node1, 29));
			node1 &= 0x3ffffffffffffff;
			break;
		case 21:
			node2 = node1;
			node2 = (node1 ^ rshft(node2, 44));
			node2 = (node1 ^ rshft(node2, 44));
			node1 = node2;
			node1 = mult_58(node1, 0x11e22dcd774701L);
			node2 = node1;
			node2 = (node1 ^ rshft(node2, 50));
			node2 = (node1 ^ rshft(node2, 50));
			node1 = node2;
			node1 = (node2 ^ rshft(node1, 34));
			node1 = (node2 ^ rshft(node1, 34));
			node2 = node1;
			node2 = (node1 ^ rshft(node2, 48));
			node2 = (node1 ^ rshft(node2, 48));
			node1 = node2;
			node1 = mult_58(node1, 0x36a0d8d37e9e601L);
			node2 = node1;
			node2 = (node1 ^ rshft(node2, 27));
			node2 = (node1 ^ rshft(node2, 27));
			node2 = (node1 ^ rshft(node2, 27));
			node1 = node2;
			node1 = (node2 ^ lshft_58(node1, 31));
			node1 = (node2 ^ lshft_58(node1, 31));
			node1 &= 0x3ffffffffffffff;
			node2 = node1;
			node2 = (node1 ^ rshft(node2, 20));
			node2 = (node1 ^ rshft(node2, 20));
			node2 = (node1 ^ rshft(node2, 20));
			node1 = node2;
			node1 = (node2 ^ lshft_58(node1, 38));
			node1 = (node2 ^ lshft_58(node1, 38));
			node1 &= 0x3ffffffffffffff;
			break;
		case 22:
			node2 = node1;
			node2 = (node1 ^ rshft(node2, 8));
			node2 = (node1 ^ rshft(node2, 8));
			node2 = (node1 ^ rshft(node2, 8));
			node2 = (node1 ^ rshft(node2, 8));
			node2 = (node1 ^ rshft(node2, 8));
			node2 = (node1 ^ rshft(node2, 8));
			node2 = (node1 ^ rshft(node2, 8));
			node2 = (node1 ^ rshft(node2, 8));
			node1 = node2;
			node1 = mult_58(node1, 0x358ac41663d0921L);
			node2 = node1;
			node2 = (node1 ^ rshft(node2, 11));
			node2 = (node1 ^ rshft(node2, 11));
			node2 = (node1 ^ rshft(node2, 11));
			node2 = (node1 ^ rshft(node2, 11));
			node2 = (node1 ^ rshft(node2, 11));
			node2 = (node1 ^ rshft(node2, 11));
			node1 = node2;
			node1 = (node2 ^ rshft(node1, 23));
			node1 = (node2 ^ rshft(node1, 23));
			node1 = (node2 ^ rshft(node1, 23));
			node2 = node1;
			node2 = (node1 ^ rshft(node2, 11));
			node2 = (node1 ^ rshft(node2, 11));
			node2 = (node1 ^ rshft(node2, 11));
			node2 = (node1 ^ rshft(node2, 11));
			node2 = (node1 ^ rshft(node2, 11));
			node2 = (node1 ^ rshft(node2, 11));
			node1 = node2;
			node1 = mult_58(node1, 0xa7320f9e9cfd01L);
			node2 = node1;
			node2 = (node1 ^ rshft(node2, 9));
			node2 = (node1 ^ rshft(node2, 9));
			node2 = (node1 ^ rshft(node2, 9));
			node2 = (node1 ^ rshft(node2, 9));
			node2 = (node1 ^ rshft(node2, 9));
			node2 = (node1 ^ rshft(node2, 9));
			node2 = (node1 ^ rshft(node2, 9));
			node1 = node2;
			node1 = (node2 ^ lshft_58(node1, 49));
			node1 = (node2 ^ lshft_58(node1, 49));
			node1 &= 0x3ffffffffffffff;
			node2 = node1;
			node2 = (node1 ^ rshft(node2, 44));
			node2 = (node1 ^ rshft(node2, 44));
			node1 = node2;
			node1 = (node2 ^ lshft_58(node1, 14));
			node1 = (node2 ^ lshft_58(node1, 14));
			node1 = (node2 ^ lshft_58(node1, 14));
			node1 = (node2 ^ lshft_58(node1, 14));
			node1 = (node2 ^ lshft_58(node1, 14));
			node1 &= 0x3ffffffffffffff;
			break;
		case 23:
			node2 = node1;
			node2 = (node1 ^ rshft(node2, 43));
			node2 = (node1 ^ rshft(node2, 43));
			node1 = node2;
			node1 = mult_58(node1, 0x18e70c0e0a798c1L);
			node2 = node1;
			node2 = (node1 ^ rshft(node2, 50));
			node2 = (node1 ^ rshft(node2, 50));
			node1 = node2;
			node1 = (node2 ^ rshft(node1, 29));
			node1 = (node2 ^ rshft(node1, 29));
			node2 = node1;
			node2 = (node1 ^ rshft(node2, 28));
			node2 = (node1 ^ rshft(node2, 28));
			node2 = (node1 ^ rshft(node2, 28));
			node1 = node2;
			node1 = mult_58(node1, 0x1a5d3987f4fc5c1L);
			node2 = node1;
			node2 = (node1 ^ rshft(node2, 39));
			node2 = (node1 ^ rshft(node2, 39));
			node1 = node2;
			node1 = (node2 ^ lshft_58(node1, 19));
			node1 = (node2 ^ lshft_58(node1, 19));
			node1 = (node2 ^ lshft_58(node1, 19));
			node1 = (node2 ^ lshft_58(node1, 19));
			node1 &= 0x3ffffffffffffff;
			node2 = node1;
			node2 = (node1 ^ rshft(node2, 38));
			node2 = (node1 ^ rshft(node2, 38));
			node1 = node2;
			node1 = (node2 ^ lshft_58(node1, 20));
			node1 = (node2 ^ lshft_58(node1, 20));
			node1 = (node2 ^ lshft_58(node1, 20));
			node1 &= 0x3ffffffffffffff;
			break;
		case 24:
			node2 = node1;
			node2 = (node1 ^ rshft(node2, 20));
			node2 = (node1 ^ rshft(node2, 20));
			node2 = (node1 ^ rshft(node2, 20));
			node1 = node2;
			node1 = mult_58(node1, 0x114d3af41ee9c381L);
			node2 = node1;
			node2 = (node1 ^ rshft(node2, 39));
			node2 = (node1 ^ rshft(node2, 39));
			node1 = node2;
			node1 = (node2 ^ rshft(node1, 14));
			node1 = (node2 ^ rshft(node1, 14));
			node1 = (node2 ^ rshft(node1, 14));
			node1 = (node2 ^ rshft(node1, 14));
			node1 = (node2 ^ rshft(node1, 14));
			node2 = node1;
			node2 = (node1 ^ rshft(node2, 32));
			node2 = (node1 ^ rshft(node2, 32));
			node1 = node2;
			node1 = mult_58(node1, 0x2c334d4e37573c1L);
			node2 = node1;
			node2 = (node1 ^ rshft(node2, 40));
			node2 = (node1 ^ rshft(node2, 40));
			node1 = node2;
			node1 = (node2 ^ lshft_58(node1, 18));
			node1 = (node2 ^ lshft_58(node1, 18));
			node1 = (node2 ^ lshft_58(node1, 18));
			node1 = (node2 ^ lshft_58(node1, 18));
			node1 &= 0x3ffffffffffffff;
			node2 = node1;
			node2 = (node1 ^ rshft(node2, 27));
			node2 = (node1 ^ rshft(node2, 27));
			node2 = (node1 ^ rshft(node2, 27));
			node1 = node2;
			node1 = (node2 ^ lshft_58(node1, 31));
			node1 = (node2 ^ lshft_58(node1, 31));
			node1 &= 0x3ffffffffffffff;
			break;
		case 25:
			node2 = node1;
			node2 = (node1 ^ rshft(node2, 25));
			node2 = (node1 ^ rshft(node2, 25));
			node2 = (node1 ^ rshft(node2, 25));
			node1 = node2;
			node1 = mult_58(node1, 0xb5b41ead3ee61L);
			node2 = node1;
			node2 = (node1 ^ rshft(node2, 18));
			node2 = (node1 ^ rshft(node2, 18));
			node2 = (node1 ^ rshft(node2, 18));
			node2 = (node1 ^ rshft(node2, 18));
			node1 = node2;
			node1 = (node2 ^ rshft(node1, 24));
			node1 = (node2 ^ rshft(node1, 24));
			node1 = (node2 ^ rshft(node1, 24));
			node2 = node1;
			node2 = (node1 ^ rshft(node2, 50));
			node2 = (node1 ^ rshft(node2, 50));
			node1 = node2;
			node1 = mult_58(node1, 0x343bd3492677c61L);
			node2 = node1;
			node2 = (node1 ^ rshft(node2, 23));
			node2 = (node1 ^ rshft(node2, 23));
			node2 = (node1 ^ rshft(node2, 23));
			node1 = node2;
			node1 = (node2 ^ lshft_58(node1, 35));
			node1 = (node2 ^ lshft_58(node1, 35));
			node1 &= 0x3ffffffffffffff;
			node2 = node1;
			node2 = (node1 ^ rshft(node2, 32));
			node2 = (node1 ^ rshft(node2, 32));
			node1 = node2;
			node1 = (node2 ^ lshft_58(node1, 26));
			node1 = (node2 ^ lshft_58(node1, 26));
			node1 = (node2 ^ lshft_58(node1, 26));
			node1 &= 0x3ffffffffffffff;
			break;
		case 26:
			node2 = node1;
			node2 = (node1 ^ rshft(node2, 17));
			node2 = (node1 ^ rshft(node2, 17));
			node2 = (node1 ^ rshft(node2, 17));
			node2 = (node1 ^ rshft(node2, 17));
			node1 = node2;
			node1 = mult_58(node1, 0xdd46cce498c601L);
			node2 = node1;
			node2 = (node1 ^ rshft(node2, 28));
			node2 = (node1 ^ rshft(node2, 28));
			node2 = (node1 ^ rshft(node2, 28));
			node1 = node2;
			node1 = (node2 ^ rshft(node1, 33));
			node1 = (node2 ^ rshft(node1, 33));
			node2 = node1;
			node2 = (node1 ^ rshft(node2, 41));
			node2 = (node1 ^ rshft(node2, 41));
			node1 = node2;
			node1 = mult_58(node1, 0xb43892c16569e1L);
			node2 = node1;
			node2 = (node1 ^ rshft(node2, 9));
			node2 = (node1 ^ rshft(node2, 9));
			node2 = (node1 ^ rshft(node2, 9));
			node2 = (node1 ^ rshft(node2, 9));
			node2 = (node1 ^ rshft(node2, 9));
			node2 = (node1 ^ rshft(node2, 9));
			node2 = (node1 ^ rshft(node2, 9));
			node1 = node2;
			node1 = (node2 ^ lshft_58(node1, 49));
			node1 = (node2 ^ lshft_58(node1, 49));
			node1 &= 0x3ffffffffffffff;
			node2 = node1;
			node2 = (node1 ^ rshft(node2, 10));
			node2 = (node1 ^ rshft(node2, 10));
			node2 = (node1 ^ rshft(node2, 10));
			node2 = (node1 ^ rshft(node2, 10));
			node2 = (node1 ^ rshft(node2, 10));
			node2 = (node1 ^ rshft(node2, 10));
			node1 = node2;
			node1 = (node2 ^ lshft_58(node1, 48));
			node1 = (node2 ^ lshft_58(node1, 48));
			node1 &= 0x3ffffffffffffff;
			break;
		case 27:
			node2 = node1;
			node2 = (node1 ^ rshft(node2, 18));
			node2 = (node1 ^ rshft(node2, 18));
			node2 = (node1 ^ rshft(node2, 18));
			node2 = (node1 ^ rshft(node2, 18));
			node1 = node2;
			node1 = mult_58(node1, 0x3e7f2be0c9cbf41L);
			node2 = node1;
			node2 = (node1 ^ rshft(node2, 46));
			node2 = (node1 ^ rshft(node2, 46));
			node1 = node2;
			node1 = (node2 ^ rshft(node1, 43));
			node1 = (node2 ^ rshft(node1, 43));
			node2 = node1;
			node2 = (node1 ^ rshft(node2, 23));
			node2 = (node1 ^ rshft(node2, 23));
			node2 = (node1 ^ rshft(node2, 23));
			node1 = node2;
			node1 = mult_58(node1, 0x2a9e04101a91781L);
			node2 = node1;
			node2 = (node1 ^ rshft(node2, 38));
			node2 = (node1 ^ rshft(node2, 38));
			node1 = node2;
			node1 = (node2 ^ lshft_58(node1, 20));
			node1 = (node2 ^ lshft_58(node1, 20));
			node1 = (node2 ^ lshft_58(node1, 20));
			node1 &= 0x3ffffffffffffff;
			node2 = node1;
			node2 = (node1 ^ rshft(node2, 32));
			node2 = (node1 ^ rshft(node2, 32));
			node1 = node2;
			node1 = (node2 ^ lshft_58(node1, 26));
			node1 = (node2 ^ lshft_58(node1, 26));
			node1 = (node2 ^ lshft_58(node1, 26));
			node1 &= 0x3ffffffffffffff;
			break;
		case 28:
			node2 = node1;
			node2 = (node1 ^ rshft(node2, 29));
			node2 = (node1 ^ rshft(node2, 29));
			node1 = node2;
			node1 = mult_58(node1, 0x1bf68f79001a3c1L);
			node2 = node1;
			node2 = (node1 ^ rshft(node2, 10));
			node2 = (node1 ^ rshft(node2, 10));
			node2 = (node1 ^ rshft(node2, 10));
			node2 = (node1 ^ rshft(node2, 10));
			node2 = (node1 ^ rshft(node2, 10));
			node2 = (node1 ^ rshft(node2, 10));
			node1 = node2;
			node1 = (node2 ^ rshft(node1, 15));
			node1 = (node2 ^ rshft(node1, 15));
			node1 = (node2 ^ rshft(node1, 15));
			node1 = (node2 ^ rshft(node1, 15));
			node2 = node1;
			node2 = (node1 ^ rshft(node2, 49));
			node2 = (node1 ^ rshft(node2, 49));
			node1 = node2;
			node1 = mult_58(node1, 0x5e966c19447561L);
			node2 = node1;
			node2 = (node1 ^ rshft(node2, 46));
			node2 = (node1 ^ rshft(node2, 46));
			node1 = node2;
			node1 = (node2 ^ lshft_58(node1, 12));
			node1 = (node2 ^ lshft_58(node1, 12));
			node1 = (node2 ^ lshft_58(node1, 12));
			node1 = (node2 ^ lshft_58(node1, 12));
			node1 = (node2 ^ lshft_58(node1, 12));
			node1 &= 0x3ffffffffffffff;
			node2 = node1;
			node2 = (node1 ^ rshft(node2, 26));
			node2 = (node1 ^ rshft(node2, 26));
			node2 = (node1 ^ rshft(node2, 26));
			node1 = node2;
			node1 = (node2 ^ lshft_58(node1, 32));
			node1 = (node2 ^ lshft_58(node1, 32));
			node1 &= 0x3ffffffffffffff;
			break;
		case 29:
			node2 = node1;
			node2 = (node1 ^ rshft(node2, 49));
			node2 = (node1 ^ rshft(node2, 49));
			node1 = node2;
			node1 = mult_58(node1, 0x11235df1f15f921L);
			node2 = node1;
			node2 = (node1 ^ rshft(node2, 27));
			node2 = (node1 ^ rshft(node2, 27));
			node2 = (node1 ^ rshft(node2, 27));
			node1 = node2;
			node1 = (node2 ^ rshft(node1, 45));
			node1 = (node2 ^ rshft(node1, 45));
			node2 = node1;
			node2 = (node1 ^ rshft(node2, 36));
			node2 = (node1 ^ rshft(node2, 36));
			node1 = node2;
			node1 = mult_58(node1, 0x5e845710612d61L);
			node2 = node1;
			node2 = (node1 ^ rshft(node2, 9));
			node2 = (node1 ^ rshft(node2, 9));
			node2 = (node1 ^ rshft(node2, 9));
			node2 = (node1 ^ rshft(node2, 9));
			node2 = (node1 ^ rshft(node2, 9));
			node2 = (node1 ^ rshft(node2, 9));
			node2 = (node1 ^ rshft(node2, 9));
			node1 = node2;
			node1 = (node2 ^ lshft_58(node1, 49));
			node1 = (node2 ^ lshft_58(node1, 49));
			node1 &= 0x3ffffffffffffff;
			node2 = node1;
			node2 = (node1 ^ rshft(node2, 28));
			node2 = (node1 ^ rshft(node2, 28));
			node2 = (node1 ^ rshft(node2, 28));
			node1 = node2;
			node1 = (node2 ^ lshft_58(node1, 30));
			node1 = (node2 ^ lshft_58(node1, 30));
			node1 &= 0x3ffffffffffffff;
			break;
		case 30:
			node2 = node1;
			node2 = (node1 ^ rshft(node2, 43));
			node2 = (node1 ^ rshft(node2, 43));
			node1 = node2;
			node1 = mult_58(node1, 0x27940ca3c661161L);
			node2 = node1;
			node2 = (node1 ^ rshft(node2, 12));
			node2 = (node1 ^ rshft(node2, 12));
			node2 = (node1 ^ rshft(node2, 12));
			node2 = (node1 ^ rshft(node2, 12));
			node2 = (node1 ^ rshft(node2, 12));
			node1 = node2;
			node1 = (node2 ^ rshft(node1, 39));
			node1 = (node2 ^ rshft(node1, 39));
			node2 = node1;
			node2 = (node1 ^ rshft(node2, 34));
			node2 = (node1 ^ rshft(node2, 34));
			node1 = node2;
			node1 = mult_58(node1, 0xe1cd6bed930441L);
			node2 = node1;
			node2 = (node1 ^ rshft(node2, 41));
			node2 = (node1 ^ rshft(node2, 41));
			node1 = node2;
			node1 = (node2 ^ lshft_58(node1, 17));
			node1 = (node2 ^ lshft_58(node1, 17));
			node1 = (node2 ^ lshft_58(node1, 17));
			node1 = (node2 ^ lshft_58(node1, 17));
			node1 &= 0x3ffffffffffffff;
			node2 = node1;
			node2 = (node1 ^ rshft(node2, 30));
			node2 = (node1 ^ rshft(node2, 30));
			node1 = node2;
			node1 = (node2 ^ lshft_58(node1, 28));
			node1 = (node2 ^ lshft_58(node1, 28));
			node1 = (node2 ^ lshft_58(node1, 28));
			node1 &= 0x3ffffffffffffff;
			break;
		case 31:
			node2 = node1;
			node2 = (node1 ^ rshft(node2, 7));
			node2 = (node1 ^ rshft(node2, 7));
			node2 = (node1 ^ rshft(node2, 7));
			node2 = (node1 ^ rshft(node2, 7));
			node2 = (node1 ^ rshft(node2, 7));
			node2 = (node1 ^ rshft(node2, 7));
			node2 = (node1 ^ rshft(node2, 7));
			node2 = (node1 ^ rshft(node2, 7));
			node2 = (node1 ^ rshft(node2, 7));
			node1 = node2;
			node1 = mult_58(node1, 0x2bd62da95deb021L);
			node2 = node1;
			node2 = (node1 ^ rshft(node2, 28));
			node2 = (node1 ^ rshft(node2, 28));
			node2 = (node1 ^ rshft(node2, 28));
			node1 = node2;
			node1 = (node2 ^ rshft(node1, 39));
			node1 = (node2 ^ rshft(node1, 39));
			node2 = node1;
			node2 = (node1 ^ rshft(node2, 50));
			node2 = (node1 ^ rshft(node2, 50));
			node1 = node2;
			node1 = mult_58(node1, 0xe61a0f027704e1L);
			node2 = node1;
			node2 = (node1 ^ rshft(node2, 48));
			node2 = (node1 ^ rshft(node2, 48));
			node1 = node2;
			node1 = (node2 ^ lshft_58(node1, 10));
			node1 = (node2 ^ lshft_58(node1, 10));
			node1 = (node2 ^ lshft_58(node1, 10));
			node1 = (node2 ^ lshft_58(node1, 10));
			node1 = (node2 ^ lshft_58(node1, 10));
			node1 = (node2 ^ lshft_58(node1, 10));
			node1 &= 0x3ffffffffffffff;
			node2 = node1;
			node2 = (node1 ^ rshft(node2, 11));
			node2 = (node1 ^ rshft(node2, 11));
			node2 = (node1 ^ rshft(node2, 11));
			node2 = (node1 ^ rshft(node2, 11));
			node2 = (node1 ^ rshft(node2, 11));
			node2 = (node1 ^ rshft(node2, 11));
			node1 = node2;
			node1 = (node2 ^ lshft_58(node1, 47));
			node1 = (node2 ^ lshft_58(node1, 47));
			node1 &= 0x3ffffffffffffff;
			break;
	}
	return node1;
}

// Retrieve ID of used hash function from a compressed root vectortree node.
inline __host__ __device__ uint8_t get_hash_id_root(compressed_nodetype n) {
	uint8_t hid = (uint8_t) (n >> 26);
	// Remove the 'new' flag.
	return (hid & 0x1F);
}

// Reconstruct uncompressed root vectortree node from the given compressed root vectortree node and the address at which it is stored in the global memory hash table.
inline __host__ __device__ nodetype get_uncompressed_node_root(compressed_nodetype n, indextype i) {
	nodetype e = (nodetype) i;
	//printf("Intermediate value 1: %lu.\n", e);
	e = (e << 26);
	//printf("Intermediate value 2: %lu.\n", e);
	// Obtain hash function ID.
	uint8_t hid = get_hash_id_root(n);
	// Remove five bits for hash function ID + one for 'new' flag.
	e |= (nodetype) (n & 0x03FFFFFF);
	//printf("Intermediate value 3: %lu.\n", e);
	// Retrieve uncompressed node and return it.
	e = RHASH_INVERSE(hid, e);
	//printf("Intermediate value 4: %lu.\n", e);	
	e = RHASH_INIT_INVERSE(e);
	//printf("Intermediate value 5: %lu.\n", e);		
	return e;
}

// Extract a global memory root hash table index from a given hash.
inline __device__ indextype get_index_root(nodetype h) {
	return (indextype) (h >> 26);
}

// Extract a global memory internal hash table index from a given hash.
inline __device__ indextype get_index_internal(nodetype h) {
	return (indextype) (h >> 35);
}

// Construct from a hash and a hash function ID a compressed root vectortree node.
inline __device__ compressed_nodetype get_compressed_node_root(nodetype h, uint8_t hid) {
	compressed_nodetype e = (compressed_nodetype) hid;
	e = (e << 26);
	e |= (compressed_nodetype) (h & 0x3ffffff);
	return e;
}

// Retrieve vectortree node at index i of the global memory hash table.
inline __host__ __device__ nodetype HT_RETRIEVE(compressed_nodetype *d_q, nodetype *d_q_i, indextype i, bool is_root) {
	if (!is_root) {
		// We are retrieving a non-root node from the internal nodes table.
		return d_q_i[i];
	}
	else {
		// We are retrieving a root node from the root table.
		compressed_nodetype j = d_q[i];
		// Construct the complete element.
		return get_uncompressed_node_root(j, i);
	}
}

// Find or put a given vectortree node in the global hash table (with Cuckoo hashing).
// First the insertion of a single node.
inline __device__ uint64_t FINDORPUT_SINGLE_CUCKOO_NODE_INSERTION(compressed_nodetype *d_q, volatile uint8_t *d_newstate_flags, shared_indextype node_index, bool claim_work, nodetype e2, uint8_t *result) {
	nodetype e1;
	compressed_nodetype compressed_node;
	compressed_nodetype compressed_element;
	uint8_t min_hashes = NR_HASH_FUNCTIONS;
	uint8_t min_hash_id = 0;
	indextype addr;
	nodetype element;
	shared_inttype shared_addr;
	// Extract information from result.
	uint8_t is_new_node = *result;
	uint8_t hash_start = (is_new_node & 0x7F);
	is_new_node = is_new_node >> 7;
	e1 = RHASH_INIT(e2);
	#pragma unroll
	for (uint8_t i = hash_start; i < NR_HASH_FUNCTIONS; i++) {
		element = RHASH(i, e1);
		compressed_node = get_compressed_node_root(element, i);
		if (is_new_node) {
			compressed_node = mark_new(compressed_node);
		}
		// Special case: if the compressed node coincides with the value for an empty compressed node, continue.
		if (compressed_node == EMPTY_COMPRESSED_NODE) {
			continue;
		}
		addr = get_index_root(element);
		compressed_element = d_q[addr];
		if (compressed_element == EMPTY_COMPRESSED_NODE) {
			compressed_element = atomicCAS(&(d_q[addr]), EMPTY_COMPRESSED_NODE, compressed_node);
			if (compressed_element == EMPTY_COMPRESSED_NODE) {
				// Successfully stored the node.
				// Try to claim the vector for future work. For this, try to increment the OPENTILECOUNT counter.
				// If the given node_index is EMPTY_CACHE_POINTER, the cache has been skipped, since it was considered full. In that case,
				// we do not try to claim the vector, on the assumption that the full cache already provides sufficient new work.
				if (is_new_node && node_index != EMPTY_CACHE_POINTER && claim_work && (shared_addr = atomicAdd((unsigned int*) &OPENTILECOUNT, 1)) < OPENTILELEN) {
					// If there is still a next successor generation iteration, store a cache reference to the root in the work tile.
					// Otherwise, store the root itself.
					if (ITERATIONS < d_kernel_iters-1) {
						shared[OPENTILEOFFSET+(2*shared_addr)+1] = node_index;
					}
					else {
						shared[OPENTILEOFFSET+(2*shared_addr)] = get_left(e2);
						shared[OPENTILEOFFSET+(2*shared_addr)+1] = get_right(e2);
					}
					// Mark the state as old in the hash table.
					atomicCAS(&(d_q[addr]), compressed_node, mark_old(compressed_node));
				}
				else {
					if (is_new_node) {
						// There is work available for some block.
						d_newstate_flags[(addr / BLOCK_SIZE) % GRID_SIZE] = 1;
					}
				}
				*result = 0;
				return addr;
			}
		}
		if (filter_compressed_bookkeeping(compressed_element) == filter_compressed_bookkeeping(compressed_node)) {
			// The node is already stored.
			*result = 0;
			return addr;
		}
		else {
			// Another node is stored at this address. Remember the hash function id if the number of hashes applied to the
			// encountered node is the minimum number so far.
			if (get_hash_id_root(compressed_element) < min_hashes) {
				min_hashes = get_hash_id_root(compressed_element);
				min_hash_id = i;
			}
		}
	}
	// Store the node in the recorded position, evict the node currently stored there and continue storing the latter node.
	// Note: eventually, if all nodes can be stored, FINDORPUT_SINGLE returns the address of the node stored last, NOT of the one
	// for which FINDORPUT_SINGLE was originally called.
	if (min_hashes == NR_HASH_FUNCTIONS) {
		min_hash_id = 0;
	}
	element = RHASH(min_hash_id, e1);
	compressed_node = get_compressed_node_root(element, min_hash_id);
	if (is_new_node) {
		compressed_node = mark_new(compressed_node);
	}
	addr = get_index_root(element);
	compressed_element = atomicExch(&(d_q[addr]), compressed_node);
	// Try to claim the vector for future work.
	if (is_new_node && node_index != EMPTY_CACHE_POINTER && claim_work && (shared_addr = atomicAdd((unsigned int*) &OPENTILECOUNT, 1)) < OPENTILELEN) {
		// If there is still a next successor generation iteration, store a cache reference to the root in the work tile.
		// Otherwise, store the root itself.
		if (ITERATIONS < d_kernel_iters-1) {
			shared[OPENTILEOFFSET+(2*shared_addr)+1] = node_index;
		}
		else {
			shared[OPENTILEOFFSET+(2*shared_addr)] = get_left(e2);
			shared[OPENTILEOFFSET+(2*shared_addr)+1] = get_right(e2);
		}
		// Mark the state as old in the hash table.
		atomicCAS(&(d_q[addr]), compressed_node, mark_old(compressed_node));
	}
	else {
		if (is_new_node) {
			// There is work available for some block.
			d_newstate_flags[(addr / BLOCK_SIZE) % GRID_SIZE] = 1;
		}
	}
	// Prepare the evicted node for storage.
	element = get_uncompressed_node_root(compressed_element, addr);
	e1 = RHASH_INIT(element);
	*result = get_hash_id_root(compressed_element) + 1;
	// Set highest bit in result in case the element is new.
	if (is_new(compressed_element)) {
		*result |= 0x80;
	}
	return element;
}

// Main FINDORPUT_SINGLE_CUCKOO method, implementing node insertion with evictions.
// Precondition: the given node is collapsed if it is a root node.
inline __device__ uint64_t FINDORPUT_SINGLE_CUCKOO(compressed_nodetype *d_q, nodetype node, volatile uint8_t *d_newstate_flags, shared_indextype node_index, bool claim_work) {
	nodetype e = node;
	uint8_t result = 0x80;
	e = FINDORPUT_SINGLE_CUCKOO_NODE_INSERTION(d_q, d_newstate_flags, node_index, (ITERATIONS == d_kernel_iters-1), e, &result);
	if (result != 0) {
		for (uint8_t i = 0; i < 10 && result != 0; i++) {
			e = FINDORPUT_SINGLE_CUCKOO_NODE_INSERTION(d_q, d_newstate_flags, node_index, (ITERATIONS == d_kernel_iters-1), e, &result);
		}
	}
	return (result == 0 ? e : HASHTABLE_FULL);
}

// Find or put a given vectortree node in the global hash table (no Cuckoo hashing).
// Precondition: the given node is collapsed if it is a root node.
inline __device__ uint64_t FINDORPUT_SINGLE_NO_CUCKOO(compressed_nodetype *d_q, nodetype *d_q_i, bool *d_dummy, nodetype node, volatile uint8_t *d_newstate_flags, shared_indextype node_index) {
	nodetype e1;
	nodetype e2;
	indextype addr;
	nodetype element;
	e1 = UHASH_INIT(node);
	#pragma unroll
	for (uint8_t i = 0; i < NR_HASH_FUNCTIONS; i++) {
			e2 = UHASH(i, e1);
			addr = get_index_internal(e2);
			element = d_q_i[addr];
			if (element == EMPTY_NODE) {
				element = atomicCAS((unsigned long long *) &(d_q_i[addr]), (unsigned long long) element, (unsigned long long) node);
				if (element == EMPTY_NODE) {
					// This write is performed to fix a problem that causes the program to hang (!)
					(*d_dummy) = true;
					// Successfully stored the node.
					return addr;
				}
			}
			if (element == node) {
				// This write is performed to fix a problem that causes the program to hang (!)
				(*d_dummy) = true;
				// The node is already stored.
				return addr;
			}
	}
	// The node could not be stored with the hash functions. Try to do it with linear probing w.r.t. the last hash value.
	// Obtain a jump size for the probing.
	e2 = UHASH(0, e1);
	e2 = (e2 & 0xFF) + 10;
	#pragma unroll
	for (uint8_t i = 0; i < LINEARPROBINGFREQ; i++) {
		addr += e2;
		if (addr >= d_internal_hash_table_size) {
			addr -= d_internal_hash_table_size;
		}
		element = d_q_i[addr];
		if (element == EMPTY_NODE) {
			element = atomicCAS((unsigned long long *) &(d_q_i[addr]), (unsigned long long) element, (unsigned long long) node);
			if (element == EMPTY_NODE) {
				// This write is performed to fix a problem that causes the program to hang (!)
				(*d_dummy) = true;
				// Successfully stored the node.
				return addr;
			}
		}
		if (element == node) {
			// This write is performed to fix a problem that causes the program to hang (!)
			(*d_dummy) = true;
			// The node is already stored.
			return addr;
		}
	}
	// Error: hash table considered full.
	return HASHTABLE_FULL;
}

// Main FINDORPUT_SINGLE method, making sure that Cuckoo hashing is performed when appropriate.
inline __device__ uint64_t FINDORPUT_SINGLE(compressed_nodetype *d_q, nodetype *d_q_i, bool *d_dummy, nodetype node, volatile uint8_t *d_newstate_flags, shared_indextype node_index, bool is_root, bool claim_work) {
	if (is_root) {
		return FINDORPUT_SINGLE_CUCKOO(d_q, node, d_newstate_flags, node_index, claim_work);
	}
	else {
		return FINDORPUT_SINGLE_NO_CUCKOO(d_q, d_q_i, d_dummy, node, d_newstate_flags, node_index);
	}
}

// Find or put all new vectortree nodes stored in the shared memory cache into the global memory hash table.
__device__ void FINDORPUT_MANY(compressed_nodetype *d_q, nodetype *d_q_i, bool *d_dummy, volatile uint8_t *d_newstate_flags) {
	nodetype node;
	indextype addr;
	indextype addr2;
	shared_inttype node_pointers;
	shared_inttype node_pointers_child;
	bool work_to_do = true;

	if (THREAD_ID == 0) {
		CONTINUE = 1;
	}
	__syncthreads();
	while (CONTINUE == 1) {
		__syncthreads();
		if (THREAD_ID == 0) {
			CONTINUE = 0;
		}
		__syncthreads();
		if (work_to_do) {
			work_to_do = false;
			for (shared_indextype i = THREAD_ID; (i*3)+2 < (d_shared_size - CACHEOFFSET) && CONTINUE < 2; i += BLOCK_SIZE) {
				node_pointers = shared[CACHEOFFSET+(i*3)+2];
				// Check if node is ready for storage. Only new leafs are ready at this point. We rely on old non-leafs having pointers with the highest
				// two bits set to '00', new non-leafs having pointers with the highest two bits set to '10', empty entries having pointers set to EMPTYVECT32,
				// old leafs having pointers with the highest two bits set to '01', and new leafs having pointers set to CACHE_POINTERS_NEW_LEAF.
				if (cached_node_is_new_leaf(node_pointers)) {
					node = combine_halfs(shared[CACHEOFFSET+(i*3)], shared[CACHEOFFSET+(i*3)+1]);
					// Store node in hash table.
					addr = FINDORPUT_SINGLE(d_q, d_q_i, d_dummy, node, d_newstate_flags, i, false, true);
					if (addr == HASHTABLE_FULL) {
						// Error: internal hash table is full.
						CONTINUE = 3;
					}
					else {
						// Store global memory address in cache.
						set_cache_pointers_to_global_address(&shared[CACHEOFFSET+(i*3)+2], addr, true);
					}
				}
				// Node is not ready yet. Check if it can be updated.
				else if (cached_node_is_new_nonleaf(node_pointers)) {
					node = combine_halfs(shared[CACHEOFFSET+(i*3)], shared[CACHEOFFSET+(i*3)+1]);
					if (vectortree_node_contains_left_gap(node)) {
						// Look up left child and check for presence of global memory address.
						node_pointers_child = sv_step(i, false);
						node_pointers_child = shared[CACHEOFFSET+(node_pointers_child*3)+2];
						if (cached_node_contains_global_address(node_pointers_child)) {
							set_left_in_vectortree_node(&node, node_pointers_child);
							// Copy back to shared memory.
							shared[CACHEOFFSET+(i*3)] = get_left(node);
							shared[CACHEOFFSET+(i*3)+1] = get_right(node);
						}
					}
					if (vectortree_node_contains_right_gap(node)) {
						// Look up right child and check for presence of global memory address.
						node_pointers_child = sv_step(i, true);
						node_pointers_child = shared[CACHEOFFSET+(node_pointers_child*3)+2];
						if (cached_node_contains_global_address(node_pointers_child)) {
							set_right_in_vectortree_node(&node, node_pointers_child);
							// Copy back to shared memory.
							shared[CACHEOFFSET+(i*3)] = get_left(node);
							shared[CACHEOFFSET+(i*3)+1] = get_right(node);
						}
					}
					// Ready now?
					if (!vectortree_node_contains_left_gap(node) && !vectortree_node_contains_right_gap(node)) {
						// Store node in hash table.
						addr = FINDORPUT_SINGLE(d_q, d_q_i, d_dummy, collapse(node), d_newstate_flags, i, is_root(node), true);
						if (addr == HASHTABLE_FULL) {
							if (is_root(node)) {
								CONTINUE = 2;
							}
							else {
								CONTINUE = 3;
							}
						}
						else if (!is_root(node)) {
							// Will there be a next successor generation iteration in the current gather kernel launch?
							if (ITERATIONS < d_kernel_iters-1) {
								// Preserve original cache pointers for future successor generation iterations.
						
								// First check repeatedly for the left child in the vectortree whether it has stored its global address,
								// otherwise the overwritten information cannot be reconstructed. This procedure stops when a leaf has been reached, or a new non-leaf with a
								// global memory address.
								// We may also encounter an empty pointer to the left child. In that case, the child has not been stored, since it is not needed for successor generation.
								// Reconstruction of the nodes later on requires us to seperately store in the cache the global address to that missing child.
								if (sv_step(i, false) == EMPTY_CACHE_POINTER) {
									addr2 = store_global_address_stub(node, i);
									// Store the pointer to the global address.
									set_left_cache_pointer((shared_inttype *) &shared[CACHEOFFSET+(i*3)+2], addr2);
								}
								while(!cached_node_contains_global_address(shared[CACHEOFFSET+(sv_step(i, false)*3)+2])) {
									node_pointers = i;
									node_pointers_child = sv_step(i, false);
									if (sv_step(node_pointers_child, false) == EMPTY_CACHE_POINTER) {
										addr2 = store_global_address_stub(combine_halfs(shared[CACHEOFFSET+(node_pointers_child*3)], shared[CACHEOFFSET+(node_pointers_child*3)+1]), node_pointers_child);
										// Store the pointer to the global address.
										set_left_cache_pointer((shared_inttype *) &shared[CACHEOFFSET+(node_pointers_child*3)+2], addr2);
									}
									while(!cached_node_contains_global_address(shared[CACHEOFFSET+(sv_step(node_pointers_child, false)*3)+2])) {
										node_pointers = node_pointers_child;
										node_pointers_child = sv_step(node_pointers_child, false);
										if (sv_step(node_pointers_child, false) == EMPTY_CACHE_POINTER) {
											addr2 = store_global_address_stub(combine_halfs(shared[CACHEOFFSET+(node_pointers_child*3)], shared[CACHEOFFSET+(node_pointers_child*3)+1]), node_pointers_child);
											// Store the pointer to the global address.
											set_left_cache_pointer((shared_inttype *) &shared[CACHEOFFSET+(node_pointers_child*3)+2], addr2);
										}
									}
									// Set global address for node at cache address node_pointers_child. This address can be retrieved from node_pointers: it is the left global
									// address in the node stored at cache address node_pointers.

									// Overwrite first half of the node to preserve the cache pointers.
									shared[CACHEOFFSET+(node_pointers_child*3)] = shared[CACHEOFFSET+(node_pointers_child*3)+2];
									// Mark the node in the (preserved) pointers as old, to distinguish it from already older nodes that are required later for successor generation.
									// When executing PREPARE_CACHE(), the latter type of node has its highest bit set to 1 in that procedure, and we must make sure that the current
									// node is not seen as such a node.
									mark_cached_node_as_old(&shared[CACHEOFFSET+(node_pointers_child*3)]);
									// Store global memory address retrieved from the parent at cache address node_pointers.
									node = combine_halfs(shared[CACHEOFFSET+(node_pointers*3)], shared[CACHEOFFSET+(node_pointers*3)+1]);
									set_cache_pointers_to_global_address(&shared[CACHEOFFSET+(node_pointers_child*3)+2], get_pointer_from_vectortree_node(node, false), false);
								}
								// Overwrite first half of the node to preserve the cache pointers.
								shared[CACHEOFFSET+(i*3)] = shared[CACHEOFFSET+(i*3)+2];
								// Mark the node in the (preserved) pointers as old, to distinguish it from nodes that are required later for successor generation and for which we have
								// not applied the pointers preservation procedure. When executing PREPARE_CACHE(), the latter type of node has its highest bit set to 1 in that procedure,
								// and we must make sure that the current node is not seen as such a node.
								mark_cached_node_as_old(&shared[CACHEOFFSET+(i*3)]);
							}
							// Store global memory address in cache.
							set_cache_pointers_to_global_address(&shared[CACHEOFFSET+(i*3)+2], addr, false);
						}
					}
					else {
						work_to_do = true;
					}
				}
			}
		}
		__syncthreads();
		if (work_to_do) {
			if (CONTINUE == 0) {
				CONTINUE = 1;
			}
		}
		__syncthreads();
	}
	if (THREAD_ID == 0) {
		// Possibly fix OPENTILECOUNT.
		if (OPENTILECOUNT > OPENTILELEN) {
			OPENTILECOUNT = OPENTILELEN;
		}
	}
	__syncthreads();
}

// Function to store a vector tree node, either in the cache or the global hash table.
inline __device__ indextype STORENODE(uint8_t mode, compressed_nodetype *d_q, nodetype *d_q_i, bool *d_dummy, nodetype part1, nodetype *part2, shared_inttype *part_cachepointers, bool is_root, bool is_non_leaf, volatile uint8_t *d_newstate_flags) {
	if (part1 != *part2) {
		if (mode == TO_CACHE) {
			if (is_root) {
				mark_root(part2);
			}
			if (is_non_leaf) {
				mark_cached_node_new_nonleaf(part_cachepointers);
			}
			else {
				*part_cachepointers = CACHE_POINTERS_NEW_LEAF;
			}
			return STOREINCACHE(*part2, *part_cachepointers, !is_non_leaf);
		}
		else {
			// Store the node directly in the global hash table.
			return FINDORPUT_SINGLE(d_q, d_q_i, d_dummy, *part2, d_newstate_flags, EMPTY_CACHE_POINTER, is_root, (ITERATIONS == d_kernel_iters-1));
		}
	}
	else {
		return EMPTY_HASH_POINTER;
	}
}

// Kernel to store the initial state in the global memory hash table.
__global__ void store_initial_state(compressed_nodetype *d_q, nodetype *d_q_i, bool *d_dummy, volatile uint8_t *d_newstate_flags, nodetype *d_worktiles) {
	if (BLOCK_ID == 0) {
		// Reset the shared variables preceding the cache, and reset the cache.
		if (THREAD_ID < SH_OFFSET) {
			shared[THREAD_ID] = 0;
		}
		// Set ITERATIONS sufficiently high, causing FINDORPUT_SINGLE to store the global memory address of the initial state
		// in the work tile.
		if (THREAD_ID == 0) {
			ITERATIONS = d_kernel_iters;
		}
		#pragma unroll
		for (uint32_t i = THREAD_ID; i < (d_shared_size - SH_OFFSET); i += BLOCK_SIZE) {
			shared[SH_OFFSET + i] = EMPTYVECT32;
		}
		__syncthreads();
		if (GLOBAL_THREAD_ID < 3) {
			switch (GLOBAL_THREAD_ID) {
				case 0:
					shared[CACHEOFFSET+(0*3)] = get_left(0x7fffffffffffffff);
					shared[CACHEOFFSET+(0*3)+1] = get_right(0x7fffffffffffffff);
					shared[CACHEOFFSET+(0*3)+2] = 0x80008002;
					break;
				case 1:
					shared[CACHEOFFSET+(1*3)] = get_left(0x10000);
					shared[CACHEOFFSET+(1*3)+1] = get_right(0x10000);
					shared[CACHEOFFSET+(1*3)+2] = CACHE_POINTERS_NEW_LEAF;
					break;
				case 2:
					shared[CACHEOFFSET+(2*3)] = get_left(0x0);
					shared[CACHEOFFSET+(2*3)+1] = get_right(0x0);
					shared[CACHEOFFSET+(2*3)+2] = CACHE_POINTERS_NEW_LEAF;
					break;
			}
		}
		__syncthreads();
		FINDORPUT_MANY(d_q, d_q_i, d_dummy, d_newstate_flags);
		__syncthreads();
		// Done. Copy the work tile to global memory.
		if (THREAD_ID < OPENTILELEN) {
			d_worktiles[(OPENTILELEN+LASTSEARCHLEN+1)*BLOCK_ID + THREAD_ID] = combine_halfs(shared[OPENTILEOFFSET+(2*THREAD_ID)], shared[OPENTILEOFFSET+(2*THREAD_ID)+1]);
		}
		if (THREAD_ID == 0) {
			d_worktiles[(OPENTILELEN+LASTSEARCHLEN+1)*BLOCK_ID + OPENTILELEN + LASTSEARCHLEN] = OPENTILECOUNT;
		}
	}
}

// Auxiliary functions for the fetching of vectortrees from the global hash table. They encode the distribution of vectortree nodes over the threads
// in a vectortree group, and their structural relations with each other, sometimes per tree level.
inline __device__ uint8_t get_vectortree_node_parent_thread(uint8_t tid, uint8_t level) {
	switch (level) {
		case 1:
			switch (tid) {
				case 0:
					return 0;
					break;
				case 1:
					return 0;
					break;
				default:
					return 3;
					break;
			}
			break;
		default:
			return 3;
			break;		
	}
}

inline __device__ uint8_t get_vectortree_nonleaf_left_child_thread(uint8_t tid) {
	switch (tid) {
		case 0:
			return 0;
			break;
		default:
			return 3;
			break;
	}
}

inline __device__ uint8_t get_vectortree_nonleaf_right_child_thread(uint8_t tid) {
	switch (tid) {
		case 0:
			return 1;
			break;
		default:
			return 3;
			break;
	}
}

// Auxiliary functions to obtain bitmasks for smart vectortree fetching based on the states of the state machines.
// Given a vectorgroup thread id and a tree level, return a bitmask expressing which vectorparts are reachable from the node in that level
// assigned to that thread.
inline __device__ uint32_t get_part_reachability(uint8_t tid, uint8_t level) {
	switch (level) {
		case 0:
			switch (tid) {
				case 0:
					return 0xc0000000;
				default:
					return 0x0;
			}
		case 1:
			switch (tid) {
				case 0:
					return 0x80000000;
				case 1:
					return 0x40000000;
				default:
					return 0x0;
			}
		default:
			return 0x0;
	}
}

// Functions to obtain a bitmask for a given state machine state that indicates which vectorparts are of interest to process outgoing transitions
// of that state.
inline __device__ uint32_t get_part_bitmask_globalObject_P_0(statetype sid) {
	switch (sid) {
		case 0:
			return 0x80000000;
		case 1:
			return 0x80000000;
		case 2:
			return 0xc0000000;
		case 3:
			return 0xc0000000;
		case 4:
			return 0xc0000000;
		default:
			return 0;
	}
}
inline __device__ uint32_t get_part_bitmask_globalObject_P_4(statetype sid) {
	switch (sid) {
		case 0:
			return 0xc0000000;
		case 1:
			return 0xc0000000;
		case 2:
			return 0xc0000000;
		case 3:
			return 0xc0000000;
		case 4:
			return 0xc0000000;
		default:
			return 0;
	}
}
inline __device__ uint32_t get_part_bitmask_globalObject_P_2(statetype sid) {
	switch (sid) {
		case 0:
			return 0xc0000000;
		case 1:
			return 0xc0000000;
		case 2:
			return 0xc0000000;
		case 3:
			return 0xc0000000;
		case 4:
			return 0xc0000000;
		default:
			return 0;
	}
}
inline __device__ uint32_t get_part_bitmask_globalObject_P_3(statetype sid) {
	switch (sid) {
		case 0:
			return 0x80000000;
		case 1:
			return 0x80000000;
		case 2:
			return 0xc0000000;
		case 3:
			return 0xc0000000;
		case 4:
			return 0xc0000000;
		default:
			return 0;
	}
}
inline __device__ uint32_t get_part_bitmask_globalObject_P_1(statetype sid) {
	switch (sid) {
		case 0:
			return 0xc0000000;
		case 1:
			return 0xc0000000;
		case 2:
			return 0xc0000000;
		case 3:
			return 0xc0000000;
		case 4:
			return 0xc0000000;
		default:
			return 0;
	}
}

// Function to construct a bitmask for smart fetching, based on the given vectorparts.
inline __device__ uint32_t get_part_bitmask_for_states_in_vectorpart(uint8_t pid, nodetype part1, nodetype part2) {
	uint32_t result = 0x0;
	statetype s;
	switch (pid) {
		case 0:
			get_globalObject_P_0(&s, part1, part2);
			result = result | get_part_bitmask_globalObject_P_0(s);
			get_globalObject_P_4(&s, part1, part2);
			result = result | get_part_bitmask_globalObject_P_4(s);
			get_globalObject_P_2(&s, part1, part2);
			result = result | get_part_bitmask_globalObject_P_2(s);
			get_globalObject_P_3(&s, part1, part2);
			result = result | get_part_bitmask_globalObject_P_3(s);
			get_globalObject_P_1(&s, part1, part2);
			result = result | get_part_bitmask_globalObject_P_1(s);
			return result;
		case 1:
			return result;
		default:
			return result;
	}
}

// Retrieve a vectortree from the global hash table and store it in the cache. This is performed in a warp-centric way.
// Address rootref points to the root of the requested vectortree.
// The function returns the address of the root of the vectortree in the cache, or CACHE_FULL in the case the cache is full.
inline __device__ indextype FETCH(thread_block_tile<VECTOR_GROUP_SIZE> treegroup, compressed_nodetype *d_q, nodetype *d_q_i, indextype rootref) {
	nodetype node = EMPTY_NODE;
	nodetype leaf_node = EMPTY_NODE;
	nodetype node_tmp_1 = EMPTY_NODE;
	nodetype node_tmp_2 = EMPTY_NODE;
	indextype node_addr = 0;
	shared_inttype cache_pointers = EMPTYVECT32;
	indextype result;
	shared_indextype cache_addr = EMPTY_CACHE_POINTER;
	shared_indextype cache_addr_child = EMPTY_CACHE_POINTER;
	uint8_t gid = treegroup.thread_rank();
	uint8_t target_thread_id;
	uint32_t smart_fetching_bitmask = 0x0;
	
	if (gid == 0) {
		node = combine_halfs(shared[OPENTILEOFFSET+(2*rootref)], shared[OPENTILEOFFSET+(2*rootref)+1]);
		// Expand the node from 58 bits to 64 bits.
		node = expand(node);
	}
	// Obtain node from vectortree parent.
	node_tmp_1 = node;
	target_thread_id = 3;
	if (gid <= 1) {
		target_thread_id = get_vectortree_node_parent_thread(gid, 1);
	}
	treegroup.sync();
	node_tmp_2 = treegroup.shfl(node_tmp_1, target_thread_id);
	// Process the received node, if applicable.
	if (target_thread_id != 3 && node_tmp_2 != EMPTY_NODE) {
		node_addr = get_pointer_from_vectortree_node(node_tmp_2, false || gid == 1);	
		// Smart fetching: first only fetch state vector nodes that can reach parts containing statemachine states.
		if ((VECTOR_SMPARTS & get_part_reachability(gid, 1)) != 0x0) {
			leaf_node = HT_RETRIEVE(d_q, d_q_i, node_addr, false);
			// Store the leaf node in the cache. Link the global memory address to it such that it can be retrieved in case of collisions
			// when generating successors.
			set_cache_pointers_to_global_address(&cache_pointers, node_addr, true);
			result = STOREINCACHE(leaf_node, cache_pointers, true);
			if (result == CACHE_FULL) {
				return CACHE_FULL;
			}
			else {
				cache_addr = result;
			}
		}
	}
	// Fetch the vectorpart to the right for the construction of the smart fetching bitmask.
	// This is needed to handle cases where state machine states are stored in multiple parts.
	node_tmp_1 = leaf_node;
	treegroup.sync();
	node_tmp_2 = treegroup.shfl(node_tmp_1, gid+1);
	// Construct smart fetching bitmask.
	if (leaf_node != EMPTY_NODE) {
		smart_fetching_bitmask = get_part_bitmask_for_states_in_vectorpart(gid, leaf_node, node_tmp_2);
	}
	// Now merge all results of the different threads, resulting in the final bitmask.
	treegroup.sync();
	for (target_thread_id = treegroup.size()/2; target_thread_id > 0; target_thread_id /= 2) {
		smart_fetching_bitmask = smart_fetching_bitmask | treegroup.shfl_xor(smart_fetching_bitmask, target_thread_id);
	}
	// Finally discard the previously retrieved vectorpart ids from the bitmask.
	smart_fetching_bitmask = smart_fetching_bitmask & ~(VECTOR_SMPARTS);
	// Obtain node from vectortree parent.
	node_tmp_1 = node;
	target_thread_id = 3;
	if (gid <= 1) {
		target_thread_id = get_vectortree_node_parent_thread(gid, 1);
	}
	treegroup.sync();
	node_tmp_2 = treegroup.shfl(node_tmp_1, target_thread_id);
	// Process the received node, if applicable.
	if (target_thread_id != 3 && node_tmp_2 != EMPTY_NODE) {
		node_addr = get_pointer_from_vectortree_node(node_tmp_2, false || gid == 1);	
		// Smart fetching: only fetch vector nodes that are required for successor generation.
		if ((smart_fetching_bitmask & get_part_reachability(gid, 1)) != 0x0) {
			leaf_node = HT_RETRIEVE(d_q, d_q_i, node_addr, false);
			// Store the leaf node in the cache. Link the global memory address to it such that it can be retrieved in case of collisions
			// when generating successors.
			set_cache_pointers_to_global_address(&cache_pointers, node_addr, true);
			result = STOREINCACHE(leaf_node, cache_pointers, true);
			if (result == CACHE_FULL) {
				return CACHE_FULL;
			}
			else {
				cache_addr = result;
			}
		}
	}
	// Add the initially retrieved vectorpart ids to the bitmask.
	smart_fetching_bitmask = smart_fetching_bitmask | VECTOR_SMPARTS;
	cache_pointers = EMPTYVECT32;
	// Obtain cache address for left child.
	target_thread_id = 3;
	if (gid == 0) {
		if ((smart_fetching_bitmask & get_part_reachability(gid, 0)) != 0x0) {
			target_thread_id = get_vectortree_nonleaf_left_child_thread(gid);
		}
	}
	treegroup.sync();
	cache_addr_child = EMPTY_CACHE_POINTER;
	cache_addr_child = treegroup.shfl(cache_addr, target_thread_id);
	// Set the received cache pointer.
	if (target_thread_id != 3 && cache_addr_child != EMPTY_CACHE_POINTER) {
		set_left_cache_pointer(&cache_pointers, cache_addr_child);
	}
	// Obtain cache address for right child.
	target_thread_id = 3;
	if (gid == 0) {
		if ((smart_fetching_bitmask & get_part_reachability(gid, 0)) != 0x0) {
			target_thread_id = get_vectortree_nonleaf_right_child_thread(gid);
		}
	}
	treegroup.sync();
	cache_addr_child = EMPTY_CACHE_POINTER;
	cache_addr_child = treegroup.shfl(cache_addr, target_thread_id);
	// Set the received cache pointer.
	if (target_thread_id != 3 && cache_addr_child != EMPTY_CACHE_POINTER) {
		set_right_cache_pointer(&cache_pointers, cache_addr_child);
	}
	// Store the non-leaf node in the cache.
	if (gid == 0) {
		if ((smart_fetching_bitmask & get_part_reachability(gid, 0)) != 0x0) {
			result = STOREINCACHE(node, cache_pointers, false);
			if (result == CACHE_FULL) {
				return CACHE_FULL;
			}
			else {
				cache_addr = result;
			}
		}
	}
	treegroup.sync();
	// Obtain cache address of the root and return it.
	cache_addr_child = treegroup.shfl(cache_addr, 0);
	return (indextype) cache_addr_child;
}

// *** END KERNELS AND FUNCTIONS FOR VECTOR TREE NODE STORAGE AND RETRIEVAL TO/FROM THE GLOBAL MEMORY HASH TABLE ***

// *** START FUNCTIONS FOR INTRA-WARP BITONIC MERGESORT (Fast Segmented Sort on GPUs, Hou et al., 2017) ***

inline __device__ void CMP_SWP(statetype *s0, statetype *s1, shared_indextype *p0, shared_indextype *p1) {
	statetype s_tmp;
	shared_indextype p_tmp;

	if (*s0 > *s1) {
		s_tmp = *s0;
		*s0 = *s1;
		*s1 = s_tmp;
		p_tmp = *p0;
		*p0 = *p1;
		*p1 = p_tmp;
	}
}

inline __device__ void EQL_SWP(statetype *s0, statetype *s1, shared_indextype *p0, shared_indextype *p1) {
	statetype s_tmp;
	shared_indextype p_tmp;

	if (*s0 != *s1) {
		s_tmp = *s0;
		*s0 = *s1;
		*s1 = s_tmp;
		p_tmp = *p0;
		*p0 = *p1;
		*p1 = p_tmp;
	}
}

inline __device__ void SWP(statetype *s0, statetype *s1, shared_indextype *p0, shared_indextype *p1) {
	statetype s_tmp;
	shared_indextype p_tmp;

	s_tmp = *s0;
	*s0 = *s1;
	*s1 = s_tmp;
	p_tmp = *p0;
	*p0 = *p1;
	*p1 = p_tmp;
}

inline __device__ void _exch_intxn(statetype *s0, statetype *s1, statetype *s2, statetype *s3, shared_indextype *p0, shared_indextype *p1, shared_indextype *p2, shared_indextype *p3, uint8_t mask, bool bit) {
	statetype ex_s0, ex_s1;
	shared_indextype ex_p0, ex_p1;
	if (bit) SWP(s0, s2, p0, p2);
	if (bit) SWP(s1, s3, p1, p3);
	ex_s0 = *s0;
	ex_s1 = __shfl_xor_sync(0xFFFFFFFF, *s1, mask);
	ex_p0 = *p0;
	ex_p1 = __shfl_xor_sync(0xFFFFFFFF, *p1, mask);
	CMP_SWP(&ex_s0, &ex_s1, &ex_p0, &ex_p1);
	if (bit) EQL_SWP(&ex_s0, &ex_s1, &ex_p0, &ex_p1);
	*s0 = ex_s0;
	*s1 = __shfl_xor_sync(0xFFFFFFFF, ex_s1, mask);
	*p0 = ex_p0;
	*p1 = __shfl_xor_sync(0xFFFFFFFF, ex_p1, mask);
	ex_s0 = *s2;
	ex_s1 = __shfl_xor_sync(0xFFFFFFFF, *s3, mask);
	ex_p0 = *p2;
	ex_p1 = __shfl_xor_sync(0xFFFFFFFF, *p3, mask);
	CMP_SWP(&ex_s0, &ex_s1, &ex_p0, &ex_p1);
	if (bit) EQL_SWP(&ex_s0, &ex_s1, &ex_p0, &ex_p1);
	*s2 = ex_s0;
	*s3 = __shfl_xor_sync(0xFFFFFFFF, ex_s1, mask);
	*p2 = ex_p0;
	*p3 = __shfl_xor_sync(0xFFFFFFFF, ex_p1, mask);
	if (bit) SWP(s0, s2, p0, p2);
	if (bit) SWP(s1, s3, p1, p3);
}

inline __device__ void _exch_paral(statetype *s0, statetype *s1, statetype *s2, statetype *s3, shared_indextype *p0, shared_indextype *p1, shared_indextype *p2, shared_indextype *p3, uint8_t mask, bool bit) {
	statetype ex_s0, ex_s1;
	shared_indextype ex_p0, ex_p1;
	if (bit) SWP(s0, s1, p0, p1);
	if (bit) SWP(s2, s3, p2, p3);
	ex_s0 = *s0;
	ex_s1 = __shfl_xor_sync(0xFFFFFFFF, *s1, mask);
	ex_p0 = *p0;
	ex_p1 = __shfl_xor_sync(0xFFFFFFFF, *p1, mask);
	CMP_SWP(&ex_s0, &ex_s1, &ex_p0, &ex_p1);
	if (bit) EQL_SWP(&ex_s0, &ex_s1, &ex_p0, &ex_p1);
	*s0 = ex_s0;
	*s1 = __shfl_xor_sync(0xFFFFFFFF, ex_s1, mask);
	*p0 = ex_p0;
	*p1 = __shfl_xor_sync(0xFFFFFFFF, ex_p1, mask);
	ex_s0 = *s2;
	ex_s1 = __shfl_xor_sync(0xFFFFFFFF, *s3, mask);
	ex_p0 = *p2;
	ex_p1 = __shfl_xor_sync(0xFFFFFFFF, *p3, mask);
	CMP_SWP(&ex_s0, &ex_s1, &ex_p0, &ex_p1);
	if (bit) EQL_SWP(&ex_s0, &ex_s1, &ex_p0, &ex_p1);
	*s2 = ex_s0;
	*s3 = __shfl_xor_sync(0xFFFFFFFF, ex_s1, mask);
	*p2 = ex_p0;
	*p3 = __shfl_xor_sync(0xFFFFFFFF, ex_p1, mask);
	if (bit) SWP(s0, s1, p0, p1);
	if (bit) SWP(s2, s3, p2, p3);
}

// The main bitonic sorting function, including loading the data to be sorted,
// and returning the tile index of the element to be subsequently used by the calling thread.
// wid is the ID of the warp executing the function. It is a parameter (as opposed to deriving the ID from the thread dynamically),
// to allow a thread to run the function with multiple IDs.
__device__ shared_indextype get_sorted_opentile_element(uint8_t wid) {
	statetype s0, s1, s2, s3;	
	shared_indextype p0, p1, p2, p3, p_tmp1, p_tmp2, p_result;
	
	// Load the tile indices.
	asm("{\n\t"
		" cvt.u16.u32 %0, %1;\n\t"
		"}" : "=h"(p0) : "r"(shared[OPENTILEOFFSET+2*(0+LANE)+1]));
	if (p0 != EMPTYVECT16) {
		// Retrieve corresponding state value.
		get_current_state(&s0, p0, wid / OPENTILE_WARP_WIDTH);
	}
	else {
		s0 = NO_STATE;
	}
	p0 = 0+LANE;
	asm("{\n\t"
		" cvt.u16.u32 %0, %1;\n\t"
		"}" : "=h"(p1) : "r"(shared[OPENTILEOFFSET+2*(32+LANE)+1]));
	if (p1 != EMPTYVECT16) {
		// Retrieve corresponding state value.
		get_current_state(&s1, p1, wid / OPENTILE_WARP_WIDTH);
	}
	else {
		s1 = NO_STATE;
	}
	p1 = 32+LANE;
	asm("{\n\t"
		" cvt.u16.u32 %0, %1;\n\t"
		"}" : "=h"(p2) : "r"(shared[OPENTILEOFFSET+2*(64+LANE)+1]));
	if (p2 != EMPTYVECT16) {
		// Retrieve corresponding state value.
		get_current_state(&s2, p2, wid / OPENTILE_WARP_WIDTH);
	}
	else {
		s2 = NO_STATE;
	}
	p2 = 64+LANE;
	if (96+LANE < OPENTILELEN) {
		asm("{\n\t"
			" cvt.u16.u32 %0, %1;\n\t"
			"}" : "=h"(p3) : "r"(shared[OPENTILEOFFSET+2*(96+LANE)+1]));
		if (p3 != EMPTYVECT16) {
			// Retrieve corresponding state value.
			get_current_state(&s3, p3, wid / OPENTILE_WARP_WIDTH);
		}
		else {
			s3 = NO_STATE;
		}
		p3 = 96+LANE;
	}
	else {
		p3 = EMPTYVECT16;
		s3 = NO_STATE;
	}
	__syncwarp();
	// Perform the sorting.
	// exch_local intxn.
	CMP_SWP(&s0, &s1, &p0, &p1);
	CMP_SWP(&s2, &s3, &p2, &p3);
	// exch_local intxn.
	CMP_SWP(&s0, &s3, &p0, &p3);
	CMP_SWP(&s1, &s2, &p1, &p2);
	// exch_local paral.
	CMP_SWP(&s0, &s1, &p0, &p1);
	CMP_SWP(&s2, &s3, &p2, &p3);
	_exch_intxn(&s0, &s1, &s2, &s3, &p0, &p1, &p2, &p3, 0x1, (LANE & 0x1) != 0);
	// exch_local paral.
	CMP_SWP(&s0, &s2, &p0, &p2);
	CMP_SWP(&s1, &s3, &p1, &p3);
	// exch_local paral.
	CMP_SWP(&s0, &s1, &p0, &p1);
	CMP_SWP(&s2, &s3, &p2, &p3);
	_exch_intxn(&s0, &s1, &s2, &s3, &p0, &p1, &p2, &p3, 0x3, (LANE & 0x2) != 0);
	_exch_paral(&s0, &s1, &s2, &s3, &p0, &p1, &p2, &p3, 0x1, (LANE & 0x1) != 0);
	// exch_local paral.
	CMP_SWP(&s0, &s2, &p0, &p2);
	CMP_SWP(&s1, &s3, &p1, &p3);
	// exch_local paral.
	CMP_SWP(&s0, &s1, &p0, &p1);
	CMP_SWP(&s2, &s3, &p2, &p3);
	_exch_intxn(&s0, &s1, &s2, &s3, &p0, &p1, &p2, &p3, 0x7, (LANE & 0x4) != 0);
	_exch_paral(&s0, &s1, &s2, &s3, &p0, &p1, &p2, &p3, 0x2, (LANE & 0x2) != 0);
	_exch_paral(&s0, &s1, &s2, &s3, &p0, &p1, &p2, &p3, 0x1, (LANE & 0x1) != 0);
	// exch_local paral.
	CMP_SWP(&s0, &s2, &p0, &p2);
	CMP_SWP(&s1, &s3, &p1, &p3);
	// exch_local paral.
	CMP_SWP(&s0, &s1, &p0, &p1);
	CMP_SWP(&s2, &s3, &p2, &p3);
	_exch_intxn(&s0, &s1, &s2, &s3, &p0, &p1, &p2, &p3, 0xf, (LANE & 0x8) != 0);
	_exch_paral(&s0, &s1, &s2, &s3, &p0, &p1, &p2, &p3, 0x4, (LANE & 0x4) != 0);
	_exch_paral(&s0, &s1, &s2, &s3, &p0, &p1, &p2, &p3, 0x2, (LANE & 0x2) != 0);
	_exch_paral(&s0, &s1, &s2, &s3, &p0, &p1, &p2, &p3, 0x1, (LANE & 0x1) != 0);
	// exch_local paral.
	CMP_SWP(&s0, &s2, &p0, &p2);
	CMP_SWP(&s1, &s3, &p1, &p3);
	// exch_local paral.
	CMP_SWP(&s0, &s1, &p0, &p1);
	CMP_SWP(&s2, &s3, &p2, &p3);
	_exch_intxn(&s0, &s1, &s2, &s3, &p0, &p1, &p2, &p3, 0x1f, (LANE & 0x10) != 0);
	_exch_paral(&s0, &s1, &s2, &s3, &p0, &p1, &p2, &p3, 0x8, (LANE & 0x8) != 0);
	_exch_paral(&s0, &s1, &s2, &s3, &p0, &p1, &p2, &p3, 0x4, (LANE & 0x4) != 0);
	_exch_paral(&s0, &s1, &s2, &s3, &p0, &p1, &p2, &p3, 0x2, (LANE & 0x2) != 0);
	_exch_paral(&s0, &s1, &s2, &s3, &p0, &p1, &p2, &p3, 0x1, (LANE & 0x1) != 0);
	// exch_local paral.
	CMP_SWP(&s0, &s2, &p0, &p2);
	CMP_SWP(&s1, &s3, &p1, &p3);
	// exch_local paral.
	CMP_SWP(&s0, &s1, &p0, &p1);
	CMP_SWP(&s2, &s3, &p2, &p3);

	// Finally, retrieve the index of the tile element of interest for the current thread.
	uint8_t offset = wid % OPENTILE_WARP_WIDTH;
	// If the index of the p0 element of the thread is within the range of interest, prepare it for communication.
	if (LANE/(WARP_SIZE/4) == offset) {
		p_tmp1 = p0;
	}
	__syncwarp();
	// Retrieve from the thread that holds the element of interest for the current thread the prepared value (if it exists).
	p_tmp2 = __shfl_sync(0xFFFFFFFF, p_tmp1, offset*(WARP_SIZE/4)+(LANE/4));
	// If the value was indeed prepared by the source thread, store it.
	if ((LANE & 0x3) == 0) {
		// Value of interest is ready to be fetched.
		p_result = p_tmp2;
	}
	// If the index of the p1 element of the thread is within the range of interest, prepare it for communication.
	if (LANE/(WARP_SIZE/4) == offset) {
		p_tmp1 = p1;
	}
	__syncwarp();
	// Retrieve from the thread that holds the element of interest for the current thread the prepared value (if it exists).
	p_tmp2 = __shfl_sync(0xFFFFFFFF, p_tmp1, offset*(WARP_SIZE/4)+(LANE/4));
	// If the value was indeed prepared by the source thread, store it.
	if ((LANE & 0x3) == 1) {
		// Value of interest is ready to be fetched.
		p_result = p_tmp2;
	}
	// If the index of the p2 element of the thread is within the range of interest, prepare it for communication.
	if (LANE/(WARP_SIZE/4) == offset) {
		p_tmp1 = p2;
	}
	__syncwarp();
	// Retrieve from the thread that holds the element of interest for the current thread the prepared value (if it exists).
	p_tmp2 = __shfl_sync(0xFFFFFFFF, p_tmp1, offset*(WARP_SIZE/4)+(LANE/4));
	// If the value was indeed prepared by the source thread, store it.
	if ((LANE & 0x3) == 2) {
		// Value of interest is ready to be fetched.
		p_result = p_tmp2;
	}
	// If the index of the p3 element of the thread is within the range of interest, prepare it for communication.
	if (LANE/(WARP_SIZE/4) == offset) {
		p_tmp1 = p3;
	}
	__syncwarp();
	// Retrieve from the thread that holds the element of interest for the current thread the prepared value (if it exists).
	p_tmp2 = __shfl_sync(0xFFFFFFFF, p_tmp1, offset*(WARP_SIZE/4)+(LANE/4));
	// If the value was indeed prepared by the source thread, store it.
	if ((LANE & 0x3) == 3) {
		// Value of interest is ready to be fetched.
		p_result = p_tmp2;
	}
	return p_result;
}

//*** END FUNCTIONS FOR INTRA-WARP BITONIC MERGESORT ***

// Exploration functions to traverse outgoing transitions of the various states.
inline __device__ Storage_mode explore_globalObject_P_0(shared_indextype node_index, compressed_nodetype *d_q, nodetype *d_q_i, bool *d_dummy, volatile uint8_t *d_newstate_flags) {
	// Fetch the current state of the state machine.
	statetype current;
	get_current_state(&current, node_index, 0);
	statetype target = NO_STATE;
	nodetype part1, part2;
	// Storage mode to determine where to store the node(s).
	Storage_mode mode;
	shared_inttype part_cachepointers;
	switch (current) {
		case 0:
			{
			// Allocate register memory to process transition(s).
			indextype bufaddr_0;
			elem_chartype buf8_0, buf8_1;
			
			// NCS --{ [ myplace := next; next := next + 1 ] }--> p1
			
			mode = STORED;
			// Fetch values of unguarded variables.
			part1 = get_vectorpart(node_index, 0);
			part2 = part1;
			get_globalObject_next(&buf8_1, part1, part2);
			
			// Statement computation.
			target = 1;
			buf8_0 = (elem_chartype) (buf8_1);
			buf8_1 = (elem_chartype) (buf8_1 + 1);
			mode = (mode == STORED ? TO_CACHE : TO_GLOBAL);
			while (mode != STORED && mode != GLOBAL_STORED) {
				// Store new state vector in the cache or the global hash table.
				get_vectortree_node(&part1, &part_cachepointers, node_index, 1);
				// Store new values.
				part2 = part1;
				set_left_globalObject_P_0_myplace(&part2, buf8_0);
				set_left_globalObject_P_0(&part2, (statetype) target);
				set_left_globalObject_next(&part2, buf8_1);
				bufaddr_0 = STORENODE(mode, d_q, d_q_i, d_dummy, part1, &part2, &part_cachepointers, false, false, d_newstate_flags);
				if (bufaddr_0 == CACHE_FULL) {
					// Construct the vector again, and store it directly in the global hash table.
					mode = TO_GLOBAL;
					continue;
				}
				else if (bufaddr_0 == HASHTABLE_FULL) {
					// Hash table is considered full. Report this back.
					return HASH_TABLE_FULL;
				}
				if (bufaddr_0 != EMPTY_HASH_POINTER) {
					get_vectortree_node(&part1, &part_cachepointers, node_index, 0);
					part2 = part1;
					if (mode == TO_CACHE) {
						set_left_cache_pointer(&part_cachepointers, bufaddr_0);
						reset_left_in_vectortree_node(&part2);
						mark_root(&part2);
						mark_cached_node_new_nonleaf(&part_cachepointers);
						bufaddr_0 = STOREINCACHE(part2, part_cachepointers, false);
						if (bufaddr_0 == CACHE_FULL) {
							// Construct the vector again, and store it directly in the global hash table.
							mode = TO_GLOBAL;
							continue;
						}
					}
					else {
						set_left_in_vectortree_node(&part2, bufaddr_0);
						bufaddr_0 = FINDORPUT_SINGLE(d_q, d_q_i, d_dummy, part2, d_newstate_flags, EMPTY_CACHE_POINTER, true, (ITERATIONS == d_kernel_iters-1));
						if (bufaddr_0 == HASHTABLE_FULL) {
							// Hash table is considered full. Report this back.
							return HASH_TABLE_FULL;
						}
					}
				}
				mode = (mode == TO_CACHE ? STORED : GLOBAL_STORED);
			}
			}
			return STORED;
		case 1:
			{
			// Allocate register memory to process transition(s).
			indextype bufaddr_0;
			elem_chartype buf8_0, buf8_1;
			
			// p1 --{ [ myplace = 5 - 1; next := next - 5 ] }--> p2
			
			mode = STORED;
			// Fetch values of unguarded variables.
			part1 = get_vectorpart(node_index, 0);
			part2 = part1;
			get_globalObject_P_0_myplace(&buf8_0, part1, part2);
			part2 = part1;
			get_globalObject_next(&buf8_1, part1, part2);
			
			// Statement computation.
			if (buf8_0 == 5 - 1) {
				target = 2;
				buf8_1 = (elem_chartype) (buf8_1 - 5);
				mode = (mode == STORED ? TO_CACHE : TO_GLOBAL);
				while (mode != STORED && mode != GLOBAL_STORED) {
					// Store new state vector in the cache or the global hash table.
					get_vectortree_node(&part1, &part_cachepointers, node_index, 1);
					// Store new values.
					part2 = part1;
					set_left_globalObject_P_0(&part2, (statetype) target);
					set_left_globalObject_next(&part2, buf8_1);
					bufaddr_0 = STORENODE(mode, d_q, d_q_i, d_dummy, part1, &part2, &part_cachepointers, false, false, d_newstate_flags);
					if (bufaddr_0 == CACHE_FULL) {
						// Construct the vector again, and store it directly in the global hash table.
						mode = TO_GLOBAL;
						continue;
					}
					else if (bufaddr_0 == HASHTABLE_FULL) {
						// Hash table is considered full. Report this back.
						return HASH_TABLE_FULL;
					}
					if (bufaddr_0 != EMPTY_HASH_POINTER) {
						get_vectortree_node(&part1, &part_cachepointers, node_index, 0);
						part2 = part1;
						if (mode == TO_CACHE) {
							set_left_cache_pointer(&part_cachepointers, bufaddr_0);
							reset_left_in_vectortree_node(&part2);
							mark_root(&part2);
							mark_cached_node_new_nonleaf(&part_cachepointers);
							bufaddr_0 = STOREINCACHE(part2, part_cachepointers, false);
							if (bufaddr_0 == CACHE_FULL) {
								// Construct the vector again, and store it directly in the global hash table.
								mode = TO_GLOBAL;
								continue;
							}
						}
						else {
							set_left_in_vectortree_node(&part2, bufaddr_0);
							bufaddr_0 = FINDORPUT_SINGLE(d_q, d_q_i, d_dummy, part2, d_newstate_flags, EMPTY_CACHE_POINTER, true, (ITERATIONS == d_kernel_iters-1));
							if (bufaddr_0 == HASHTABLE_FULL) {
								// Hash table is considered full. Report this back.
								return HASH_TABLE_FULL;
							}
						}
					}
					mode = (mode == TO_CACHE ? STORED : GLOBAL_STORED);
				}
			}
			
			// p1 --{ [ myplace <> 5 - 1; myplace := myplace % 5 ] }--> p2
			
			mode = STORED;
			// Fetch values of unguarded variables.
			part1 = get_vectorpart(node_index, 0);
			part2 = part1;
			get_globalObject_P_0_myplace(&buf8_0, part1, part2);
			
			// Statement computation.
			if (buf8_0 != 5 - 1) {
				target = 2;
				buf8_0 = (elem_chartype) (buf8_0 % 5);
				mode = (mode == STORED ? TO_CACHE : TO_GLOBAL);
				while (mode != STORED && mode != GLOBAL_STORED) {
					// Store new state vector in the cache or the global hash table.
					get_vectortree_node(&part1, &part_cachepointers, node_index, 1);
					// Store new values.
					part2 = part1;
					set_left_globalObject_P_0_myplace(&part2, buf8_0);
					set_left_globalObject_P_0(&part2, (statetype) target);
					bufaddr_0 = STORENODE(mode, d_q, d_q_i, d_dummy, part1, &part2, &part_cachepointers, false, false, d_newstate_flags);
					if (bufaddr_0 == CACHE_FULL) {
						// Construct the vector again, and store it directly in the global hash table.
						mode = TO_GLOBAL;
						continue;
					}
					else if (bufaddr_0 == HASHTABLE_FULL) {
						// Hash table is considered full. Report this back.
						return HASH_TABLE_FULL;
					}
					if (bufaddr_0 != EMPTY_HASH_POINTER) {
						get_vectortree_node(&part1, &part_cachepointers, node_index, 0);
						part2 = part1;
						if (mode == TO_CACHE) {
							set_left_cache_pointer(&part_cachepointers, bufaddr_0);
							reset_left_in_vectortree_node(&part2);
							mark_root(&part2);
							mark_cached_node_new_nonleaf(&part_cachepointers);
							bufaddr_0 = STOREINCACHE(part2, part_cachepointers, false);
							if (bufaddr_0 == CACHE_FULL) {
								// Construct the vector again, and store it directly in the global hash table.
								mode = TO_GLOBAL;
								continue;
							}
						}
						else {
							set_left_in_vectortree_node(&part2, bufaddr_0);
							bufaddr_0 = FINDORPUT_SINGLE(d_q, d_q_i, d_dummy, part2, d_newstate_flags, EMPTY_CACHE_POINTER, true, (ITERATIONS == d_kernel_iters-1));
							if (bufaddr_0 == HASHTABLE_FULL) {
								// Hash table is considered full. Report this back.
								return HASH_TABLE_FULL;
							}
						}
					}
					mode = (mode == TO_CACHE ? STORED : GLOBAL_STORED);
				}
			}
			}
			return STORED;
		case 2:
			{
			// Allocate register memory to process transition(s).
			indextype bufaddr_0;
			elem_chartype buf8_0, buf8_1;
			// Allocate register memory for dynamic array indexing.
			array_indextype idx_0;
			
			// p2 --{ Slot[myplace] = 1 }--> p3
			
			mode = STORED;
			// Reset storage of array indices.
			idx_0 = EMPTY_INDEX;
			// Fetch values of unguarded variables.
			part1 = get_vectorpart(node_index, 0);
			part2 = part1;
			get_globalObject_P_0_myplace(&buf8_0, part1, part2);
			
			// Statement computation.
			if (globalObject_Slot(node_index, &idx_0, &buf8_1, buf8_0) == 1) {
				target = 3;
				mode = (mode == STORED ? TO_CACHE : TO_GLOBAL);
				while (mode != STORED && mode != GLOBAL_STORED) {
					// Store new state vector in the cache or the global hash table.
					get_vectortree_node(&part1, &part_cachepointers, node_index, 1);
					// Store new values.
					part2 = part1;
					set_left_globalObject_P_0(&part2, (statetype) target);
					bufaddr_0 = STORENODE(mode, d_q, d_q_i, d_dummy, part1, &part2, &part_cachepointers, false, false, d_newstate_flags);
					if (bufaddr_0 == CACHE_FULL) {
						// Construct the vector again, and store it directly in the global hash table.
						mode = TO_GLOBAL;
						continue;
					}
					else if (bufaddr_0 == HASHTABLE_FULL) {
						// Hash table is considered full. Report this back.
						return HASH_TABLE_FULL;
					}
					if (bufaddr_0 != EMPTY_HASH_POINTER) {
						get_vectortree_node(&part1, &part_cachepointers, node_index, 0);
						part2 = part1;
						if (mode == TO_CACHE) {
							set_left_cache_pointer(&part_cachepointers, bufaddr_0);
							reset_left_in_vectortree_node(&part2);
							mark_root(&part2);
							mark_cached_node_new_nonleaf(&part_cachepointers);
							bufaddr_0 = STOREINCACHE(part2, part_cachepointers, false);
							if (bufaddr_0 == CACHE_FULL) {
								// Construct the vector again, and store it directly in the global hash table.
								mode = TO_GLOBAL;
								continue;
							}
						}
						else {
							set_left_in_vectortree_node(&part2, bufaddr_0);
							bufaddr_0 = FINDORPUT_SINGLE(d_q, d_q_i, d_dummy, part2, d_newstate_flags, EMPTY_CACHE_POINTER, true, (ITERATIONS == d_kernel_iters-1));
							if (bufaddr_0 == HASHTABLE_FULL) {
								// Hash table is considered full. Report this back.
								return HASH_TABLE_FULL;
							}
						}
					}
					mode = (mode == TO_CACHE ? STORED : GLOBAL_STORED);
				}
			}
			}
			return STORED;
		case 3:
			{
			// Allocate register memory to process transition(s).
			indextype bufaddr_0, bufaddr_1;
			elem_chartype buf8_0, buf8_1;
			// Allocate register memory for dynamic array indexing.
			array_indextype idx_0;
			
			// p3 --{ [ Slot[(myplace + 5 - 1) % 5] := 0 ] }--> CS
			
			mode = STORED;
			// Reset storage of array indices.
			idx_0 = EMPTY_INDEX;
			// Fetch values of unguarded variables.
			part1 = get_vectorpart(node_index, 0);
			part2 = part1;
			get_globalObject_P_0_myplace(&buf8_1, part1, part2);
			
			// Statement computation.
			target = 4;
			A_STR_1(&idx_0, &buf8_0, (array_indextype) (buf8_1 + 5 - 1) % 5, (elem_chartype) 0);
			mode = (mode == STORED ? TO_CACHE : TO_GLOBAL);
			while (mode != STORED && mode != GLOBAL_STORED) {
				// Store new state vector in the cache or the global hash table.
				get_vectortree_node(&part1, &part_cachepointers, node_index, 1);
				// Store new values.
				part2 = part1;
				// Write array buffer content.
				if (0 >= 0 && 0 <= 1) {
					if (idx_0 != EMPTY_INDEX) {
						if (array_element_is_in_vectorpart_globalObject_Slot(idx_0, 0)) {
							if (is_left_vectorpart_for_array_element_globalObject_Slot(idx_0, 0)) {
								set_left_globalObject_Slot(&part2, idx_0, buf8_0, 0);
							}
						}
					}
				}
				set_left_globalObject_P_0(&part2, (statetype) target);
				bufaddr_0 = STORENODE(mode, d_q, d_q_i, d_dummy, part1, &part2, &part_cachepointers, false, false, d_newstate_flags);
				if (bufaddr_0 == CACHE_FULL) {
					// Construct the vector again, and store it directly in the global hash table.
					mode = TO_GLOBAL;
					continue;
				}
				else if (bufaddr_0 == HASHTABLE_FULL) {
					// Hash table is considered full. Report this back.
					return HASH_TABLE_FULL;
				}
				get_vectortree_node(&part1, &part_cachepointers, node_index, 2);
				// Store new values.
				part2 = part1;
				// Write array buffer content.
				if (1 >= 0 && 1 <= 1) {
					if (idx_0 != EMPTY_INDEX) {
						if (array_element_is_in_vectorpart_globalObject_Slot(idx_0, 1)) {
							if (is_left_vectorpart_for_array_element_globalObject_Slot(idx_0, 1)) {
								set_left_globalObject_Slot(&part2, idx_0, buf8_0, 1);
							}
						}
					}
				}
				bufaddr_1 = STORENODE(mode, d_q, d_q_i, d_dummy, part1, &part2, &part_cachepointers, false, false, d_newstate_flags);
				if (bufaddr_1 == CACHE_FULL) {
					// Construct the vector again, and store it directly in the global hash table.
					mode = TO_GLOBAL;
					continue;
				}
				else if (bufaddr_1 == HASHTABLE_FULL) {
					// Hash table is considered full. Report this back.
					return HASH_TABLE_FULL;
				}
				if (bufaddr_0 != EMPTY_HASH_POINTER || bufaddr_1 != EMPTY_HASH_POINTER) {
					get_vectortree_node(&part1, &part_cachepointers, node_index, 0);
					part2 = part1;
					if (bufaddr_0 != EMPTY_HASH_POINTER) {
						if (mode == TO_CACHE) {
							set_left_cache_pointer(&part_cachepointers, bufaddr_0);
							reset_left_in_vectortree_node(&part2);
						}
						else {
							set_left_in_vectortree_node(&part2, bufaddr_0);
						}
					}
					if (bufaddr_1 != EMPTY_HASH_POINTER) {
						if (mode == TO_CACHE) {
							set_right_cache_pointer(&part_cachepointers, bufaddr_1);
							reset_right_in_vectortree_node(&part2);
						}
						else {
							set_right_in_vectortree_node(&part2, bufaddr_1);
						}
					}
					if (mode == TO_CACHE) {
						// This part has been altered. Store it and remember address of new part.
						mark_root(&part2);
						mark_cached_node_new_nonleaf(&part_cachepointers);
						bufaddr_0 = STOREINCACHE(part2, part_cachepointers, false);
						if (bufaddr_0 == CACHE_FULL) {
							// Construct the vector again, and store it directly in the global hash table.
							mode = TO_GLOBAL;
							continue;
						}
					}
					else {
						bufaddr_0 = FINDORPUT_SINGLE(d_q, d_q_i, d_dummy, part2, d_newstate_flags, EMPTY_CACHE_POINTER, true, (ITERATIONS == d_kernel_iters-1));
						if (bufaddr_0 == HASHTABLE_FULL) {
							// Hash table is considered full. Report this back.
							return HASH_TABLE_FULL;
						}
					}
				}
				mode = (mode == TO_CACHE ? STORED : GLOBAL_STORED);
			}
			}
			return STORED;
		case 4:
			{
			// Allocate register memory to process transition(s).
			indextype bufaddr_0, bufaddr_1;
			elem_chartype buf8_0, buf8_1;
			// Allocate register memory for dynamic array indexing.
			array_indextype idx_0;
			
			// CS --{ [ Slot[(myplace + 1) % 5] := 1 ] }--> NCS
			
			mode = STORED;
			// Reset storage of array indices.
			idx_0 = EMPTY_INDEX;
			// Fetch values of unguarded variables.
			part1 = get_vectorpart(node_index, 0);
			part2 = part1;
			get_globalObject_P_0_myplace(&buf8_1, part1, part2);
			
			// Statement computation.
			target = 0;
			A_STR_1(&idx_0, &buf8_0, (array_indextype) (buf8_1 + 1) % 5, (elem_chartype) 1);
			mode = (mode == STORED ? TO_CACHE : TO_GLOBAL);
			while (mode != STORED && mode != GLOBAL_STORED) {
				// Store new state vector in the cache or the global hash table.
				get_vectortree_node(&part1, &part_cachepointers, node_index, 1);
				// Store new values.
				part2 = part1;
				// Write array buffer content.
				if (0 >= 0 && 0 <= 1) {
					if (idx_0 != EMPTY_INDEX) {
						if (array_element_is_in_vectorpart_globalObject_Slot(idx_0, 0)) {
							if (is_left_vectorpart_for_array_element_globalObject_Slot(idx_0, 0)) {
								set_left_globalObject_Slot(&part2, idx_0, buf8_0, 0);
							}
						}
					}
				}
				set_left_globalObject_P_0(&part2, (statetype) target);
				bufaddr_0 = STORENODE(mode, d_q, d_q_i, d_dummy, part1, &part2, &part_cachepointers, false, false, d_newstate_flags);
				if (bufaddr_0 == CACHE_FULL) {
					// Construct the vector again, and store it directly in the global hash table.
					mode = TO_GLOBAL;
					continue;
				}
				else if (bufaddr_0 == HASHTABLE_FULL) {
					// Hash table is considered full. Report this back.
					return HASH_TABLE_FULL;
				}
				get_vectortree_node(&part1, &part_cachepointers, node_index, 2);
				// Store new values.
				part2 = part1;
				// Write array buffer content.
				if (1 >= 0 && 1 <= 1) {
					if (idx_0 != EMPTY_INDEX) {
						if (array_element_is_in_vectorpart_globalObject_Slot(idx_0, 1)) {
							if (is_left_vectorpart_for_array_element_globalObject_Slot(idx_0, 1)) {
								set_left_globalObject_Slot(&part2, idx_0, buf8_0, 1);
							}
						}
					}
				}
				bufaddr_1 = STORENODE(mode, d_q, d_q_i, d_dummy, part1, &part2, &part_cachepointers, false, false, d_newstate_flags);
				if (bufaddr_1 == CACHE_FULL) {
					// Construct the vector again, and store it directly in the global hash table.
					mode = TO_GLOBAL;
					continue;
				}
				else if (bufaddr_1 == HASHTABLE_FULL) {
					// Hash table is considered full. Report this back.
					return HASH_TABLE_FULL;
				}
				if (bufaddr_0 != EMPTY_HASH_POINTER || bufaddr_1 != EMPTY_HASH_POINTER) {
					get_vectortree_node(&part1, &part_cachepointers, node_index, 0);
					part2 = part1;
					if (bufaddr_0 != EMPTY_HASH_POINTER) {
						if (mode == TO_CACHE) {
							set_left_cache_pointer(&part_cachepointers, bufaddr_0);
							reset_left_in_vectortree_node(&part2);
						}
						else {
							set_left_in_vectortree_node(&part2, bufaddr_0);
						}
					}
					if (bufaddr_1 != EMPTY_HASH_POINTER) {
						if (mode == TO_CACHE) {
							set_right_cache_pointer(&part_cachepointers, bufaddr_1);
							reset_right_in_vectortree_node(&part2);
						}
						else {
							set_right_in_vectortree_node(&part2, bufaddr_1);
						}
					}
					if (mode == TO_CACHE) {
						// This part has been altered. Store it and remember address of new part.
						mark_root(&part2);
						mark_cached_node_new_nonleaf(&part_cachepointers);
						bufaddr_0 = STOREINCACHE(part2, part_cachepointers, false);
						if (bufaddr_0 == CACHE_FULL) {
							// Construct the vector again, and store it directly in the global hash table.
							mode = TO_GLOBAL;
							continue;
						}
					}
					else {
						bufaddr_0 = FINDORPUT_SINGLE(d_q, d_q_i, d_dummy, part2, d_newstate_flags, EMPTY_CACHE_POINTER, true, (ITERATIONS == d_kernel_iters-1));
						if (bufaddr_0 == HASHTABLE_FULL) {
							// Hash table is considered full. Report this back.
							return HASH_TABLE_FULL;
						}
					}
				}
				mode = (mode == TO_CACHE ? STORED : GLOBAL_STORED);
			}
			}
			return STORED;
		default:
			return STORED;
	}
}

inline __device__ Storage_mode explore_globalObject_P_4(shared_indextype node_index, compressed_nodetype *d_q, nodetype *d_q_i, bool *d_dummy, volatile uint8_t *d_newstate_flags) {
	// Fetch the current state of the state machine.
	statetype current;
	get_current_state(&current, node_index, 1);
	statetype target = NO_STATE;
	nodetype part1, part2;
	// Storage mode to determine where to store the node(s).
	Storage_mode mode;
	shared_inttype part_cachepointers;
	switch (current) {
		case 0:
			{
			// Allocate register memory to process transition(s).
			indextype bufaddr_0, bufaddr_1;
			elem_chartype buf8_0, buf8_1;
			
			// NCS --{ [ myplace := next; next := next + 1 ] }--> p1
			
			mode = STORED;
			// Fetch values of unguarded variables.
			part1 = get_vectorpart(node_index, 0);
			part2 = part1;
			get_globalObject_next(&buf8_1, part1, part2);
			
			// Statement computation.
			target = 1;
			buf8_0 = (elem_chartype) (buf8_1);
			buf8_1 = (elem_chartype) (buf8_1 + 1);
			mode = (mode == STORED ? TO_CACHE : TO_GLOBAL);
			while (mode != STORED && mode != GLOBAL_STORED) {
				// Store new state vector in the cache or the global hash table.
				get_vectortree_node(&part1, &part_cachepointers, node_index, 1);
				// Store new values.
				part2 = part1;
				set_left_globalObject_P_4(&part2, (statetype) target);
				set_left_globalObject_next(&part2, buf8_1);
				bufaddr_0 = STORENODE(mode, d_q, d_q_i, d_dummy, part1, &part2, &part_cachepointers, false, false, d_newstate_flags);
				if (bufaddr_0 == CACHE_FULL) {
					// Construct the vector again, and store it directly in the global hash table.
					mode = TO_GLOBAL;
					continue;
				}
				else if (bufaddr_0 == HASHTABLE_FULL) {
					// Hash table is considered full. Report this back.
					return HASH_TABLE_FULL;
				}
				get_vectortree_node(&part1, &part_cachepointers, node_index, 2);
				// Store new values.
				part2 = part1;
				set_left_globalObject_P_4_myplace(&part2, buf8_0);
				bufaddr_1 = STORENODE(mode, d_q, d_q_i, d_dummy, part1, &part2, &part_cachepointers, false, false, d_newstate_flags);
				if (bufaddr_1 == CACHE_FULL) {
					// Construct the vector again, and store it directly in the global hash table.
					mode = TO_GLOBAL;
					continue;
				}
				else if (bufaddr_1 == HASHTABLE_FULL) {
					// Hash table is considered full. Report this back.
					return HASH_TABLE_FULL;
				}
				if (bufaddr_0 != EMPTY_HASH_POINTER || bufaddr_1 != EMPTY_HASH_POINTER) {
					get_vectortree_node(&part1, &part_cachepointers, node_index, 0);
					part2 = part1;
					if (bufaddr_0 != EMPTY_HASH_POINTER) {
						if (mode == TO_CACHE) {
							set_left_cache_pointer(&part_cachepointers, bufaddr_0);
							reset_left_in_vectortree_node(&part2);
						}
						else {
							set_left_in_vectortree_node(&part2, bufaddr_0);
						}
					}
					if (bufaddr_1 != EMPTY_HASH_POINTER) {
						if (mode == TO_CACHE) {
							set_right_cache_pointer(&part_cachepointers, bufaddr_1);
							reset_right_in_vectortree_node(&part2);
						}
						else {
							set_right_in_vectortree_node(&part2, bufaddr_1);
						}
					}
					if (mode == TO_CACHE) {
						// This part has been altered. Store it and remember address of new part.
						mark_root(&part2);
						mark_cached_node_new_nonleaf(&part_cachepointers);
						bufaddr_0 = STOREINCACHE(part2, part_cachepointers, false);
						if (bufaddr_0 == CACHE_FULL) {
							// Construct the vector again, and store it directly in the global hash table.
							mode = TO_GLOBAL;
							continue;
						}
					}
					else {
						bufaddr_0 = FINDORPUT_SINGLE(d_q, d_q_i, d_dummy, part2, d_newstate_flags, EMPTY_CACHE_POINTER, true, (ITERATIONS == d_kernel_iters-1));
						if (bufaddr_0 == HASHTABLE_FULL) {
							// Hash table is considered full. Report this back.
							return HASH_TABLE_FULL;
						}
					}
				}
				mode = (mode == TO_CACHE ? STORED : GLOBAL_STORED);
			}
			}
			return STORED;
		case 1:
			{
			// Allocate register memory to process transition(s).
			indextype bufaddr_0, bufaddr_1;
			elem_chartype buf8_0, buf8_1;
			
			// p1 --{ [ myplace = 5 - 1; next := next - 5 ] }--> p2
			
			mode = STORED;
			// Fetch values of unguarded variables.
			part1 = get_vectorpart(node_index, 1);
			part2 = part1;
			get_globalObject_P_4_myplace(&buf8_0, part1, part2);
			
			// Statement computation.
			if (buf8_0 == 5 - 1) {
				// Fetch values of guarded variables.
				part1 = get_vectorpart(node_index, 0);
				part2 = part1;
				get_globalObject_next(&buf8_1, part1, part2);
				
				target = 2;
				buf8_1 = (elem_chartype) (buf8_1 - 5);
				mode = (mode == STORED ? TO_CACHE : TO_GLOBAL);
				while (mode != STORED && mode != GLOBAL_STORED) {
					// Store new state vector in the cache or the global hash table.
					get_vectortree_node(&part1, &part_cachepointers, node_index, 1);
					// Store new values.
					part2 = part1;
					set_left_globalObject_P_4(&part2, (statetype) target);
					set_left_globalObject_next(&part2, buf8_1);
					bufaddr_0 = STORENODE(mode, d_q, d_q_i, d_dummy, part1, &part2, &part_cachepointers, false, false, d_newstate_flags);
					if (bufaddr_0 == CACHE_FULL) {
						// Construct the vector again, and store it directly in the global hash table.
						mode = TO_GLOBAL;
						continue;
					}
					else if (bufaddr_0 == HASHTABLE_FULL) {
						// Hash table is considered full. Report this back.
						return HASH_TABLE_FULL;
					}
					if (bufaddr_0 != EMPTY_HASH_POINTER) {
						get_vectortree_node(&part1, &part_cachepointers, node_index, 0);
						part2 = part1;
						if (mode == TO_CACHE) {
							set_left_cache_pointer(&part_cachepointers, bufaddr_0);
							reset_left_in_vectortree_node(&part2);
							mark_root(&part2);
							mark_cached_node_new_nonleaf(&part_cachepointers);
							bufaddr_0 = STOREINCACHE(part2, part_cachepointers, false);
							if (bufaddr_0 == CACHE_FULL) {
								// Construct the vector again, and store it directly in the global hash table.
								mode = TO_GLOBAL;
								continue;
							}
						}
						else {
							set_left_in_vectortree_node(&part2, bufaddr_0);
							bufaddr_0 = FINDORPUT_SINGLE(d_q, d_q_i, d_dummy, part2, d_newstate_flags, EMPTY_CACHE_POINTER, true, (ITERATIONS == d_kernel_iters-1));
							if (bufaddr_0 == HASHTABLE_FULL) {
								// Hash table is considered full. Report this back.
								return HASH_TABLE_FULL;
							}
						}
					}
					mode = (mode == TO_CACHE ? STORED : GLOBAL_STORED);
				}
			}
			
			// p1 --{ [ myplace <> 5 - 1; myplace := myplace % 5 ] }--> p2
			
			mode = STORED;
			// Fetch values of unguarded variables.
			part1 = get_vectorpart(node_index, 1);
			part2 = part1;
			get_globalObject_P_4_myplace(&buf8_0, part1, part2);
			
			// Statement computation.
			if (buf8_0 != 5 - 1) {
				target = 2;
				buf8_0 = (elem_chartype) (buf8_0 % 5);
				mode = (mode == STORED ? TO_CACHE : TO_GLOBAL);
				while (mode != STORED && mode != GLOBAL_STORED) {
					// Store new state vector in the cache or the global hash table.
					get_vectortree_node(&part1, &part_cachepointers, node_index, 1);
					// Store new values.
					part2 = part1;
					set_left_globalObject_P_4(&part2, (statetype) target);
					bufaddr_0 = STORENODE(mode, d_q, d_q_i, d_dummy, part1, &part2, &part_cachepointers, false, false, d_newstate_flags);
					if (bufaddr_0 == CACHE_FULL) {
						// Construct the vector again, and store it directly in the global hash table.
						mode = TO_GLOBAL;
						continue;
					}
					else if (bufaddr_0 == HASHTABLE_FULL) {
						// Hash table is considered full. Report this back.
						return HASH_TABLE_FULL;
					}
					get_vectortree_node(&part1, &part_cachepointers, node_index, 2);
					// Store new values.
					part2 = part1;
					set_left_globalObject_P_4_myplace(&part2, buf8_0);
					bufaddr_1 = STORENODE(mode, d_q, d_q_i, d_dummy, part1, &part2, &part_cachepointers, false, false, d_newstate_flags);
					if (bufaddr_1 == CACHE_FULL) {
						// Construct the vector again, and store it directly in the global hash table.
						mode = TO_GLOBAL;
						continue;
					}
					else if (bufaddr_1 == HASHTABLE_FULL) {
						// Hash table is considered full. Report this back.
						return HASH_TABLE_FULL;
					}
					if (bufaddr_0 != EMPTY_HASH_POINTER || bufaddr_1 != EMPTY_HASH_POINTER) {
						get_vectortree_node(&part1, &part_cachepointers, node_index, 0);
						part2 = part1;
						if (bufaddr_0 != EMPTY_HASH_POINTER) {
							if (mode == TO_CACHE) {
								set_left_cache_pointer(&part_cachepointers, bufaddr_0);
								reset_left_in_vectortree_node(&part2);
							}
							else {
								set_left_in_vectortree_node(&part2, bufaddr_0);
							}
						}
						if (bufaddr_1 != EMPTY_HASH_POINTER) {
							if (mode == TO_CACHE) {
								set_right_cache_pointer(&part_cachepointers, bufaddr_1);
								reset_right_in_vectortree_node(&part2);
							}
							else {
								set_right_in_vectortree_node(&part2, bufaddr_1);
							}
						}
						if (mode == TO_CACHE) {
							// This part has been altered. Store it and remember address of new part.
							mark_root(&part2);
							mark_cached_node_new_nonleaf(&part_cachepointers);
							bufaddr_0 = STOREINCACHE(part2, part_cachepointers, false);
							if (bufaddr_0 == CACHE_FULL) {
								// Construct the vector again, and store it directly in the global hash table.
								mode = TO_GLOBAL;
								continue;
							}
						}
						else {
							bufaddr_0 = FINDORPUT_SINGLE(d_q, d_q_i, d_dummy, part2, d_newstate_flags, EMPTY_CACHE_POINTER, true, (ITERATIONS == d_kernel_iters-1));
							if (bufaddr_0 == HASHTABLE_FULL) {
								// Hash table is considered full. Report this back.
								return HASH_TABLE_FULL;
							}
						}
					}
					mode = (mode == TO_CACHE ? STORED : GLOBAL_STORED);
				}
			}
			}
			return STORED;
		case 2:
			{
			// Allocate register memory to process transition(s).
			indextype bufaddr_0;
			elem_chartype buf8_0, buf8_1;
			// Allocate register memory for dynamic array indexing.
			array_indextype idx_0;
			
			// p2 --{ Slot[myplace] = 1 }--> p3
			
			mode = STORED;
			// Reset storage of array indices.
			idx_0 = EMPTY_INDEX;
			// Fetch values of unguarded variables.
			part1 = get_vectorpart(node_index, 1);
			part2 = part1;
			get_globalObject_P_4_myplace(&buf8_0, part1, part2);
			
			// Statement computation.
			if (globalObject_Slot(node_index, &idx_0, &buf8_1, buf8_0) == 1) {
				target = 3;
				mode = (mode == STORED ? TO_CACHE : TO_GLOBAL);
				while (mode != STORED && mode != GLOBAL_STORED) {
					// Store new state vector in the cache or the global hash table.
					get_vectortree_node(&part1, &part_cachepointers, node_index, 1);
					// Store new values.
					part2 = part1;
					set_left_globalObject_P_4(&part2, (statetype) target);
					bufaddr_0 = STORENODE(mode, d_q, d_q_i, d_dummy, part1, &part2, &part_cachepointers, false, false, d_newstate_flags);
					if (bufaddr_0 == CACHE_FULL) {
						// Construct the vector again, and store it directly in the global hash table.
						mode = TO_GLOBAL;
						continue;
					}
					else if (bufaddr_0 == HASHTABLE_FULL) {
						// Hash table is considered full. Report this back.
						return HASH_TABLE_FULL;
					}
					if (bufaddr_0 != EMPTY_HASH_POINTER) {
						get_vectortree_node(&part1, &part_cachepointers, node_index, 0);
						part2 = part1;
						if (mode == TO_CACHE) {
							set_left_cache_pointer(&part_cachepointers, bufaddr_0);
							reset_left_in_vectortree_node(&part2);
							mark_root(&part2);
							mark_cached_node_new_nonleaf(&part_cachepointers);
							bufaddr_0 = STOREINCACHE(part2, part_cachepointers, false);
							if (bufaddr_0 == CACHE_FULL) {
								// Construct the vector again, and store it directly in the global hash table.
								mode = TO_GLOBAL;
								continue;
							}
						}
						else {
							set_left_in_vectortree_node(&part2, bufaddr_0);
							bufaddr_0 = FINDORPUT_SINGLE(d_q, d_q_i, d_dummy, part2, d_newstate_flags, EMPTY_CACHE_POINTER, true, (ITERATIONS == d_kernel_iters-1));
							if (bufaddr_0 == HASHTABLE_FULL) {
								// Hash table is considered full. Report this back.
								return HASH_TABLE_FULL;
							}
						}
					}
					mode = (mode == TO_CACHE ? STORED : GLOBAL_STORED);
				}
			}
			}
			return STORED;
		case 3:
			{
			// Allocate register memory to process transition(s).
			indextype bufaddr_0, bufaddr_1;
			elem_chartype buf8_0, buf8_1;
			// Allocate register memory for dynamic array indexing.
			array_indextype idx_0;
			
			// p3 --{ [ Slot[(myplace + 5 - 1) % 5] := 0 ] }--> CS
			
			mode = STORED;
			// Reset storage of array indices.
			idx_0 = EMPTY_INDEX;
			// Fetch values of unguarded variables.
			part1 = get_vectorpart(node_index, 1);
			part2 = part1;
			get_globalObject_P_4_myplace(&buf8_0, part1, part2);
			
			// Statement computation.
			target = 4;
			A_STR_1(&idx_0, &buf8_1, (array_indextype) (buf8_0 + 5 - 1) % 5, (elem_chartype) 0);
			mode = (mode == STORED ? TO_CACHE : TO_GLOBAL);
			while (mode != STORED && mode != GLOBAL_STORED) {
				// Store new state vector in the cache or the global hash table.
				get_vectortree_node(&part1, &part_cachepointers, node_index, 1);
				// Store new values.
				part2 = part1;
				// Write array buffer content.
				if (0 >= 0 && 0 <= 1) {
					if (idx_0 != EMPTY_INDEX) {
						if (array_element_is_in_vectorpart_globalObject_Slot(idx_0, 0)) {
							if (is_left_vectorpart_for_array_element_globalObject_Slot(idx_0, 0)) {
								set_left_globalObject_Slot(&part2, idx_0, buf8_1, 0);
							}
						}
					}
				}
				set_left_globalObject_P_4(&part2, (statetype) target);
				bufaddr_0 = STORENODE(mode, d_q, d_q_i, d_dummy, part1, &part2, &part_cachepointers, false, false, d_newstate_flags);
				if (bufaddr_0 == CACHE_FULL) {
					// Construct the vector again, and store it directly in the global hash table.
					mode = TO_GLOBAL;
					continue;
				}
				else if (bufaddr_0 == HASHTABLE_FULL) {
					// Hash table is considered full. Report this back.
					return HASH_TABLE_FULL;
				}
				get_vectortree_node(&part1, &part_cachepointers, node_index, 2);
				// Store new values.
				part2 = part1;
				// Write array buffer content.
				if (1 >= 0 && 1 <= 1) {
					if (idx_0 != EMPTY_INDEX) {
						if (array_element_is_in_vectorpart_globalObject_Slot(idx_0, 1)) {
							if (is_left_vectorpart_for_array_element_globalObject_Slot(idx_0, 1)) {
								set_left_globalObject_Slot(&part2, idx_0, buf8_1, 1);
							}
						}
					}
				}
				bufaddr_1 = STORENODE(mode, d_q, d_q_i, d_dummy, part1, &part2, &part_cachepointers, false, false, d_newstate_flags);
				if (bufaddr_1 == CACHE_FULL) {
					// Construct the vector again, and store it directly in the global hash table.
					mode = TO_GLOBAL;
					continue;
				}
				else if (bufaddr_1 == HASHTABLE_FULL) {
					// Hash table is considered full. Report this back.
					return HASH_TABLE_FULL;
				}
				if (bufaddr_0 != EMPTY_HASH_POINTER || bufaddr_1 != EMPTY_HASH_POINTER) {
					get_vectortree_node(&part1, &part_cachepointers, node_index, 0);
					part2 = part1;
					if (bufaddr_0 != EMPTY_HASH_POINTER) {
						if (mode == TO_CACHE) {
							set_left_cache_pointer(&part_cachepointers, bufaddr_0);
							reset_left_in_vectortree_node(&part2);
						}
						else {
							set_left_in_vectortree_node(&part2, bufaddr_0);
						}
					}
					if (bufaddr_1 != EMPTY_HASH_POINTER) {
						if (mode == TO_CACHE) {
							set_right_cache_pointer(&part_cachepointers, bufaddr_1);
							reset_right_in_vectortree_node(&part2);
						}
						else {
							set_right_in_vectortree_node(&part2, bufaddr_1);
						}
					}
					if (mode == TO_CACHE) {
						// This part has been altered. Store it and remember address of new part.
						mark_root(&part2);
						mark_cached_node_new_nonleaf(&part_cachepointers);
						bufaddr_0 = STOREINCACHE(part2, part_cachepointers, false);
						if (bufaddr_0 == CACHE_FULL) {
							// Construct the vector again, and store it directly in the global hash table.
							mode = TO_GLOBAL;
							continue;
						}
					}
					else {
						bufaddr_0 = FINDORPUT_SINGLE(d_q, d_q_i, d_dummy, part2, d_newstate_flags, EMPTY_CACHE_POINTER, true, (ITERATIONS == d_kernel_iters-1));
						if (bufaddr_0 == HASHTABLE_FULL) {
							// Hash table is considered full. Report this back.
							return HASH_TABLE_FULL;
						}
					}
				}
				mode = (mode == TO_CACHE ? STORED : GLOBAL_STORED);
			}
			}
			return STORED;
		case 4:
			{
			// Allocate register memory to process transition(s).
			indextype bufaddr_0, bufaddr_1;
			elem_chartype buf8_0, buf8_1;
			// Allocate register memory for dynamic array indexing.
			array_indextype idx_0;
			
			// CS --{ [ Slot[(myplace + 1) % 5] := 1 ] }--> NCS
			
			mode = STORED;
			// Reset storage of array indices.
			idx_0 = EMPTY_INDEX;
			// Fetch values of unguarded variables.
			part1 = get_vectorpart(node_index, 1);
			part2 = part1;
			get_globalObject_P_4_myplace(&buf8_0, part1, part2);
			
			// Statement computation.
			target = 0;
			A_STR_1(&idx_0, &buf8_1, (array_indextype) (buf8_0 + 1) % 5, (elem_chartype) 1);
			mode = (mode == STORED ? TO_CACHE : TO_GLOBAL);
			while (mode != STORED && mode != GLOBAL_STORED) {
				// Store new state vector in the cache or the global hash table.
				get_vectortree_node(&part1, &part_cachepointers, node_index, 1);
				// Store new values.
				part2 = part1;
				// Write array buffer content.
				if (0 >= 0 && 0 <= 1) {
					if (idx_0 != EMPTY_INDEX) {
						if (array_element_is_in_vectorpart_globalObject_Slot(idx_0, 0)) {
							if (is_left_vectorpart_for_array_element_globalObject_Slot(idx_0, 0)) {
								set_left_globalObject_Slot(&part2, idx_0, buf8_1, 0);
							}
						}
					}
				}
				set_left_globalObject_P_4(&part2, (statetype) target);
				bufaddr_0 = STORENODE(mode, d_q, d_q_i, d_dummy, part1, &part2, &part_cachepointers, false, false, d_newstate_flags);
				if (bufaddr_0 == CACHE_FULL) {
					// Construct the vector again, and store it directly in the global hash table.
					mode = TO_GLOBAL;
					continue;
				}
				else if (bufaddr_0 == HASHTABLE_FULL) {
					// Hash table is considered full. Report this back.
					return HASH_TABLE_FULL;
				}
				get_vectortree_node(&part1, &part_cachepointers, node_index, 2);
				// Store new values.
				part2 = part1;
				// Write array buffer content.
				if (1 >= 0 && 1 <= 1) {
					if (idx_0 != EMPTY_INDEX) {
						if (array_element_is_in_vectorpart_globalObject_Slot(idx_0, 1)) {
							if (is_left_vectorpart_for_array_element_globalObject_Slot(idx_0, 1)) {
								set_left_globalObject_Slot(&part2, idx_0, buf8_1, 1);
							}
						}
					}
				}
				bufaddr_1 = STORENODE(mode, d_q, d_q_i, d_dummy, part1, &part2, &part_cachepointers, false, false, d_newstate_flags);
				if (bufaddr_1 == CACHE_FULL) {
					// Construct the vector again, and store it directly in the global hash table.
					mode = TO_GLOBAL;
					continue;
				}
				else if (bufaddr_1 == HASHTABLE_FULL) {
					// Hash table is considered full. Report this back.
					return HASH_TABLE_FULL;
				}
				if (bufaddr_0 != EMPTY_HASH_POINTER || bufaddr_1 != EMPTY_HASH_POINTER) {
					get_vectortree_node(&part1, &part_cachepointers, node_index, 0);
					part2 = part1;
					if (bufaddr_0 != EMPTY_HASH_POINTER) {
						if (mode == TO_CACHE) {
							set_left_cache_pointer(&part_cachepointers, bufaddr_0);
							reset_left_in_vectortree_node(&part2);
						}
						else {
							set_left_in_vectortree_node(&part2, bufaddr_0);
						}
					}
					if (bufaddr_1 != EMPTY_HASH_POINTER) {
						if (mode == TO_CACHE) {
							set_right_cache_pointer(&part_cachepointers, bufaddr_1);
							reset_right_in_vectortree_node(&part2);
						}
						else {
							set_right_in_vectortree_node(&part2, bufaddr_1);
						}
					}
					if (mode == TO_CACHE) {
						// This part has been altered. Store it and remember address of new part.
						mark_root(&part2);
						mark_cached_node_new_nonleaf(&part_cachepointers);
						bufaddr_0 = STOREINCACHE(part2, part_cachepointers, false);
						if (bufaddr_0 == CACHE_FULL) {
							// Construct the vector again, and store it directly in the global hash table.
							mode = TO_GLOBAL;
							continue;
						}
					}
					else {
						bufaddr_0 = FINDORPUT_SINGLE(d_q, d_q_i, d_dummy, part2, d_newstate_flags, EMPTY_CACHE_POINTER, true, (ITERATIONS == d_kernel_iters-1));
						if (bufaddr_0 == HASHTABLE_FULL) {
							// Hash table is considered full. Report this back.
							return HASH_TABLE_FULL;
						}
					}
				}
				mode = (mode == TO_CACHE ? STORED : GLOBAL_STORED);
			}
			}
			return STORED;
		default:
			return STORED;
	}
}

inline __device__ Storage_mode explore_globalObject_P_2(shared_indextype node_index, compressed_nodetype *d_q, nodetype *d_q_i, bool *d_dummy, volatile uint8_t *d_newstate_flags) {
	// Fetch the current state of the state machine.
	statetype current;
	get_current_state(&current, node_index, 2);
	statetype target = NO_STATE;
	nodetype part1, part2;
	// Storage mode to determine where to store the node(s).
	Storage_mode mode;
	shared_inttype part_cachepointers;
	switch (current) {
		case 0:
			{
			// Allocate register memory to process transition(s).
			indextype bufaddr_0, bufaddr_1;
			elem_chartype buf8_0, buf8_1;
			
			// NCS --{ [ myplace := next; next := next + 1 ] }--> p1
			
			mode = STORED;
			// Fetch values of unguarded variables.
			part1 = get_vectorpart(node_index, 0);
			part2 = part1;
			get_globalObject_next(&buf8_1, part1, part2);
			
			// Statement computation.
			target = 1;
			buf8_0 = (elem_chartype) (buf8_1);
			buf8_1 = (elem_chartype) (buf8_1 + 1);
			mode = (mode == STORED ? TO_CACHE : TO_GLOBAL);
			while (mode != STORED && mode != GLOBAL_STORED) {
				// Store new state vector in the cache or the global hash table.
				get_vectortree_node(&part1, &part_cachepointers, node_index, 1);
				// Store new values.
				part2 = part1;
				set_left_globalObject_P_2(&part2, (statetype) target);
				set_left_globalObject_next(&part2, buf8_1);
				bufaddr_0 = STORENODE(mode, d_q, d_q_i, d_dummy, part1, &part2, &part_cachepointers, false, false, d_newstate_flags);
				if (bufaddr_0 == CACHE_FULL) {
					// Construct the vector again, and store it directly in the global hash table.
					mode = TO_GLOBAL;
					continue;
				}
				else if (bufaddr_0 == HASHTABLE_FULL) {
					// Hash table is considered full. Report this back.
					return HASH_TABLE_FULL;
				}
				get_vectortree_node(&part1, &part_cachepointers, node_index, 2);
				// Store new values.
				part2 = part1;
				set_left_globalObject_P_2_myplace(&part2, buf8_0);
				bufaddr_1 = STORENODE(mode, d_q, d_q_i, d_dummy, part1, &part2, &part_cachepointers, false, false, d_newstate_flags);
				if (bufaddr_1 == CACHE_FULL) {
					// Construct the vector again, and store it directly in the global hash table.
					mode = TO_GLOBAL;
					continue;
				}
				else if (bufaddr_1 == HASHTABLE_FULL) {
					// Hash table is considered full. Report this back.
					return HASH_TABLE_FULL;
				}
				if (bufaddr_0 != EMPTY_HASH_POINTER || bufaddr_1 != EMPTY_HASH_POINTER) {
					get_vectortree_node(&part1, &part_cachepointers, node_index, 0);
					part2 = part1;
					if (bufaddr_0 != EMPTY_HASH_POINTER) {
						if (mode == TO_CACHE) {
							set_left_cache_pointer(&part_cachepointers, bufaddr_0);
							reset_left_in_vectortree_node(&part2);
						}
						else {
							set_left_in_vectortree_node(&part2, bufaddr_0);
						}
					}
					if (bufaddr_1 != EMPTY_HASH_POINTER) {
						if (mode == TO_CACHE) {
							set_right_cache_pointer(&part_cachepointers, bufaddr_1);
							reset_right_in_vectortree_node(&part2);
						}
						else {
							set_right_in_vectortree_node(&part2, bufaddr_1);
						}
					}
					if (mode == TO_CACHE) {
						// This part has been altered. Store it and remember address of new part.
						mark_root(&part2);
						mark_cached_node_new_nonleaf(&part_cachepointers);
						bufaddr_0 = STOREINCACHE(part2, part_cachepointers, false);
						if (bufaddr_0 == CACHE_FULL) {
							// Construct the vector again, and store it directly in the global hash table.
							mode = TO_GLOBAL;
							continue;
						}
					}
					else {
						bufaddr_0 = FINDORPUT_SINGLE(d_q, d_q_i, d_dummy, part2, d_newstate_flags, EMPTY_CACHE_POINTER, true, (ITERATIONS == d_kernel_iters-1));
						if (bufaddr_0 == HASHTABLE_FULL) {
							// Hash table is considered full. Report this back.
							return HASH_TABLE_FULL;
						}
					}
				}
				mode = (mode == TO_CACHE ? STORED : GLOBAL_STORED);
			}
			}
			return STORED;
		case 1:
			{
			// Allocate register memory to process transition(s).
			indextype bufaddr_0, bufaddr_1;
			elem_chartype buf8_0, buf8_1;
			
			// p1 --{ [ myplace = 5 - 1; next := next - 5 ] }--> p2
			
			mode = STORED;
			// Fetch values of unguarded variables.
			part1 = get_vectorpart(node_index, 1);
			part2 = part1;
			get_globalObject_P_2_myplace(&buf8_0, part1, part2);
			
			// Statement computation.
			if (buf8_0 == 5 - 1) {
				// Fetch values of guarded variables.
				part1 = get_vectorpart(node_index, 0);
				part2 = part1;
				get_globalObject_next(&buf8_1, part1, part2);
				
				target = 2;
				buf8_1 = (elem_chartype) (buf8_1 - 5);
				mode = (mode == STORED ? TO_CACHE : TO_GLOBAL);
				while (mode != STORED && mode != GLOBAL_STORED) {
					// Store new state vector in the cache or the global hash table.
					get_vectortree_node(&part1, &part_cachepointers, node_index, 1);
					// Store new values.
					part2 = part1;
					set_left_globalObject_P_2(&part2, (statetype) target);
					set_left_globalObject_next(&part2, buf8_1);
					bufaddr_0 = STORENODE(mode, d_q, d_q_i, d_dummy, part1, &part2, &part_cachepointers, false, false, d_newstate_flags);
					if (bufaddr_0 == CACHE_FULL) {
						// Construct the vector again, and store it directly in the global hash table.
						mode = TO_GLOBAL;
						continue;
					}
					else if (bufaddr_0 == HASHTABLE_FULL) {
						// Hash table is considered full. Report this back.
						return HASH_TABLE_FULL;
					}
					if (bufaddr_0 != EMPTY_HASH_POINTER) {
						get_vectortree_node(&part1, &part_cachepointers, node_index, 0);
						part2 = part1;
						if (mode == TO_CACHE) {
							set_left_cache_pointer(&part_cachepointers, bufaddr_0);
							reset_left_in_vectortree_node(&part2);
							mark_root(&part2);
							mark_cached_node_new_nonleaf(&part_cachepointers);
							bufaddr_0 = STOREINCACHE(part2, part_cachepointers, false);
							if (bufaddr_0 == CACHE_FULL) {
								// Construct the vector again, and store it directly in the global hash table.
								mode = TO_GLOBAL;
								continue;
							}
						}
						else {
							set_left_in_vectortree_node(&part2, bufaddr_0);
							bufaddr_0 = FINDORPUT_SINGLE(d_q, d_q_i, d_dummy, part2, d_newstate_flags, EMPTY_CACHE_POINTER, true, (ITERATIONS == d_kernel_iters-1));
							if (bufaddr_0 == HASHTABLE_FULL) {
								// Hash table is considered full. Report this back.
								return HASH_TABLE_FULL;
							}
						}
					}
					mode = (mode == TO_CACHE ? STORED : GLOBAL_STORED);
				}
			}
			
			// p1 --{ [ myplace <> 5 - 1; myplace := myplace % 5 ] }--> p2
			
			mode = STORED;
			// Fetch values of unguarded variables.
			part1 = get_vectorpart(node_index, 1);
			part2 = part1;
			get_globalObject_P_2_myplace(&buf8_0, part1, part2);
			
			// Statement computation.
			if (buf8_0 != 5 - 1) {
				target = 2;
				buf8_0 = (elem_chartype) (buf8_0 % 5);
				mode = (mode == STORED ? TO_CACHE : TO_GLOBAL);
				while (mode != STORED && mode != GLOBAL_STORED) {
					// Store new state vector in the cache or the global hash table.
					get_vectortree_node(&part1, &part_cachepointers, node_index, 1);
					// Store new values.
					part2 = part1;
					set_left_globalObject_P_2(&part2, (statetype) target);
					bufaddr_0 = STORENODE(mode, d_q, d_q_i, d_dummy, part1, &part2, &part_cachepointers, false, false, d_newstate_flags);
					if (bufaddr_0 == CACHE_FULL) {
						// Construct the vector again, and store it directly in the global hash table.
						mode = TO_GLOBAL;
						continue;
					}
					else if (bufaddr_0 == HASHTABLE_FULL) {
						// Hash table is considered full. Report this back.
						return HASH_TABLE_FULL;
					}
					get_vectortree_node(&part1, &part_cachepointers, node_index, 2);
					// Store new values.
					part2 = part1;
					set_left_globalObject_P_2_myplace(&part2, buf8_0);
					bufaddr_1 = STORENODE(mode, d_q, d_q_i, d_dummy, part1, &part2, &part_cachepointers, false, false, d_newstate_flags);
					if (bufaddr_1 == CACHE_FULL) {
						// Construct the vector again, and store it directly in the global hash table.
						mode = TO_GLOBAL;
						continue;
					}
					else if (bufaddr_1 == HASHTABLE_FULL) {
						// Hash table is considered full. Report this back.
						return HASH_TABLE_FULL;
					}
					if (bufaddr_0 != EMPTY_HASH_POINTER || bufaddr_1 != EMPTY_HASH_POINTER) {
						get_vectortree_node(&part1, &part_cachepointers, node_index, 0);
						part2 = part1;
						if (bufaddr_0 != EMPTY_HASH_POINTER) {
							if (mode == TO_CACHE) {
								set_left_cache_pointer(&part_cachepointers, bufaddr_0);
								reset_left_in_vectortree_node(&part2);
							}
							else {
								set_left_in_vectortree_node(&part2, bufaddr_0);
							}
						}
						if (bufaddr_1 != EMPTY_HASH_POINTER) {
							if (mode == TO_CACHE) {
								set_right_cache_pointer(&part_cachepointers, bufaddr_1);
								reset_right_in_vectortree_node(&part2);
							}
							else {
								set_right_in_vectortree_node(&part2, bufaddr_1);
							}
						}
						if (mode == TO_CACHE) {
							// This part has been altered. Store it and remember address of new part.
							mark_root(&part2);
							mark_cached_node_new_nonleaf(&part_cachepointers);
							bufaddr_0 = STOREINCACHE(part2, part_cachepointers, false);
							if (bufaddr_0 == CACHE_FULL) {
								// Construct the vector again, and store it directly in the global hash table.
								mode = TO_GLOBAL;
								continue;
							}
						}
						else {
							bufaddr_0 = FINDORPUT_SINGLE(d_q, d_q_i, d_dummy, part2, d_newstate_flags, EMPTY_CACHE_POINTER, true, (ITERATIONS == d_kernel_iters-1));
							if (bufaddr_0 == HASHTABLE_FULL) {
								// Hash table is considered full. Report this back.
								return HASH_TABLE_FULL;
							}
						}
					}
					mode = (mode == TO_CACHE ? STORED : GLOBAL_STORED);
				}
			}
			}
			return STORED;
		case 2:
			{
			// Allocate register memory to process transition(s).
			indextype bufaddr_0;
			elem_chartype buf8_0, buf8_1;
			// Allocate register memory for dynamic array indexing.
			array_indextype idx_0;
			
			// p2 --{ Slot[myplace] = 1 }--> p3
			
			mode = STORED;
			// Reset storage of array indices.
			idx_0 = EMPTY_INDEX;
			// Fetch values of unguarded variables.
			part1 = get_vectorpart(node_index, 1);
			part2 = part1;
			get_globalObject_P_2_myplace(&buf8_0, part1, part2);
			
			// Statement computation.
			if (globalObject_Slot(node_index, &idx_0, &buf8_1, buf8_0) == 1) {
				target = 3;
				mode = (mode == STORED ? TO_CACHE : TO_GLOBAL);
				while (mode != STORED && mode != GLOBAL_STORED) {
					// Store new state vector in the cache or the global hash table.
					get_vectortree_node(&part1, &part_cachepointers, node_index, 1);
					// Store new values.
					part2 = part1;
					set_left_globalObject_P_2(&part2, (statetype) target);
					bufaddr_0 = STORENODE(mode, d_q, d_q_i, d_dummy, part1, &part2, &part_cachepointers, false, false, d_newstate_flags);
					if (bufaddr_0 == CACHE_FULL) {
						// Construct the vector again, and store it directly in the global hash table.
						mode = TO_GLOBAL;
						continue;
					}
					else if (bufaddr_0 == HASHTABLE_FULL) {
						// Hash table is considered full. Report this back.
						return HASH_TABLE_FULL;
					}
					if (bufaddr_0 != EMPTY_HASH_POINTER) {
						get_vectortree_node(&part1, &part_cachepointers, node_index, 0);
						part2 = part1;
						if (mode == TO_CACHE) {
							set_left_cache_pointer(&part_cachepointers, bufaddr_0);
							reset_left_in_vectortree_node(&part2);
							mark_root(&part2);
							mark_cached_node_new_nonleaf(&part_cachepointers);
							bufaddr_0 = STOREINCACHE(part2, part_cachepointers, false);
							if (bufaddr_0 == CACHE_FULL) {
								// Construct the vector again, and store it directly in the global hash table.
								mode = TO_GLOBAL;
								continue;
							}
						}
						else {
							set_left_in_vectortree_node(&part2, bufaddr_0);
							bufaddr_0 = FINDORPUT_SINGLE(d_q, d_q_i, d_dummy, part2, d_newstate_flags, EMPTY_CACHE_POINTER, true, (ITERATIONS == d_kernel_iters-1));
							if (bufaddr_0 == HASHTABLE_FULL) {
								// Hash table is considered full. Report this back.
								return HASH_TABLE_FULL;
							}
						}
					}
					mode = (mode == TO_CACHE ? STORED : GLOBAL_STORED);
				}
			}
			}
			return STORED;
		case 3:
			{
			// Allocate register memory to process transition(s).
			indextype bufaddr_0, bufaddr_1;
			elem_chartype buf8_0, buf8_1;
			// Allocate register memory for dynamic array indexing.
			array_indextype idx_0;
			
			// p3 --{ [ Slot[(myplace + 5 - 1) % 5] := 0 ] }--> CS
			
			mode = STORED;
			// Reset storage of array indices.
			idx_0 = EMPTY_INDEX;
			// Fetch values of unguarded variables.
			part1 = get_vectorpart(node_index, 1);
			part2 = part1;
			get_globalObject_P_2_myplace(&buf8_0, part1, part2);
			
			// Statement computation.
			target = 4;
			A_STR_1(&idx_0, &buf8_1, (array_indextype) (buf8_0 + 5 - 1) % 5, (elem_chartype) 0);
			mode = (mode == STORED ? TO_CACHE : TO_GLOBAL);
			while (mode != STORED && mode != GLOBAL_STORED) {
				// Store new state vector in the cache or the global hash table.
				get_vectortree_node(&part1, &part_cachepointers, node_index, 1);
				// Store new values.
				part2 = part1;
				// Write array buffer content.
				if (0 >= 0 && 0 <= 1) {
					if (idx_0 != EMPTY_INDEX) {
						if (array_element_is_in_vectorpart_globalObject_Slot(idx_0, 0)) {
							if (is_left_vectorpart_for_array_element_globalObject_Slot(idx_0, 0)) {
								set_left_globalObject_Slot(&part2, idx_0, buf8_1, 0);
							}
						}
					}
				}
				set_left_globalObject_P_2(&part2, (statetype) target);
				bufaddr_0 = STORENODE(mode, d_q, d_q_i, d_dummy, part1, &part2, &part_cachepointers, false, false, d_newstate_flags);
				if (bufaddr_0 == CACHE_FULL) {
					// Construct the vector again, and store it directly in the global hash table.
					mode = TO_GLOBAL;
					continue;
				}
				else if (bufaddr_0 == HASHTABLE_FULL) {
					// Hash table is considered full. Report this back.
					return HASH_TABLE_FULL;
				}
				get_vectortree_node(&part1, &part_cachepointers, node_index, 2);
				// Store new values.
				part2 = part1;
				// Write array buffer content.
				if (1 >= 0 && 1 <= 1) {
					if (idx_0 != EMPTY_INDEX) {
						if (array_element_is_in_vectorpart_globalObject_Slot(idx_0, 1)) {
							if (is_left_vectorpart_for_array_element_globalObject_Slot(idx_0, 1)) {
								set_left_globalObject_Slot(&part2, idx_0, buf8_1, 1);
							}
						}
					}
				}
				bufaddr_1 = STORENODE(mode, d_q, d_q_i, d_dummy, part1, &part2, &part_cachepointers, false, false, d_newstate_flags);
				if (bufaddr_1 == CACHE_FULL) {
					// Construct the vector again, and store it directly in the global hash table.
					mode = TO_GLOBAL;
					continue;
				}
				else if (bufaddr_1 == HASHTABLE_FULL) {
					// Hash table is considered full. Report this back.
					return HASH_TABLE_FULL;
				}
				if (bufaddr_0 != EMPTY_HASH_POINTER || bufaddr_1 != EMPTY_HASH_POINTER) {
					get_vectortree_node(&part1, &part_cachepointers, node_index, 0);
					part2 = part1;
					if (bufaddr_0 != EMPTY_HASH_POINTER) {
						if (mode == TO_CACHE) {
							set_left_cache_pointer(&part_cachepointers, bufaddr_0);
							reset_left_in_vectortree_node(&part2);
						}
						else {
							set_left_in_vectortree_node(&part2, bufaddr_0);
						}
					}
					if (bufaddr_1 != EMPTY_HASH_POINTER) {
						if (mode == TO_CACHE) {
							set_right_cache_pointer(&part_cachepointers, bufaddr_1);
							reset_right_in_vectortree_node(&part2);
						}
						else {
							set_right_in_vectortree_node(&part2, bufaddr_1);
						}
					}
					if (mode == TO_CACHE) {
						// This part has been altered. Store it and remember address of new part.
						mark_root(&part2);
						mark_cached_node_new_nonleaf(&part_cachepointers);
						bufaddr_0 = STOREINCACHE(part2, part_cachepointers, false);
						if (bufaddr_0 == CACHE_FULL) {
							// Construct the vector again, and store it directly in the global hash table.
							mode = TO_GLOBAL;
							continue;
						}
					}
					else {
						bufaddr_0 = FINDORPUT_SINGLE(d_q, d_q_i, d_dummy, part2, d_newstate_flags, EMPTY_CACHE_POINTER, true, (ITERATIONS == d_kernel_iters-1));
						if (bufaddr_0 == HASHTABLE_FULL) {
							// Hash table is considered full. Report this back.
							return HASH_TABLE_FULL;
						}
					}
				}
				mode = (mode == TO_CACHE ? STORED : GLOBAL_STORED);
			}
			}
			return STORED;
		case 4:
			{
			// Allocate register memory to process transition(s).
			indextype bufaddr_0, bufaddr_1;
			elem_chartype buf8_0, buf8_1;
			// Allocate register memory for dynamic array indexing.
			array_indextype idx_0;
			
			// CS --{ [ Slot[(myplace + 1) % 5] := 1 ] }--> NCS
			
			mode = STORED;
			// Reset storage of array indices.
			idx_0 = EMPTY_INDEX;
			// Fetch values of unguarded variables.
			part1 = get_vectorpart(node_index, 1);
			part2 = part1;
			get_globalObject_P_2_myplace(&buf8_0, part1, part2);
			
			// Statement computation.
			target = 0;
			A_STR_1(&idx_0, &buf8_1, (array_indextype) (buf8_0 + 1) % 5, (elem_chartype) 1);
			mode = (mode == STORED ? TO_CACHE : TO_GLOBAL);
			while (mode != STORED && mode != GLOBAL_STORED) {
				// Store new state vector in the cache or the global hash table.
				get_vectortree_node(&part1, &part_cachepointers, node_index, 1);
				// Store new values.
				part2 = part1;
				// Write array buffer content.
				if (0 >= 0 && 0 <= 1) {
					if (idx_0 != EMPTY_INDEX) {
						if (array_element_is_in_vectorpart_globalObject_Slot(idx_0, 0)) {
							if (is_left_vectorpart_for_array_element_globalObject_Slot(idx_0, 0)) {
								set_left_globalObject_Slot(&part2, idx_0, buf8_1, 0);
							}
						}
					}
				}
				set_left_globalObject_P_2(&part2, (statetype) target);
				bufaddr_0 = STORENODE(mode, d_q, d_q_i, d_dummy, part1, &part2, &part_cachepointers, false, false, d_newstate_flags);
				if (bufaddr_0 == CACHE_FULL) {
					// Construct the vector again, and store it directly in the global hash table.
					mode = TO_GLOBAL;
					continue;
				}
				else if (bufaddr_0 == HASHTABLE_FULL) {
					// Hash table is considered full. Report this back.
					return HASH_TABLE_FULL;
				}
				get_vectortree_node(&part1, &part_cachepointers, node_index, 2);
				// Store new values.
				part2 = part1;
				// Write array buffer content.
				if (1 >= 0 && 1 <= 1) {
					if (idx_0 != EMPTY_INDEX) {
						if (array_element_is_in_vectorpart_globalObject_Slot(idx_0, 1)) {
							if (is_left_vectorpart_for_array_element_globalObject_Slot(idx_0, 1)) {
								set_left_globalObject_Slot(&part2, idx_0, buf8_1, 1);
							}
						}
					}
				}
				bufaddr_1 = STORENODE(mode, d_q, d_q_i, d_dummy, part1, &part2, &part_cachepointers, false, false, d_newstate_flags);
				if (bufaddr_1 == CACHE_FULL) {
					// Construct the vector again, and store it directly in the global hash table.
					mode = TO_GLOBAL;
					continue;
				}
				else if (bufaddr_1 == HASHTABLE_FULL) {
					// Hash table is considered full. Report this back.
					return HASH_TABLE_FULL;
				}
				if (bufaddr_0 != EMPTY_HASH_POINTER || bufaddr_1 != EMPTY_HASH_POINTER) {
					get_vectortree_node(&part1, &part_cachepointers, node_index, 0);
					part2 = part1;
					if (bufaddr_0 != EMPTY_HASH_POINTER) {
						if (mode == TO_CACHE) {
							set_left_cache_pointer(&part_cachepointers, bufaddr_0);
							reset_left_in_vectortree_node(&part2);
						}
						else {
							set_left_in_vectortree_node(&part2, bufaddr_0);
						}
					}
					if (bufaddr_1 != EMPTY_HASH_POINTER) {
						if (mode == TO_CACHE) {
							set_right_cache_pointer(&part_cachepointers, bufaddr_1);
							reset_right_in_vectortree_node(&part2);
						}
						else {
							set_right_in_vectortree_node(&part2, bufaddr_1);
						}
					}
					if (mode == TO_CACHE) {
						// This part has been altered. Store it and remember address of new part.
						mark_root(&part2);
						mark_cached_node_new_nonleaf(&part_cachepointers);
						bufaddr_0 = STOREINCACHE(part2, part_cachepointers, false);
						if (bufaddr_0 == CACHE_FULL) {
							// Construct the vector again, and store it directly in the global hash table.
							mode = TO_GLOBAL;
							continue;
						}
					}
					else {
						bufaddr_0 = FINDORPUT_SINGLE(d_q, d_q_i, d_dummy, part2, d_newstate_flags, EMPTY_CACHE_POINTER, true, (ITERATIONS == d_kernel_iters-1));
						if (bufaddr_0 == HASHTABLE_FULL) {
							// Hash table is considered full. Report this back.
							return HASH_TABLE_FULL;
						}
					}
				}
				mode = (mode == TO_CACHE ? STORED : GLOBAL_STORED);
			}
			}
			return STORED;
		default:
			return STORED;
	}
}

inline __device__ Storage_mode explore_globalObject_P_3(shared_indextype node_index, compressed_nodetype *d_q, nodetype *d_q_i, bool *d_dummy, volatile uint8_t *d_newstate_flags) {
	// Fetch the current state of the state machine.
	statetype current;
	get_current_state(&current, node_index, 3);
	statetype target = NO_STATE;
	nodetype part1, part2;
	// Storage mode to determine where to store the node(s).
	Storage_mode mode;
	shared_inttype part_cachepointers;
	switch (current) {
		case 0:
			{
			// Allocate register memory to process transition(s).
			indextype bufaddr_0;
			elem_chartype buf8_0, buf8_1;
			
			// NCS --{ [ myplace := next; next := next + 1 ] }--> p1
			
			mode = STORED;
			// Fetch values of unguarded variables.
			part1 = get_vectorpart(node_index, 0);
			part2 = part1;
			get_globalObject_next(&buf8_0, part1, part2);
			
			// Statement computation.
			target = 1;
			buf8_1 = (elem_chartype) (buf8_0);
			buf8_0 = (elem_chartype) (buf8_0 + 1);
			mode = (mode == STORED ? TO_CACHE : TO_GLOBAL);
			while (mode != STORED && mode != GLOBAL_STORED) {
				// Store new state vector in the cache or the global hash table.
				get_vectortree_node(&part1, &part_cachepointers, node_index, 1);
				// Store new values.
				part2 = part1;
				set_left_globalObject_next(&part2, buf8_0);
				set_left_globalObject_P_3(&part2, (statetype) target);
				set_left_globalObject_P_3_myplace(&part2, buf8_1);
				bufaddr_0 = STORENODE(mode, d_q, d_q_i, d_dummy, part1, &part2, &part_cachepointers, false, false, d_newstate_flags);
				if (bufaddr_0 == CACHE_FULL) {
					// Construct the vector again, and store it directly in the global hash table.
					mode = TO_GLOBAL;
					continue;
				}
				else if (bufaddr_0 == HASHTABLE_FULL) {
					// Hash table is considered full. Report this back.
					return HASH_TABLE_FULL;
				}
				if (bufaddr_0 != EMPTY_HASH_POINTER) {
					get_vectortree_node(&part1, &part_cachepointers, node_index, 0);
					part2 = part1;
					if (mode == TO_CACHE) {
						set_left_cache_pointer(&part_cachepointers, bufaddr_0);
						reset_left_in_vectortree_node(&part2);
						mark_root(&part2);
						mark_cached_node_new_nonleaf(&part_cachepointers);
						bufaddr_0 = STOREINCACHE(part2, part_cachepointers, false);
						if (bufaddr_0 == CACHE_FULL) {
							// Construct the vector again, and store it directly in the global hash table.
							mode = TO_GLOBAL;
							continue;
						}
					}
					else {
						set_left_in_vectortree_node(&part2, bufaddr_0);
						bufaddr_0 = FINDORPUT_SINGLE(d_q, d_q_i, d_dummy, part2, d_newstate_flags, EMPTY_CACHE_POINTER, true, (ITERATIONS == d_kernel_iters-1));
						if (bufaddr_0 == HASHTABLE_FULL) {
							// Hash table is considered full. Report this back.
							return HASH_TABLE_FULL;
						}
					}
				}
				mode = (mode == TO_CACHE ? STORED : GLOBAL_STORED);
			}
			}
			return STORED;
		case 1:
			{
			// Allocate register memory to process transition(s).
			indextype bufaddr_0;
			elem_chartype buf8_0, buf8_1;
			
			// p1 --{ [ myplace = 5 - 1; next := next - 5 ] }--> p2
			
			mode = STORED;
			// Fetch values of unguarded variables.
			part1 = get_vectorpart(node_index, 0);
			part2 = part1;
			get_globalObject_P_3_myplace(&buf8_1, part1, part2);
			part2 = part1;
			get_globalObject_next(&buf8_0, part1, part2);
			
			// Statement computation.
			if (buf8_1 == 5 - 1) {
				target = 2;
				buf8_0 = (elem_chartype) (buf8_0 - 5);
				mode = (mode == STORED ? TO_CACHE : TO_GLOBAL);
				while (mode != STORED && mode != GLOBAL_STORED) {
					// Store new state vector in the cache or the global hash table.
					get_vectortree_node(&part1, &part_cachepointers, node_index, 1);
					// Store new values.
					part2 = part1;
					set_left_globalObject_P_3(&part2, (statetype) target);
					set_left_globalObject_next(&part2, buf8_0);
					bufaddr_0 = STORENODE(mode, d_q, d_q_i, d_dummy, part1, &part2, &part_cachepointers, false, false, d_newstate_flags);
					if (bufaddr_0 == CACHE_FULL) {
						// Construct the vector again, and store it directly in the global hash table.
						mode = TO_GLOBAL;
						continue;
					}
					else if (bufaddr_0 == HASHTABLE_FULL) {
						// Hash table is considered full. Report this back.
						return HASH_TABLE_FULL;
					}
					if (bufaddr_0 != EMPTY_HASH_POINTER) {
						get_vectortree_node(&part1, &part_cachepointers, node_index, 0);
						part2 = part1;
						if (mode == TO_CACHE) {
							set_left_cache_pointer(&part_cachepointers, bufaddr_0);
							reset_left_in_vectortree_node(&part2);
							mark_root(&part2);
							mark_cached_node_new_nonleaf(&part_cachepointers);
							bufaddr_0 = STOREINCACHE(part2, part_cachepointers, false);
							if (bufaddr_0 == CACHE_FULL) {
								// Construct the vector again, and store it directly in the global hash table.
								mode = TO_GLOBAL;
								continue;
							}
						}
						else {
							set_left_in_vectortree_node(&part2, bufaddr_0);
							bufaddr_0 = FINDORPUT_SINGLE(d_q, d_q_i, d_dummy, part2, d_newstate_flags, EMPTY_CACHE_POINTER, true, (ITERATIONS == d_kernel_iters-1));
							if (bufaddr_0 == HASHTABLE_FULL) {
								// Hash table is considered full. Report this back.
								return HASH_TABLE_FULL;
							}
						}
					}
					mode = (mode == TO_CACHE ? STORED : GLOBAL_STORED);
				}
			}
			
			// p1 --{ [ myplace <> 5 - 1; myplace := myplace % 5 ] }--> p2
			
			mode = STORED;
			// Fetch values of unguarded variables.
			part1 = get_vectorpart(node_index, 0);
			part2 = part1;
			get_globalObject_P_3_myplace(&buf8_0, part1, part2);
			
			// Statement computation.
			if (buf8_0 != 5 - 1) {
				target = 2;
				buf8_0 = (elem_chartype) (buf8_0 % 5);
				mode = (mode == STORED ? TO_CACHE : TO_GLOBAL);
				while (mode != STORED && mode != GLOBAL_STORED) {
					// Store new state vector in the cache or the global hash table.
					get_vectortree_node(&part1, &part_cachepointers, node_index, 1);
					// Store new values.
					part2 = part1;
					set_left_globalObject_P_3(&part2, (statetype) target);
					set_left_globalObject_P_3_myplace(&part2, buf8_0);
					bufaddr_0 = STORENODE(mode, d_q, d_q_i, d_dummy, part1, &part2, &part_cachepointers, false, false, d_newstate_flags);
					if (bufaddr_0 == CACHE_FULL) {
						// Construct the vector again, and store it directly in the global hash table.
						mode = TO_GLOBAL;
						continue;
					}
					else if (bufaddr_0 == HASHTABLE_FULL) {
						// Hash table is considered full. Report this back.
						return HASH_TABLE_FULL;
					}
					if (bufaddr_0 != EMPTY_HASH_POINTER) {
						get_vectortree_node(&part1, &part_cachepointers, node_index, 0);
						part2 = part1;
						if (mode == TO_CACHE) {
							set_left_cache_pointer(&part_cachepointers, bufaddr_0);
							reset_left_in_vectortree_node(&part2);
							mark_root(&part2);
							mark_cached_node_new_nonleaf(&part_cachepointers);
							bufaddr_0 = STOREINCACHE(part2, part_cachepointers, false);
							if (bufaddr_0 == CACHE_FULL) {
								// Construct the vector again, and store it directly in the global hash table.
								mode = TO_GLOBAL;
								continue;
							}
						}
						else {
							set_left_in_vectortree_node(&part2, bufaddr_0);
							bufaddr_0 = FINDORPUT_SINGLE(d_q, d_q_i, d_dummy, part2, d_newstate_flags, EMPTY_CACHE_POINTER, true, (ITERATIONS == d_kernel_iters-1));
							if (bufaddr_0 == HASHTABLE_FULL) {
								// Hash table is considered full. Report this back.
								return HASH_TABLE_FULL;
							}
						}
					}
					mode = (mode == TO_CACHE ? STORED : GLOBAL_STORED);
				}
			}
			}
			return STORED;
		case 2:
			{
			// Allocate register memory to process transition(s).
			indextype bufaddr_0;
			elem_chartype buf8_0, buf8_1;
			// Allocate register memory for dynamic array indexing.
			array_indextype idx_0;
			
			// p2 --{ Slot[myplace] = 1 }--> p3
			
			mode = STORED;
			// Reset storage of array indices.
			idx_0 = EMPTY_INDEX;
			// Fetch values of unguarded variables.
			part1 = get_vectorpart(node_index, 0);
			part2 = part1;
			get_globalObject_P_3_myplace(&buf8_0, part1, part2);
			
			// Statement computation.
			if (globalObject_Slot(node_index, &idx_0, &buf8_1, buf8_0) == 1) {
				target = 3;
				mode = (mode == STORED ? TO_CACHE : TO_GLOBAL);
				while (mode != STORED && mode != GLOBAL_STORED) {
					// Store new state vector in the cache or the global hash table.
					get_vectortree_node(&part1, &part_cachepointers, node_index, 1);
					// Store new values.
					part2 = part1;
					set_left_globalObject_P_3(&part2, (statetype) target);
					bufaddr_0 = STORENODE(mode, d_q, d_q_i, d_dummy, part1, &part2, &part_cachepointers, false, false, d_newstate_flags);
					if (bufaddr_0 == CACHE_FULL) {
						// Construct the vector again, and store it directly in the global hash table.
						mode = TO_GLOBAL;
						continue;
					}
					else if (bufaddr_0 == HASHTABLE_FULL) {
						// Hash table is considered full. Report this back.
						return HASH_TABLE_FULL;
					}
					if (bufaddr_0 != EMPTY_HASH_POINTER) {
						get_vectortree_node(&part1, &part_cachepointers, node_index, 0);
						part2 = part1;
						if (mode == TO_CACHE) {
							set_left_cache_pointer(&part_cachepointers, bufaddr_0);
							reset_left_in_vectortree_node(&part2);
							mark_root(&part2);
							mark_cached_node_new_nonleaf(&part_cachepointers);
							bufaddr_0 = STOREINCACHE(part2, part_cachepointers, false);
							if (bufaddr_0 == CACHE_FULL) {
								// Construct the vector again, and store it directly in the global hash table.
								mode = TO_GLOBAL;
								continue;
							}
						}
						else {
							set_left_in_vectortree_node(&part2, bufaddr_0);
							bufaddr_0 = FINDORPUT_SINGLE(d_q, d_q_i, d_dummy, part2, d_newstate_flags, EMPTY_CACHE_POINTER, true, (ITERATIONS == d_kernel_iters-1));
							if (bufaddr_0 == HASHTABLE_FULL) {
								// Hash table is considered full. Report this back.
								return HASH_TABLE_FULL;
							}
						}
					}
					mode = (mode == TO_CACHE ? STORED : GLOBAL_STORED);
				}
			}
			}
			return STORED;
		case 3:
			{
			// Allocate register memory to process transition(s).
			indextype bufaddr_0, bufaddr_1;
			elem_chartype buf8_0, buf8_1;
			// Allocate register memory for dynamic array indexing.
			array_indextype idx_0;
			
			// p3 --{ [ Slot[(myplace + 5 - 1) % 5] := 0 ] }--> CS
			
			mode = STORED;
			// Reset storage of array indices.
			idx_0 = EMPTY_INDEX;
			// Fetch values of unguarded variables.
			part1 = get_vectorpart(node_index, 0);
			part2 = part1;
			get_globalObject_P_3_myplace(&buf8_0, part1, part2);
			
			// Statement computation.
			target = 4;
			A_STR_1(&idx_0, &buf8_1, (array_indextype) (buf8_0 + 5 - 1) % 5, (elem_chartype) 0);
			mode = (mode == STORED ? TO_CACHE : TO_GLOBAL);
			while (mode != STORED && mode != GLOBAL_STORED) {
				// Store new state vector in the cache or the global hash table.
				get_vectortree_node(&part1, &part_cachepointers, node_index, 1);
				// Store new values.
				part2 = part1;
				// Write array buffer content.
				if (0 >= 0 && 0 <= 1) {
					if (idx_0 != EMPTY_INDEX) {
						if (array_element_is_in_vectorpart_globalObject_Slot(idx_0, 0)) {
							if (is_left_vectorpart_for_array_element_globalObject_Slot(idx_0, 0)) {
								set_left_globalObject_Slot(&part2, idx_0, buf8_1, 0);
							}
						}
					}
				}
				set_left_globalObject_P_3(&part2, (statetype) target);
				bufaddr_0 = STORENODE(mode, d_q, d_q_i, d_dummy, part1, &part2, &part_cachepointers, false, false, d_newstate_flags);
				if (bufaddr_0 == CACHE_FULL) {
					// Construct the vector again, and store it directly in the global hash table.
					mode = TO_GLOBAL;
					continue;
				}
				else if (bufaddr_0 == HASHTABLE_FULL) {
					// Hash table is considered full. Report this back.
					return HASH_TABLE_FULL;
				}
				get_vectortree_node(&part1, &part_cachepointers, node_index, 2);
				// Store new values.
				part2 = part1;
				// Write array buffer content.
				if (1 >= 0 && 1 <= 1) {
					if (idx_0 != EMPTY_INDEX) {
						if (array_element_is_in_vectorpart_globalObject_Slot(idx_0, 1)) {
							if (is_left_vectorpart_for_array_element_globalObject_Slot(idx_0, 1)) {
								set_left_globalObject_Slot(&part2, idx_0, buf8_1, 1);
							}
						}
					}
				}
				bufaddr_1 = STORENODE(mode, d_q, d_q_i, d_dummy, part1, &part2, &part_cachepointers, false, false, d_newstate_flags);
				if (bufaddr_1 == CACHE_FULL) {
					// Construct the vector again, and store it directly in the global hash table.
					mode = TO_GLOBAL;
					continue;
				}
				else if (bufaddr_1 == HASHTABLE_FULL) {
					// Hash table is considered full. Report this back.
					return HASH_TABLE_FULL;
				}
				if (bufaddr_0 != EMPTY_HASH_POINTER || bufaddr_1 != EMPTY_HASH_POINTER) {
					get_vectortree_node(&part1, &part_cachepointers, node_index, 0);
					part2 = part1;
					if (bufaddr_0 != EMPTY_HASH_POINTER) {
						if (mode == TO_CACHE) {
							set_left_cache_pointer(&part_cachepointers, bufaddr_0);
							reset_left_in_vectortree_node(&part2);
						}
						else {
							set_left_in_vectortree_node(&part2, bufaddr_0);
						}
					}
					if (bufaddr_1 != EMPTY_HASH_POINTER) {
						if (mode == TO_CACHE) {
							set_right_cache_pointer(&part_cachepointers, bufaddr_1);
							reset_right_in_vectortree_node(&part2);
						}
						else {
							set_right_in_vectortree_node(&part2, bufaddr_1);
						}
					}
					if (mode == TO_CACHE) {
						// This part has been altered. Store it and remember address of new part.
						mark_root(&part2);
						mark_cached_node_new_nonleaf(&part_cachepointers);
						bufaddr_0 = STOREINCACHE(part2, part_cachepointers, false);
						if (bufaddr_0 == CACHE_FULL) {
							// Construct the vector again, and store it directly in the global hash table.
							mode = TO_GLOBAL;
							continue;
						}
					}
					else {
						bufaddr_0 = FINDORPUT_SINGLE(d_q, d_q_i, d_dummy, part2, d_newstate_flags, EMPTY_CACHE_POINTER, true, (ITERATIONS == d_kernel_iters-1));
						if (bufaddr_0 == HASHTABLE_FULL) {
							// Hash table is considered full. Report this back.
							return HASH_TABLE_FULL;
						}
					}
				}
				mode = (mode == TO_CACHE ? STORED : GLOBAL_STORED);
			}
			}
			return STORED;
		case 4:
			{
			// Allocate register memory to process transition(s).
			indextype bufaddr_0, bufaddr_1;
			elem_chartype buf8_0, buf8_1;
			// Allocate register memory for dynamic array indexing.
			array_indextype idx_0;
			
			// CS --{ [ Slot[(myplace + 1) % 5] := 1 ] }--> NCS
			
			mode = STORED;
			// Reset storage of array indices.
			idx_0 = EMPTY_INDEX;
			// Fetch values of unguarded variables.
			part1 = get_vectorpart(node_index, 0);
			part2 = part1;
			get_globalObject_P_3_myplace(&buf8_0, part1, part2);
			
			// Statement computation.
			target = 0;
			A_STR_1(&idx_0, &buf8_1, (array_indextype) (buf8_0 + 1) % 5, (elem_chartype) 1);
			mode = (mode == STORED ? TO_CACHE : TO_GLOBAL);
			while (mode != STORED && mode != GLOBAL_STORED) {
				// Store new state vector in the cache or the global hash table.
				get_vectortree_node(&part1, &part_cachepointers, node_index, 1);
				// Store new values.
				part2 = part1;
				// Write array buffer content.
				if (0 >= 0 && 0 <= 1) {
					if (idx_0 != EMPTY_INDEX) {
						if (array_element_is_in_vectorpart_globalObject_Slot(idx_0, 0)) {
							if (is_left_vectorpart_for_array_element_globalObject_Slot(idx_0, 0)) {
								set_left_globalObject_Slot(&part2, idx_0, buf8_1, 0);
							}
						}
					}
				}
				set_left_globalObject_P_3(&part2, (statetype) target);
				bufaddr_0 = STORENODE(mode, d_q, d_q_i, d_dummy, part1, &part2, &part_cachepointers, false, false, d_newstate_flags);
				if (bufaddr_0 == CACHE_FULL) {
					// Construct the vector again, and store it directly in the global hash table.
					mode = TO_GLOBAL;
					continue;
				}
				else if (bufaddr_0 == HASHTABLE_FULL) {
					// Hash table is considered full. Report this back.
					return HASH_TABLE_FULL;
				}
				get_vectortree_node(&part1, &part_cachepointers, node_index, 2);
				// Store new values.
				part2 = part1;
				// Write array buffer content.
				if (1 >= 0 && 1 <= 1) {
					if (idx_0 != EMPTY_INDEX) {
						if (array_element_is_in_vectorpart_globalObject_Slot(idx_0, 1)) {
							if (is_left_vectorpart_for_array_element_globalObject_Slot(idx_0, 1)) {
								set_left_globalObject_Slot(&part2, idx_0, buf8_1, 1);
							}
						}
					}
				}
				bufaddr_1 = STORENODE(mode, d_q, d_q_i, d_dummy, part1, &part2, &part_cachepointers, false, false, d_newstate_flags);
				if (bufaddr_1 == CACHE_FULL) {
					// Construct the vector again, and store it directly in the global hash table.
					mode = TO_GLOBAL;
					continue;
				}
				else if (bufaddr_1 == HASHTABLE_FULL) {
					// Hash table is considered full. Report this back.
					return HASH_TABLE_FULL;
				}
				if (bufaddr_0 != EMPTY_HASH_POINTER || bufaddr_1 != EMPTY_HASH_POINTER) {
					get_vectortree_node(&part1, &part_cachepointers, node_index, 0);
					part2 = part1;
					if (bufaddr_0 != EMPTY_HASH_POINTER) {
						if (mode == TO_CACHE) {
							set_left_cache_pointer(&part_cachepointers, bufaddr_0);
							reset_left_in_vectortree_node(&part2);
						}
						else {
							set_left_in_vectortree_node(&part2, bufaddr_0);
						}
					}
					if (bufaddr_1 != EMPTY_HASH_POINTER) {
						if (mode == TO_CACHE) {
							set_right_cache_pointer(&part_cachepointers, bufaddr_1);
							reset_right_in_vectortree_node(&part2);
						}
						else {
							set_right_in_vectortree_node(&part2, bufaddr_1);
						}
					}
					if (mode == TO_CACHE) {
						// This part has been altered. Store it and remember address of new part.
						mark_root(&part2);
						mark_cached_node_new_nonleaf(&part_cachepointers);
						bufaddr_0 = STOREINCACHE(part2, part_cachepointers, false);
						if (bufaddr_0 == CACHE_FULL) {
							// Construct the vector again, and store it directly in the global hash table.
							mode = TO_GLOBAL;
							continue;
						}
					}
					else {
						bufaddr_0 = FINDORPUT_SINGLE(d_q, d_q_i, d_dummy, part2, d_newstate_flags, EMPTY_CACHE_POINTER, true, (ITERATIONS == d_kernel_iters-1));
						if (bufaddr_0 == HASHTABLE_FULL) {
							// Hash table is considered full. Report this back.
							return HASH_TABLE_FULL;
						}
					}
				}
				mode = (mode == TO_CACHE ? STORED : GLOBAL_STORED);
			}
			}
			return STORED;
		default:
			return STORED;
	}
}

inline __device__ Storage_mode explore_globalObject_P_1(shared_indextype node_index, compressed_nodetype *d_q, nodetype *d_q_i, bool *d_dummy, volatile uint8_t *d_newstate_flags) {
	// Fetch the current state of the state machine.
	statetype current;
	get_current_state(&current, node_index, 4);
	statetype target = NO_STATE;
	nodetype part1, part2;
	// Storage mode to determine where to store the node(s).
	Storage_mode mode;
	shared_inttype part_cachepointers;
	switch (current) {
		case 0:
			{
			// Allocate register memory to process transition(s).
			indextype bufaddr_0, bufaddr_1;
			elem_chartype buf8_0, buf8_1;
			
			// NCS --{ [ myplace := next; next := next + 1 ] }--> p1
			
			mode = STORED;
			// Fetch values of unguarded variables.
			part1 = get_vectorpart(node_index, 0);
			part2 = part1;
			get_globalObject_next(&buf8_1, part1, part2);
			
			// Statement computation.
			target = 1;
			buf8_0 = (elem_chartype) (buf8_1);
			buf8_1 = (elem_chartype) (buf8_1 + 1);
			mode = (mode == STORED ? TO_CACHE : TO_GLOBAL);
			while (mode != STORED && mode != GLOBAL_STORED) {
				// Store new state vector in the cache or the global hash table.
				get_vectortree_node(&part1, &part_cachepointers, node_index, 1);
				// Store new values.
				part2 = part1;
				set_left_globalObject_P_1(&part2, (statetype) target);
				set_left_globalObject_next(&part2, buf8_1);
				bufaddr_0 = STORENODE(mode, d_q, d_q_i, d_dummy, part1, &part2, &part_cachepointers, false, false, d_newstate_flags);
				if (bufaddr_0 == CACHE_FULL) {
					// Construct the vector again, and store it directly in the global hash table.
					mode = TO_GLOBAL;
					continue;
				}
				else if (bufaddr_0 == HASHTABLE_FULL) {
					// Hash table is considered full. Report this back.
					return HASH_TABLE_FULL;
				}
				get_vectortree_node(&part1, &part_cachepointers, node_index, 2);
				// Store new values.
				part2 = part1;
				set_left_globalObject_P_1_myplace(&part2, buf8_0);
				bufaddr_1 = STORENODE(mode, d_q, d_q_i, d_dummy, part1, &part2, &part_cachepointers, false, false, d_newstate_flags);
				if (bufaddr_1 == CACHE_FULL) {
					// Construct the vector again, and store it directly in the global hash table.
					mode = TO_GLOBAL;
					continue;
				}
				else if (bufaddr_1 == HASHTABLE_FULL) {
					// Hash table is considered full. Report this back.
					return HASH_TABLE_FULL;
				}
				if (bufaddr_0 != EMPTY_HASH_POINTER || bufaddr_1 != EMPTY_HASH_POINTER) {
					get_vectortree_node(&part1, &part_cachepointers, node_index, 0);
					part2 = part1;
					if (bufaddr_0 != EMPTY_HASH_POINTER) {
						if (mode == TO_CACHE) {
							set_left_cache_pointer(&part_cachepointers, bufaddr_0);
							reset_left_in_vectortree_node(&part2);
						}
						else {
							set_left_in_vectortree_node(&part2, bufaddr_0);
						}
					}
					if (bufaddr_1 != EMPTY_HASH_POINTER) {
						if (mode == TO_CACHE) {
							set_right_cache_pointer(&part_cachepointers, bufaddr_1);
							reset_right_in_vectortree_node(&part2);
						}
						else {
							set_right_in_vectortree_node(&part2, bufaddr_1);
						}
					}
					if (mode == TO_CACHE) {
						// This part has been altered. Store it and remember address of new part.
						mark_root(&part2);
						mark_cached_node_new_nonleaf(&part_cachepointers);
						bufaddr_0 = STOREINCACHE(part2, part_cachepointers, false);
						if (bufaddr_0 == CACHE_FULL) {
							// Construct the vector again, and store it directly in the global hash table.
							mode = TO_GLOBAL;
							continue;
						}
					}
					else {
						bufaddr_0 = FINDORPUT_SINGLE(d_q, d_q_i, d_dummy, part2, d_newstate_flags, EMPTY_CACHE_POINTER, true, (ITERATIONS == d_kernel_iters-1));
						if (bufaddr_0 == HASHTABLE_FULL) {
							// Hash table is considered full. Report this back.
							return HASH_TABLE_FULL;
						}
					}
				}
				mode = (mode == TO_CACHE ? STORED : GLOBAL_STORED);
			}
			}
			return STORED;
		case 1:
			{
			// Allocate register memory to process transition(s).
			indextype bufaddr_0, bufaddr_1;
			elem_chartype buf8_0, buf8_1;
			
			// p1 --{ [ myplace = 5 - 1; next := next - 5 ] }--> p2
			
			mode = STORED;
			// Fetch values of unguarded variables.
			part1 = get_vectorpart(node_index, 1);
			part2 = part1;
			get_globalObject_P_1_myplace(&buf8_0, part1, part2);
			
			// Statement computation.
			if (buf8_0 == 5 - 1) {
				// Fetch values of guarded variables.
				part1 = get_vectorpart(node_index, 0);
				part2 = part1;
				get_globalObject_next(&buf8_1, part1, part2);
				
				target = 2;
				buf8_1 = (elem_chartype) (buf8_1 - 5);
				mode = (mode == STORED ? TO_CACHE : TO_GLOBAL);
				while (mode != STORED && mode != GLOBAL_STORED) {
					// Store new state vector in the cache or the global hash table.
					get_vectortree_node(&part1, &part_cachepointers, node_index, 1);
					// Store new values.
					part2 = part1;
					set_left_globalObject_P_1(&part2, (statetype) target);
					set_left_globalObject_next(&part2, buf8_1);
					bufaddr_0 = STORENODE(mode, d_q, d_q_i, d_dummy, part1, &part2, &part_cachepointers, false, false, d_newstate_flags);
					if (bufaddr_0 == CACHE_FULL) {
						// Construct the vector again, and store it directly in the global hash table.
						mode = TO_GLOBAL;
						continue;
					}
					else if (bufaddr_0 == HASHTABLE_FULL) {
						// Hash table is considered full. Report this back.
						return HASH_TABLE_FULL;
					}
					if (bufaddr_0 != EMPTY_HASH_POINTER) {
						get_vectortree_node(&part1, &part_cachepointers, node_index, 0);
						part2 = part1;
						if (mode == TO_CACHE) {
							set_left_cache_pointer(&part_cachepointers, bufaddr_0);
							reset_left_in_vectortree_node(&part2);
							mark_root(&part2);
							mark_cached_node_new_nonleaf(&part_cachepointers);
							bufaddr_0 = STOREINCACHE(part2, part_cachepointers, false);
							if (bufaddr_0 == CACHE_FULL) {
								// Construct the vector again, and store it directly in the global hash table.
								mode = TO_GLOBAL;
								continue;
							}
						}
						else {
							set_left_in_vectortree_node(&part2, bufaddr_0);
							bufaddr_0 = FINDORPUT_SINGLE(d_q, d_q_i, d_dummy, part2, d_newstate_flags, EMPTY_CACHE_POINTER, true, (ITERATIONS == d_kernel_iters-1));
							if (bufaddr_0 == HASHTABLE_FULL) {
								// Hash table is considered full. Report this back.
								return HASH_TABLE_FULL;
							}
						}
					}
					mode = (mode == TO_CACHE ? STORED : GLOBAL_STORED);
				}
			}
			
			// p1 --{ [ myplace <> 5 - 1; myplace := myplace % 5 ] }--> p2
			
			mode = STORED;
			// Fetch values of unguarded variables.
			part1 = get_vectorpart(node_index, 1);
			part2 = part1;
			get_globalObject_P_1_myplace(&buf8_0, part1, part2);
			
			// Statement computation.
			if (buf8_0 != 5 - 1) {
				target = 2;
				buf8_0 = (elem_chartype) (buf8_0 % 5);
				mode = (mode == STORED ? TO_CACHE : TO_GLOBAL);
				while (mode != STORED && mode != GLOBAL_STORED) {
					// Store new state vector in the cache or the global hash table.
					get_vectortree_node(&part1, &part_cachepointers, node_index, 1);
					// Store new values.
					part2 = part1;
					set_left_globalObject_P_1(&part2, (statetype) target);
					bufaddr_0 = STORENODE(mode, d_q, d_q_i, d_dummy, part1, &part2, &part_cachepointers, false, false, d_newstate_flags);
					if (bufaddr_0 == CACHE_FULL) {
						// Construct the vector again, and store it directly in the global hash table.
						mode = TO_GLOBAL;
						continue;
					}
					else if (bufaddr_0 == HASHTABLE_FULL) {
						// Hash table is considered full. Report this back.
						return HASH_TABLE_FULL;
					}
					get_vectortree_node(&part1, &part_cachepointers, node_index, 2);
					// Store new values.
					part2 = part1;
					set_left_globalObject_P_1_myplace(&part2, buf8_0);
					bufaddr_1 = STORENODE(mode, d_q, d_q_i, d_dummy, part1, &part2, &part_cachepointers, false, false, d_newstate_flags);
					if (bufaddr_1 == CACHE_FULL) {
						// Construct the vector again, and store it directly in the global hash table.
						mode = TO_GLOBAL;
						continue;
					}
					else if (bufaddr_1 == HASHTABLE_FULL) {
						// Hash table is considered full. Report this back.
						return HASH_TABLE_FULL;
					}
					if (bufaddr_0 != EMPTY_HASH_POINTER || bufaddr_1 != EMPTY_HASH_POINTER) {
						get_vectortree_node(&part1, &part_cachepointers, node_index, 0);
						part2 = part1;
						if (bufaddr_0 != EMPTY_HASH_POINTER) {
							if (mode == TO_CACHE) {
								set_left_cache_pointer(&part_cachepointers, bufaddr_0);
								reset_left_in_vectortree_node(&part2);
							}
							else {
								set_left_in_vectortree_node(&part2, bufaddr_0);
							}
						}
						if (bufaddr_1 != EMPTY_HASH_POINTER) {
							if (mode == TO_CACHE) {
								set_right_cache_pointer(&part_cachepointers, bufaddr_1);
								reset_right_in_vectortree_node(&part2);
							}
							else {
								set_right_in_vectortree_node(&part2, bufaddr_1);
							}
						}
						if (mode == TO_CACHE) {
							// This part has been altered. Store it and remember address of new part.
							mark_root(&part2);
							mark_cached_node_new_nonleaf(&part_cachepointers);
							bufaddr_0 = STOREINCACHE(part2, part_cachepointers, false);
							if (bufaddr_0 == CACHE_FULL) {
								// Construct the vector again, and store it directly in the global hash table.
								mode = TO_GLOBAL;
								continue;
							}
						}
						else {
							bufaddr_0 = FINDORPUT_SINGLE(d_q, d_q_i, d_dummy, part2, d_newstate_flags, EMPTY_CACHE_POINTER, true, (ITERATIONS == d_kernel_iters-1));
							if (bufaddr_0 == HASHTABLE_FULL) {
								// Hash table is considered full. Report this back.
								return HASH_TABLE_FULL;
							}
						}
					}
					mode = (mode == TO_CACHE ? STORED : GLOBAL_STORED);
				}
			}
			}
			return STORED;
		case 2:
			{
			// Allocate register memory to process transition(s).
			indextype bufaddr_0;
			elem_chartype buf8_0, buf8_1;
			// Allocate register memory for dynamic array indexing.
			array_indextype idx_0;
			
			// p2 --{ Slot[myplace] = 1 }--> p3
			
			mode = STORED;
			// Reset storage of array indices.
			idx_0 = EMPTY_INDEX;
			// Fetch values of unguarded variables.
			part1 = get_vectorpart(node_index, 1);
			part2 = part1;
			get_globalObject_P_1_myplace(&buf8_1, part1, part2);
			
			// Statement computation.
			if (globalObject_Slot(node_index, &idx_0, &buf8_0, buf8_1) == 1) {
				target = 3;
				mode = (mode == STORED ? TO_CACHE : TO_GLOBAL);
				while (mode != STORED && mode != GLOBAL_STORED) {
					// Store new state vector in the cache or the global hash table.
					get_vectortree_node(&part1, &part_cachepointers, node_index, 1);
					// Store new values.
					part2 = part1;
					set_left_globalObject_P_1(&part2, (statetype) target);
					bufaddr_0 = STORENODE(mode, d_q, d_q_i, d_dummy, part1, &part2, &part_cachepointers, false, false, d_newstate_flags);
					if (bufaddr_0 == CACHE_FULL) {
						// Construct the vector again, and store it directly in the global hash table.
						mode = TO_GLOBAL;
						continue;
					}
					else if (bufaddr_0 == HASHTABLE_FULL) {
						// Hash table is considered full. Report this back.
						return HASH_TABLE_FULL;
					}
					if (bufaddr_0 != EMPTY_HASH_POINTER) {
						get_vectortree_node(&part1, &part_cachepointers, node_index, 0);
						part2 = part1;
						if (mode == TO_CACHE) {
							set_left_cache_pointer(&part_cachepointers, bufaddr_0);
							reset_left_in_vectortree_node(&part2);
							mark_root(&part2);
							mark_cached_node_new_nonleaf(&part_cachepointers);
							bufaddr_0 = STOREINCACHE(part2, part_cachepointers, false);
							if (bufaddr_0 == CACHE_FULL) {
								// Construct the vector again, and store it directly in the global hash table.
								mode = TO_GLOBAL;
								continue;
							}
						}
						else {
							set_left_in_vectortree_node(&part2, bufaddr_0);
							bufaddr_0 = FINDORPUT_SINGLE(d_q, d_q_i, d_dummy, part2, d_newstate_flags, EMPTY_CACHE_POINTER, true, (ITERATIONS == d_kernel_iters-1));
							if (bufaddr_0 == HASHTABLE_FULL) {
								// Hash table is considered full. Report this back.
								return HASH_TABLE_FULL;
							}
						}
					}
					mode = (mode == TO_CACHE ? STORED : GLOBAL_STORED);
				}
			}
			}
			return STORED;
		case 3:
			{
			// Allocate register memory to process transition(s).
			indextype bufaddr_0, bufaddr_1;
			elem_chartype buf8_0, buf8_1;
			// Allocate register memory for dynamic array indexing.
			array_indextype idx_0;
			
			// p3 --{ [ Slot[(myplace + 5 - 1) % 5] := 0 ] }--> CS
			
			mode = STORED;
			// Reset storage of array indices.
			idx_0 = EMPTY_INDEX;
			// Fetch values of unguarded variables.
			part1 = get_vectorpart(node_index, 1);
			part2 = part1;
			get_globalObject_P_1_myplace(&buf8_0, part1, part2);
			
			// Statement computation.
			target = 4;
			A_STR_1(&idx_0, &buf8_1, (array_indextype) (buf8_0 + 5 - 1) % 5, (elem_chartype) 0);
			mode = (mode == STORED ? TO_CACHE : TO_GLOBAL);
			while (mode != STORED && mode != GLOBAL_STORED) {
				// Store new state vector in the cache or the global hash table.
				get_vectortree_node(&part1, &part_cachepointers, node_index, 1);
				// Store new values.
				part2 = part1;
				// Write array buffer content.
				if (0 >= 0 && 0 <= 1) {
					if (idx_0 != EMPTY_INDEX) {
						if (array_element_is_in_vectorpart_globalObject_Slot(idx_0, 0)) {
							if (is_left_vectorpart_for_array_element_globalObject_Slot(idx_0, 0)) {
								set_left_globalObject_Slot(&part2, idx_0, buf8_1, 0);
							}
						}
					}
				}
				set_left_globalObject_P_1(&part2, (statetype) target);
				bufaddr_0 = STORENODE(mode, d_q, d_q_i, d_dummy, part1, &part2, &part_cachepointers, false, false, d_newstate_flags);
				if (bufaddr_0 == CACHE_FULL) {
					// Construct the vector again, and store it directly in the global hash table.
					mode = TO_GLOBAL;
					continue;
				}
				else if (bufaddr_0 == HASHTABLE_FULL) {
					// Hash table is considered full. Report this back.
					return HASH_TABLE_FULL;
				}
				get_vectortree_node(&part1, &part_cachepointers, node_index, 2);
				// Store new values.
				part2 = part1;
				// Write array buffer content.
				if (1 >= 0 && 1 <= 1) {
					if (idx_0 != EMPTY_INDEX) {
						if (array_element_is_in_vectorpart_globalObject_Slot(idx_0, 1)) {
							if (is_left_vectorpart_for_array_element_globalObject_Slot(idx_0, 1)) {
								set_left_globalObject_Slot(&part2, idx_0, buf8_1, 1);
							}
						}
					}
				}
				bufaddr_1 = STORENODE(mode, d_q, d_q_i, d_dummy, part1, &part2, &part_cachepointers, false, false, d_newstate_flags);
				if (bufaddr_1 == CACHE_FULL) {
					// Construct the vector again, and store it directly in the global hash table.
					mode = TO_GLOBAL;
					continue;
				}
				else if (bufaddr_1 == HASHTABLE_FULL) {
					// Hash table is considered full. Report this back.
					return HASH_TABLE_FULL;
				}
				if (bufaddr_0 != EMPTY_HASH_POINTER || bufaddr_1 != EMPTY_HASH_POINTER) {
					get_vectortree_node(&part1, &part_cachepointers, node_index, 0);
					part2 = part1;
					if (bufaddr_0 != EMPTY_HASH_POINTER) {
						if (mode == TO_CACHE) {
							set_left_cache_pointer(&part_cachepointers, bufaddr_0);
							reset_left_in_vectortree_node(&part2);
						}
						else {
							set_left_in_vectortree_node(&part2, bufaddr_0);
						}
					}
					if (bufaddr_1 != EMPTY_HASH_POINTER) {
						if (mode == TO_CACHE) {
							set_right_cache_pointer(&part_cachepointers, bufaddr_1);
							reset_right_in_vectortree_node(&part2);
						}
						else {
							set_right_in_vectortree_node(&part2, bufaddr_1);
						}
					}
					if (mode == TO_CACHE) {
						// This part has been altered. Store it and remember address of new part.
						mark_root(&part2);
						mark_cached_node_new_nonleaf(&part_cachepointers);
						bufaddr_0 = STOREINCACHE(part2, part_cachepointers, false);
						if (bufaddr_0 == CACHE_FULL) {
							// Construct the vector again, and store it directly in the global hash table.
							mode = TO_GLOBAL;
							continue;
						}
					}
					else {
						bufaddr_0 = FINDORPUT_SINGLE(d_q, d_q_i, d_dummy, part2, d_newstate_flags, EMPTY_CACHE_POINTER, true, (ITERATIONS == d_kernel_iters-1));
						if (bufaddr_0 == HASHTABLE_FULL) {
							// Hash table is considered full. Report this back.
							return HASH_TABLE_FULL;
						}
					}
				}
				mode = (mode == TO_CACHE ? STORED : GLOBAL_STORED);
			}
			}
			return STORED;
		case 4:
			{
			// Allocate register memory to process transition(s).
			indextype bufaddr_0, bufaddr_1;
			elem_chartype buf8_0, buf8_1;
			// Allocate register memory for dynamic array indexing.
			array_indextype idx_0;
			
			// CS --{ [ Slot[(myplace + 1) % 5] := 1 ] }--> NCS
			
			mode = STORED;
			// Reset storage of array indices.
			idx_0 = EMPTY_INDEX;
			// Fetch values of unguarded variables.
			part1 = get_vectorpart(node_index, 1);
			part2 = part1;
			get_globalObject_P_1_myplace(&buf8_1, part1, part2);
			
			// Statement computation.
			target = 0;
			A_STR_1(&idx_0, &buf8_0, (array_indextype) (buf8_1 + 1) % 5, (elem_chartype) 1);
			mode = (mode == STORED ? TO_CACHE : TO_GLOBAL);
			while (mode != STORED && mode != GLOBAL_STORED) {
				// Store new state vector in the cache or the global hash table.
				get_vectortree_node(&part1, &part_cachepointers, node_index, 1);
				// Store new values.
				part2 = part1;
				// Write array buffer content.
				if (0 >= 0 && 0 <= 1) {
					if (idx_0 != EMPTY_INDEX) {
						if (array_element_is_in_vectorpart_globalObject_Slot(idx_0, 0)) {
							if (is_left_vectorpart_for_array_element_globalObject_Slot(idx_0, 0)) {
								set_left_globalObject_Slot(&part2, idx_0, buf8_0, 0);
							}
						}
					}
				}
				set_left_globalObject_P_1(&part2, (statetype) target);
				bufaddr_0 = STORENODE(mode, d_q, d_q_i, d_dummy, part1, &part2, &part_cachepointers, false, false, d_newstate_flags);
				if (bufaddr_0 == CACHE_FULL) {
					// Construct the vector again, and store it directly in the global hash table.
					mode = TO_GLOBAL;
					continue;
				}
				else if (bufaddr_0 == HASHTABLE_FULL) {
					// Hash table is considered full. Report this back.
					return HASH_TABLE_FULL;
				}
				get_vectortree_node(&part1, &part_cachepointers, node_index, 2);
				// Store new values.
				part2 = part1;
				// Write array buffer content.
				if (1 >= 0 && 1 <= 1) {
					if (idx_0 != EMPTY_INDEX) {
						if (array_element_is_in_vectorpart_globalObject_Slot(idx_0, 1)) {
							if (is_left_vectorpart_for_array_element_globalObject_Slot(idx_0, 1)) {
								set_left_globalObject_Slot(&part2, idx_0, buf8_0, 1);
							}
						}
					}
				}
				bufaddr_1 = STORENODE(mode, d_q, d_q_i, d_dummy, part1, &part2, &part_cachepointers, false, false, d_newstate_flags);
				if (bufaddr_1 == CACHE_FULL) {
					// Construct the vector again, and store it directly in the global hash table.
					mode = TO_GLOBAL;
					continue;
				}
				else if (bufaddr_1 == HASHTABLE_FULL) {
					// Hash table is considered full. Report this back.
					return HASH_TABLE_FULL;
				}
				if (bufaddr_0 != EMPTY_HASH_POINTER || bufaddr_1 != EMPTY_HASH_POINTER) {
					get_vectortree_node(&part1, &part_cachepointers, node_index, 0);
					part2 = part1;
					if (bufaddr_0 != EMPTY_HASH_POINTER) {
						if (mode == TO_CACHE) {
							set_left_cache_pointer(&part_cachepointers, bufaddr_0);
							reset_left_in_vectortree_node(&part2);
						}
						else {
							set_left_in_vectortree_node(&part2, bufaddr_0);
						}
					}
					if (bufaddr_1 != EMPTY_HASH_POINTER) {
						if (mode == TO_CACHE) {
							set_right_cache_pointer(&part_cachepointers, bufaddr_1);
							reset_right_in_vectortree_node(&part2);
						}
						else {
							set_right_in_vectortree_node(&part2, bufaddr_1);
						}
					}
					if (mode == TO_CACHE) {
						// This part has been altered. Store it and remember address of new part.
						mark_root(&part2);
						mark_cached_node_new_nonleaf(&part_cachepointers);
						bufaddr_0 = STOREINCACHE(part2, part_cachepointers, false);
						if (bufaddr_0 == CACHE_FULL) {
							// Construct the vector again, and store it directly in the global hash table.
							mode = TO_GLOBAL;
							continue;
						}
					}
					else {
						bufaddr_0 = FINDORPUT_SINGLE(d_q, d_q_i, d_dummy, part2, d_newstate_flags, EMPTY_CACHE_POINTER, true, (ITERATIONS == d_kernel_iters-1));
						if (bufaddr_0 == HASHTABLE_FULL) {
							// Hash table is considered full. Report this back.
							return HASH_TABLE_FULL;
						}
					}
				}
				mode = (mode == TO_CACHE ? STORED : GLOBAL_STORED);
			}
			}
			return STORED;
		default:
			return STORED;
	}
}

// Successor construction function for a particular state machine. Given a state vector, construct its successor state vectors w.r.t. the state machine, and store them in cache.
// Vgtid is the identity of the thread calling the function (id of thread relevant for successor generation).
inline __device__ Storage_mode get_successors_of_sm(shared_indextype node_index, uint8_t vgtid, compressed_nodetype *d_q, nodetype *d_q_i, bool *d_dummy, volatile uint8_t *d_newstate_flags) {
	// explore the outgoing transitions of the current state of the state machine assigned to vgtid.
	switch (vgtid) {
		case 0:
			return explore_globalObject_P_0(node_index, d_q, d_q_i, d_dummy, d_newstate_flags);
		case 1:
			return explore_globalObject_P_4(node_index, d_q, d_q_i, d_dummy, d_newstate_flags);
		case 2:
			return explore_globalObject_P_2(node_index, d_q, d_q_i, d_dummy, d_newstate_flags);
		case 3:
			return explore_globalObject_P_3(node_index, d_q, d_q_i, d_dummy, d_newstate_flags);
		case 4:
			return explore_globalObject_P_1(node_index, d_q, d_q_i, d_dummy, d_newstate_flags);
		default:
			return STORED;
	}
}

// Kernel function to start parallel successor generation.
// Precondition: a tile of vectortree pointers to roots of cache-preloaded vectortrees is stored in the shared memory.
inline __device__ Storage_mode GENERATE_SUCCESSORS(compressed_nodetype *d_q, nodetype *d_q_i, bool *d_dummy, volatile uint8_t *d_newstate_flags) {
	// Iterate over the designated work.
	shared_indextype entry_id;
	shared_inttype src_state;

	#pragma unroll
	for (shared_indextype i = WARP_ID; i/OPENTILE_WARP_WIDTH < NR_SMS; i += NR_WARPS_PER_BLOCK) {	
		entry_id = ((i % OPENTILE_WARP_WIDTH) * WARP_SIZE);
		if (entry_id < OPENTILECOUNT) {
			entry_id = get_sorted_opentile_element(i);
		}
		if (entry_id < OPENTILECOUNT) {
			src_state = shared[OPENTILEOFFSET+(2*entry_id)+1];
			if (get_successors_of_sm((shared_indextype) src_state, i/OPENTILE_WARP_WIDTH, d_q, d_q_i, d_dummy, d_newstate_flags) == HASH_TABLE_FULL) {
				return HASH_TABLE_FULL;
			}
		}
	}
	return STORED;
} 

// *** START STRUCT AND FUNCTIONS FOR PRINTING ***

// A struct to represent a system state (used on the CPU side).
struct systemstate_t {
	statetype globalObject_P_0;
	statetype globalObject_P_4;
	statetype globalObject_P_2;
	statetype globalObject_P_3;
	statetype globalObject_P_1;
	elem_chartype globalObject_P_3_myplace;
	elem_chartype globalObject_P_0_myplace;
	elem_chartype globalObject_next;
	elem_chartype globalObject_Slot_0_;
	elem_chartype globalObject_Slot_1_;
	elem_chartype globalObject_Slot_2_;
	elem_chartype globalObject_Slot_3_;
	elem_chartype globalObject_Slot_4_;
	elem_chartype globalObject_P_1_myplace;
	elem_chartype globalObject_P_4_myplace;
	elem_chartype globalObject_P_2_myplace;
};

// A function to create a system state struct instance with the given values.
systemstate_t create_systemstate(statetype globalObject_P_0, statetype globalObject_P_4, statetype globalObject_P_2, statetype globalObject_P_3, statetype globalObject_P_1, elem_chartype globalObject_P_3_myplace, elem_chartype globalObject_P_0_myplace, elem_chartype globalObject_next, elem_chartype globalObject_Slot_0_, elem_chartype globalObject_Slot_1_, elem_chartype globalObject_Slot_2_, elem_chartype globalObject_Slot_3_, elem_chartype globalObject_Slot_4_, elem_chartype globalObject_P_1_myplace, elem_chartype globalObject_P_4_myplace, elem_chartype globalObject_P_2_myplace) {
	systemstate_t newstate;
	newstate.globalObject_P_0 = globalObject_P_0;
	newstate.globalObject_P_4 = globalObject_P_4;
	newstate.globalObject_P_2 = globalObject_P_2;
	newstate.globalObject_P_3 = globalObject_P_3;
	newstate.globalObject_P_1 = globalObject_P_1;
	newstate.globalObject_P_3_myplace = globalObject_P_3_myplace;
	newstate.globalObject_P_0_myplace = globalObject_P_0_myplace;
	newstate.globalObject_next = globalObject_next;
	newstate.globalObject_Slot_0_ = globalObject_Slot_0_;
	newstate.globalObject_Slot_1_ = globalObject_Slot_1_;
	newstate.globalObject_Slot_2_ = globalObject_Slot_2_;
	newstate.globalObject_Slot_3_ = globalObject_Slot_3_;
	newstate.globalObject_Slot_4_ = globalObject_Slot_4_;
	newstate.globalObject_P_1_myplace = globalObject_P_1_myplace;
	newstate.globalObject_P_4_myplace = globalObject_P_4_myplace;
	newstate.globalObject_P_2_myplace = globalObject_P_2_myplace;
	return newstate;
}

// State comparison function for sorting states.
bool systemstates_compare(systemstate_t s, systemstate_t t) {
	if (s.globalObject_P_0 != t.globalObject_P_0) {
		return (s.globalObject_P_0 < t.globalObject_P_0);
	}
	if (s.globalObject_P_4 != t.globalObject_P_4) {
		return (s.globalObject_P_4 < t.globalObject_P_4);
	}
	if (s.globalObject_P_2 != t.globalObject_P_2) {
		return (s.globalObject_P_2 < t.globalObject_P_2);
	}
	if (s.globalObject_P_3 != t.globalObject_P_3) {
		return (s.globalObject_P_3 < t.globalObject_P_3);
	}
	if (s.globalObject_P_1 != t.globalObject_P_1) {
		return (s.globalObject_P_1 < t.globalObject_P_1);
	}
	if (s.globalObject_P_3_myplace != t.globalObject_P_3_myplace) {
		return (s.globalObject_P_3_myplace < t.globalObject_P_3_myplace);
	}
	if (s.globalObject_P_0_myplace != t.globalObject_P_0_myplace) {
		return (s.globalObject_P_0_myplace < t.globalObject_P_0_myplace);
	}
	if (s.globalObject_next != t.globalObject_next) {
		return (s.globalObject_next < t.globalObject_next);
	}
	if (s.globalObject_Slot_0_ != t.globalObject_Slot_0_) {
		return (s.globalObject_Slot_0_ < t.globalObject_Slot_0_);
	}
	if (s.globalObject_Slot_1_ != t.globalObject_Slot_1_) {
		return (s.globalObject_Slot_1_ < t.globalObject_Slot_1_);
	}
	if (s.globalObject_Slot_2_ != t.globalObject_Slot_2_) {
		return (s.globalObject_Slot_2_ < t.globalObject_Slot_2_);
	}
	if (s.globalObject_Slot_3_ != t.globalObject_Slot_3_) {
		return (s.globalObject_Slot_3_ < t.globalObject_Slot_3_);
	}
	if (s.globalObject_Slot_4_ != t.globalObject_Slot_4_) {
		return (s.globalObject_Slot_4_ < t.globalObject_Slot_4_);
	}
	if (s.globalObject_P_1_myplace != t.globalObject_P_1_myplace) {
		return (s.globalObject_P_1_myplace < t.globalObject_P_1_myplace);
	}
	if (s.globalObject_P_4_myplace != t.globalObject_P_4_myplace) {
		return (s.globalObject_P_4_myplace < t.globalObject_P_4_myplace);
	}
	if (s.globalObject_P_2_myplace != t.globalObject_P_2_myplace) {
		return (s.globalObject_P_2_myplace < t.globalObject_P_2_myplace);
	}
	return false;
}

void print_content_hash_table(FILE* stream, compressed_nodetype *q, nodetype *q_i, uint64_t q_size, uint64_t q_i_size, bool print_pointers) {
	uint64_t counter = 0;
	fprintf(stream, "BEGIN HASH TABLE CONTENTS.\n");
	for (uint64_t i = 0; i < q_size; i++) {
		if (q[i] != EMPTY_COMPRESSED_NODE) {
			counter++;
			// Retrieve state vector.
			nodetype root = HT_RETRIEVE(q, q_i, i, true);
			printf("retrieved node: %lu\n", root);
			nodetype part0 = host_direct_get_vectorpart_0(q, q_i, root, stream, print_pointers);
			nodetype part1 = host_direct_get_vectorpart_1(q, q_i, root, stream, print_pointers);
			// Print the contents of the state.
			nodetype *p1, *p2;
			statetype e_st;
			elem_booltype e_b;
			elem_chartype e_c;
			elem_inttype e_i;
			fprintf(stream, "-----\n");
			fprintf(stream, "At index %lu:\n", i);
			p1 = &part0;
			p2 = p1;
			host_get_globalObject_P_0(&e_st, *p1, *p2);
			fprintf(stream, "state globalObject'P_0: %d\n", (uint8_t) e_st);
			p1 = &part0;
			p2 = p1;
			host_get_globalObject_P_4(&e_st, *p1, *p2);
			fprintf(stream, "state globalObject'P_4: %d\n", (uint8_t) e_st);
			p1 = &part0;
			p2 = p1;
			host_get_globalObject_P_2(&e_st, *p1, *p2);
			fprintf(stream, "state globalObject'P_2: %d\n", (uint8_t) e_st);
			p1 = &part0;
			p2 = p1;
			host_get_globalObject_P_3(&e_st, *p1, *p2);
			fprintf(stream, "state globalObject'P_3: %d\n", (uint8_t) e_st);
			p1 = &part0;
			p2 = p1;
			host_get_globalObject_P_1(&e_st, *p1, *p2);
			fprintf(stream, "state globalObject'P_1: %d\n", (uint8_t) e_st);
			p1 = &part0;
			p2 = p1;
			host_get_globalObject_P_3_myplace(&e_c, *p1, *p2);
			fprintf(stream, "variable globalObject'P_3'myplace: %d\n", (uint8_t) e_c);
			p1 = &part0;
			p2 = p1;
			host_get_globalObject_P_0_myplace(&e_c, *p1, *p2);
			fprintf(stream, "variable globalObject'P_0'myplace: %d\n", (uint8_t) e_c);
			p1 = &part0;
			p2 = p1;
			host_get_globalObject_next(&e_c, *p1, *p2);
			fprintf(stream, "variable globalObject'next: %d\n", (uint8_t) e_c);
			p1 = &part1;
			p2 = p1;
			host_get_globalObject_P_1_myplace(&e_c, *p1, *p2);
			fprintf(stream, "variable globalObject'P_1'myplace: %d\n", (uint8_t) e_c);
			p1 = &part1;
			p2 = p1;
			host_get_globalObject_P_4_myplace(&e_c, *p1, *p2);
			fprintf(stream, "variable globalObject'P_4'myplace: %d\n", (uint8_t) e_c);
			p1 = &part1;
			p2 = p1;
			host_get_globalObject_P_2_myplace(&e_c, *p1, *p2);
			fprintf(stream, "variable globalObject'P_2'myplace: %d\n", (uint8_t) e_c);
			p1 = &part0;
			p2 = p1;
			host_get_globalObject_Slot(&e_c, *p1, *p2, 0);
			fprintf(stream, "array element globalObject'Slot[0]: %u\n", (uint32_t) e_c);
			p1 = &part0;
			p2 = p1;
			host_get_globalObject_Slot(&e_c, *p1, *p2, 1);
			fprintf(stream, "array element globalObject'Slot[1]: %u\n", (uint32_t) e_c);
			p1 = &part0;
			p2 = p1;
			host_get_globalObject_Slot(&e_c, *p1, *p2, 2);
			fprintf(stream, "array element globalObject'Slot[2]: %u\n", (uint32_t) e_c);
			p1 = &part1;
			p2 = p1;
			host_get_globalObject_Slot(&e_c, *p1, *p2, 3);
			fprintf(stream, "array element globalObject'Slot[3]: %u\n", (uint32_t) e_c);
			p1 = &part1;
			p2 = p1;
			host_get_globalObject_Slot(&e_c, *p1, *p2, 4);
			fprintf(stream, "array element globalObject'Slot[4]: %u\n", (uint32_t) e_c);
			fprintf(stream, "-----\n");
		}
	}
	if (print_pointers) {
		for (uint64_t i = 0; i < q_i_size; i++) {
			if (q_i[i] != EMPTY_NODE) {
				fprintf(stream, "internal node present at index %lu.\n", i);
			}
		}
	}
	fprintf(stream, "END HASH TABLE CONTENTS.\n");
	fprintf(stream, "NR. OF STATES: %lu.\n", counter);
}

systemstate_t get_systemstate(compressed_nodetype *q, uint64_t index, nodetype *q_i) {
	systemstate_t newstate;
	// Retrieve state vector.
	nodetype root = HT_RETRIEVE(q, q_i, index, true);
	nodetype part0 = host_direct_get_vectorpart_0(q, q_i, root, NULL, false);
	nodetype part1 = host_direct_get_vectorpart_1(q, q_i, root, NULL, false);
	nodetype *p1, *p2;
	statetype e_st;
	elem_booltype e_b;
	elem_chartype e_c;
	elem_inttype e_i;
	p1 = &part0;
	p2 = p1;
	host_get_globalObject_P_0(&e_st, *p1, *p2);
	newstate.globalObject_P_0 = e_st;
	p1 = &part0;
	p2 = p1;
	host_get_globalObject_P_4(&e_st, *p1, *p2);
	newstate.globalObject_P_4 = e_st;
	p1 = &part0;
	p2 = p1;
	host_get_globalObject_P_2(&e_st, *p1, *p2);
	newstate.globalObject_P_2 = e_st;
	p1 = &part0;
	p2 = p1;
	host_get_globalObject_P_3(&e_st, *p1, *p2);
	newstate.globalObject_P_3 = e_st;
	p1 = &part0;
	p2 = p1;
	host_get_globalObject_P_1(&e_st, *p1, *p2);
	newstate.globalObject_P_1 = e_st;
	p1 = &part0;
	p2 = p1;
	host_get_globalObject_P_3_myplace(&e_c, *p1, *p2);
	newstate.globalObject_P_3_myplace = e_c;
	p1 = &part0;
	p2 = p1;
	host_get_globalObject_P_0_myplace(&e_c, *p1, *p2);
	newstate.globalObject_P_0_myplace = e_c;
	p1 = &part0;
	p2 = p1;
	host_get_globalObject_next(&e_c, *p1, *p2);
	newstate.globalObject_next = e_c;
	p1 = &part1;
	p2 = p1;
	host_get_globalObject_P_1_myplace(&e_c, *p1, *p2);
	newstate.globalObject_P_1_myplace = e_c;
	p1 = &part1;
	p2 = p1;
	host_get_globalObject_P_4_myplace(&e_c, *p1, *p2);
	newstate.globalObject_P_4_myplace = e_c;
	p1 = &part1;
	p2 = p1;
	host_get_globalObject_P_2_myplace(&e_c, *p1, *p2);
	newstate.globalObject_P_2_myplace = e_c;
	p1 = &part0;
	p2 = p1;
	host_get_globalObject_Slot(&e_c, *p1, *p2, 0);
	newstate.globalObject_Slot_0_ = e_c;
	p1 = &part0;
	p2 = p1;
	host_get_globalObject_Slot(&e_c, *p1, *p2, 1);
	newstate.globalObject_Slot_1_ = e_c;
	p1 = &part0;
	p2 = p1;
	host_get_globalObject_Slot(&e_c, *p1, *p2, 2);
	newstate.globalObject_Slot_2_ = e_c;
	p1 = &part1;
	p2 = p1;
	host_get_globalObject_Slot(&e_c, *p1, *p2, 3);
	newstate.globalObject_Slot_3_ = e_c;
	p1 = &part1;
	p2 = p1;
	host_get_globalObject_Slot(&e_c, *p1, *p2, 4);
	newstate.globalObject_Slot_4_ = e_c;
	return newstate;
}

void print_systemstate(FILE* stream, systemstate_t s) {
	fprintf(stream, "%d ", (uint8_t) s.globalObject_P_0);
	fprintf(stream, "%d ", (uint8_t) s.globalObject_P_4);
	fprintf(stream, "%d ", (uint8_t) s.globalObject_P_2);
	fprintf(stream, "%d ", (uint8_t) s.globalObject_P_3);
	fprintf(stream, "%d ", (uint8_t) s.globalObject_P_1);
	fprintf(stream, "%d ", (uint8_t) s.globalObject_P_3_myplace);
	fprintf(stream, "%d ", (uint8_t) s.globalObject_P_0_myplace);
	fprintf(stream, "%d ", (uint8_t) s.globalObject_next);
	fprintf(stream, "%d ", (uint8_t) s.globalObject_Slot_0_);
	fprintf(stream, "%d ", (uint8_t) s.globalObject_Slot_1_);
	fprintf(stream, "%d ", (uint8_t) s.globalObject_Slot_2_);
	fprintf(stream, "%d ", (uint8_t) s.globalObject_Slot_3_);
	fprintf(stream, "%d ", (uint8_t) s.globalObject_Slot_4_);
	fprintf(stream, "%d ", (uint8_t) s.globalObject_P_1_myplace);
	fprintf(stream, "%d ", (uint8_t) s.globalObject_P_4_myplace);
	fprintf(stream, "%d ", (uint8_t) s.globalObject_P_2_myplace);
	fprintf(stream, "\n");	
}

void print_systemstates(FILE* stream, std::vector<systemstate_t> states) {
	fprintf(stream, "GENERATED SYSTEM STATES:\n");
	for (std::vector<systemstate_t>::iterator it = states.begin(); it != states.end(); ++it) {
		print_systemstate(stream, (*it));
	}
	fprintf(stream, "END OF STATE LIST.\n");
	fprintf(stream, "NR. OF STATES: %lu.\n", states.size());
}

// *** END STRUCT AND FUNCTIONS FOR PRINTING ***