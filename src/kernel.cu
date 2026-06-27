#define GLM_FORCE_CUDA

#include <cuda.h>
#include "kernel.h"
#include "utilityCore.hpp"

#include <cmath>
#include <cstdio>
#include <iostream>
#include <vector>

#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/device_vector.h>

#include <glm/glm.hpp>

// LOOK-2.1 potentially useful for doing grid-based neighbor search
#ifndef imax
#define imax( a, b ) ( ((a) > (b)) ? (a) : (b) )
#endif

#ifndef imin
#define imin( a, b ) ( ((a) < (b)) ? (a) : (b) )
#endif

#define checkCUDAErrorWithLine(msg) checkCUDAError(msg, __LINE__)

/**
* Check for CUDA errors; print and exit if there was a problem.
*/
void checkCUDAError(const char *msg, int line = -1) {
  cudaError_t err = cudaGetLastError();
  if (cudaSuccess != err) {
    if (line >= 0) {
      fprintf(stderr, "Line %d: ", line);
    }
    fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
}


/*****************
* Configuration *
*****************/

/*! Block size used for CUDA kernel launch. */
#define blockSize 128

// Runtime configurable block size (overrides blockSize if set)
int g_runtimeBlockSize = 0;  // 0 means use default blockSize

// LOOK-1.2 Parameters for the boids algorithm.
// These worked well in our reference implementation.
#define rule1Distance 5.0f
#define rule2Distance 3.0f
#define rule3Distance 5.0f

#define rule1Scale 0.01f
#define rule2Scale 0.1f
#define rule3Scale 0.1f

#define maxSpeed 1.0f

/*! Size of the starting area in simulation space. */
#define scene_scale 100.0f

/***********************************************
* Kernel state (pointers are device pointers) *
***********************************************/

int numObjects;
dim3 threadsPerBlock(blockSize);

// Helper function to get effective block size
inline int getEffectiveBlockSize() {
    return (g_runtimeBlockSize > 0) ? g_runtimeBlockSize : blockSize;
}

// Helper function to update threadsPerBlock
inline void updateThreadsPerBlock() {
    int effectiveBlockSize = getEffectiveBlockSize();
    threadsPerBlock = dim3(effectiveBlockSize);
}

// LOOK-1.2 - These buffers are here to hold all your boid information.
// These get allocated for you in Boids::initSimulation.
// Consider why you would need two velocity buffers in a simulation where each
// boid cares about its neighbors' velocities.
// These are called ping-pong buffers.
glm::vec3 *dev_pos;
glm::vec3 *dev_vel1;//old velocities
glm::vec3 *dev_vel2;//new velocities

// LOOK-2.1 - these are NOT allocated for you. You'll have to set up the thrust
// pointers on your own too.

// For efficient sorting and the uniform grid. These should always be parallel.
int *dev_particleArrayIndices; // What index in dev_pos and dev_velX represents this particle?
int *dev_particleGridIndices; // What grid cell is this particle in?
// needed for use with thrust
thrust::device_ptr<int> dev_thrust_particleArrayIndices;
thrust::device_ptr<int> dev_thrust_particleGridIndices;

int *dev_gridCellStartIndices; // What part of dev_particleArrayIndices belongs
int *dev_gridCellEndIndices;   // to this cell?

// TODO-2.3 - consider what additional buffers you might need to reshuffle
// the position and velocity data to be coherent within cells.

// Coherent buffers: positions and velocities rearranged to be contiguous within cells
glm::vec3 *dev_coherentPos;
glm::vec3 *dev_coherentVel1; // Old velocities (coherent)
glm::vec3 *dev_coherentVel2; // New velocities (coherent)

// LOOK-2.1 - Grid parameters based on simulation parameters.
// These are automatically computed for you in Boids::initSimulation
int gridCellCount;
int gridSideCount;
float gridCellWidth;
float gridInverseCellWidth;
glm::vec3 gridMinimum;

/***********************************************
* Performance Measurement using CUDA Events   *
***********************************************/

// CUDA Events for performance measurement
cudaEvent_t perfEvent_start;
cudaEvent_t perfEvent_stop;
cudaEvent_t perfEvent_kernelStart;
cudaEvent_t perfEvent_kernelStop;

// Performance metrics storage
Boids::PerformanceMetrics g_perfMetrics;

/******************
* initSimulation *
******************/

__host__ __device__ unsigned int hash(unsigned int a) {
  a = (a + 0x7ed55d16) + (a << 12);
  a = (a ^ 0xc761c23c) ^ (a >> 19);
  a = (a + 0x165667b1) + (a << 5);
  a = (a + 0xd3a2646c) ^ (a << 9);
  a = (a + 0xfd7046c5) + (a << 3);
  a = (a ^ 0xb55a4f09) ^ (a >> 16);
  return a;
}

/**
* LOOK-1.2 - this is a typical helper function for a CUDA kernel.
* Function for generating a random vec3.
*/
__host__ __device__ glm::vec3 generateRandomVec3(float time, int index) {
  thrust::default_random_engine rng(hash((int)(index * time)));
  thrust::uniform_real_distribution<float> unitDistrib(-1, 1);

  return glm::vec3((float)unitDistrib(rng), (float)unitDistrib(rng), (float)unitDistrib(rng));
}

/**
* LOOK-1.2 - This is a basic CUDA kernel.
* CUDA kernel for generating boids with a specified mass randomly around the star.
*/
__global__ void kernGenerateRandomPosArray(int time, int N, glm::vec3 * arr, float scale) {
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (index < N) {
    glm::vec3 rand = generateRandomVec3(time, index);
    arr[index].x = scale * rand.x;
    arr[index].y = scale * rand.y;
    arr[index].z = scale * rand.z;
  }
}

/**
* Initialize memory, update some globals
*/
void Boids::initSimulation(int N, int blockSz) {
  // Set runtime block size before calling the main init
  g_runtimeBlockSize = blockSz;
  Boids::initSimulation(N);
}

void Boids::initSimulation(int N) {
  numObjects = N;
  int effBS = getEffectiveBlockSize();
  dim3 fullBlocksPerGrid((N + effBS - 1) / effBS);

  // LOOK-1.2 - This is basic CUDA memory management and error checking.
  // Don't forget to cudaFree in  Boids::endSimulation.
  cudaMalloc((void**)&dev_pos, N * sizeof(glm::vec3));
  checkCUDAErrorWithLine("cudaMalloc dev_pos failed!");

  cudaMalloc((void**)&dev_vel1, N * sizeof(glm::vec3));
  checkCUDAErrorWithLine("cudaMalloc dev_vel1 failed!");

  cudaMalloc((void**)&dev_vel2, N * sizeof(glm::vec3));
  checkCUDAErrorWithLine("cudaMalloc dev_vel2 failed!");

  // LOOK-1.2 - This is a typical CUDA kernel invocation.
  kernGenerateRandomPosArray<<<fullBlocksPerGrid, blockSize>>>(1, numObjects,
    dev_pos, scene_scale);
  checkCUDAErrorWithLine("kernGenerateRandomPosArray failed!");

  // LOOK-2.1 computing grid params
  gridCellWidth = 2.0f * std::max(std::max(rule1Distance, rule2Distance), rule3Distance);
  int halfSideCount = (int)(scene_scale / gridCellWidth) + 1;
  gridSideCount = 2 * halfSideCount;

  gridCellCount = gridSideCount * gridSideCount * gridSideCount;
  gridInverseCellWidth = 1.0f / gridCellWidth;
  float halfGridWidth = gridCellWidth * halfSideCount;
  gridMinimum.x -= halfGridWidth;
  gridMinimum.y -= halfGridWidth;
  gridMinimum.z -= halfGridWidth;

  // TODO-2.1 TODO-2.3 - Allocate additional buffers here.
  cudaMalloc((void**)&dev_particleArrayIndices, N * sizeof(int));
  checkCUDAErrorWithLine("cudaMalloc dev_particleArrayIndices failed!");

  cudaMalloc((void**)&dev_particleGridIndices, N * sizeof(int));
  checkCUDAErrorWithLine("cudaMalloc dev_particleGridIndices failed!");

  cudaMalloc((void**)&dev_gridCellStartIndices, gridCellCount * sizeof(int));
  checkCUDAErrorWithLine("cudaMalloc dev_gridCellStartIndices failed!");

  cudaMalloc((void**)&dev_gridCellEndIndices, gridCellCount * sizeof(int));
  checkCUDAErrorWithLine("cudaMalloc dev_gridCellEndIndices failed!");

  // TODO-2.3 - Allocate coherent buffers for rearranged data
  cudaMalloc((void**)&dev_coherentPos, N * sizeof(glm::vec3));
  checkCUDAErrorWithLine("cudaMalloc dev_coherentPos failed!");

  cudaMalloc((void**)&dev_coherentVel1, N * sizeof(glm::vec3));
  checkCUDAErrorWithLine("cudaMalloc dev_coherentVel1 failed!");

  cudaMalloc((void**)&dev_coherentVel2, N * sizeof(glm::vec3));
  checkCUDAErrorWithLine("cudaMalloc dev_coherentVel2 failed!");

  // Wrap device pointers in thrust device_ptr for sorting
  //i.e. int* -> device_prt<int>,avoids ambiguous sorting mechanism
  dev_thrust_particleArrayIndices = thrust::device_ptr<int>(dev_particleArrayIndices);
  dev_thrust_particleGridIndices = thrust::device_ptr<int>(dev_particleGridIndices);
  
  // Initialize CUDA Events for performance measurement
  //hardware handle creation is expensive,reutilize is strongly recommended
  cudaEventCreate(&perfEvent_start);
  cudaEventCreate(&perfEvent_stop);
  cudaEventCreate(&perfEvent_kernelStart);
  cudaEventCreate(&perfEvent_kernelStop);
  
  // Reset performance metrics
  resetPerformanceMetrics();
  
  cudaDeviceSynchronize();
}


/******************
* copyBoidsToVBO *
******************/

/**
* Copy the boid positions into the VBO so that they can be drawn by OpenGL.
*/
__global__ void kernCopyPositionsToVBO(int N, glm::vec3 *pos, float *vbo, float s_scale) {
  int index = threadIdx.x + (blockIdx.x * blockDim.x);

  float c_scale = -1.0f / s_scale;

  if (index < N) {
    vbo[4 * index + 0] = pos[index].x * c_scale;
    vbo[4 * index + 1] = pos[index].y * c_scale;
    vbo[4 * index + 2] = pos[index].z * c_scale;
    vbo[4 * index + 3] = 1.0f;//represents a point
    //0.0f：represetns a vector(translation matrix is invalid)
  }
}

__global__ void kernCopyVelocitiesToVBO(int N, glm::vec3 *vel, float *vbo, float s_scale) {
  int index = threadIdx.x + (blockIdx.x * blockDim.x);

  if (index < N) {
    vbo[4 * index + 0] = vel[index].x + 0.3f;//base brightness,avoids pure black
    vbo[4 * index + 1] = vel[index].y + 0.3f;
    vbo[4 * index + 2] = vel[index].z + 0.3f;
    vbo[4 * index + 3] = 1.0f;//represents a point
  }
}

/**
* Wrapper for call to the kernCopyboidsToVBO CUDA kernel.
*/
void Boids::copyBoidsToVBO(float *vbodptr_positions, float *vbodptr_velocities) {
  dim3 fullBlocksPerGrid((numObjects + blockSize - 1) / blockSize);

  kernCopyPositionsToVBO << <fullBlocksPerGrid, blockSize >> >(numObjects, dev_pos, vbodptr_positions, scene_scale);
  kernCopyVelocitiesToVBO << <fullBlocksPerGrid, blockSize >> >(numObjects, dev_vel1, vbodptr_velocities, scene_scale);

  checkCUDAErrorWithLine("copyBoidsToVBO failed!");

  cudaDeviceSynchronize();
}


/******************
* stepSimulation *
******************/

/**
* LOOK-1.2 You can use this as a helper for kernUpdateVelocityBruteForce.
* __device__ code can be called from a __global__ context
* Compute the new velocity on the body with index `iSelf` due to the `N` boids
* in the `pos` and `vel` arrays.
*/
__device__ glm::vec3 computeVelocityChange(int N, int iSelf, const glm::vec3 *pos, const glm::vec3 *vel) {
  glm::vec3 centerOfMass(0.0f, 0.0f, 0.0f);
  glm::vec3 separation(0.0f, 0.0f, 0.0f);
  glm::vec3 averageVelocity(0.0f, 0.0f, 0.0f);

  int rule1Count = 0;
  int rule3Count = 0;

  glm::vec3 selfPos = pos[iSelf];

  for (int i = 0; i < N; i++) {
    if (i == iSelf) {
      continue;
    }

    glm::vec3 offset = pos[i] - selfPos;//the offset vector from the current boid to the neighbor boid
    float distance = sqrtf(offset.x * offset.x + offset.y * offset.y + offset.z * offset.z);

    if (distance < rule1Distance) {
      centerOfMass += pos[i];
      rule1Count++;
    }

    if (distance < rule2Distance) {
      separation -= offset;//the closer the neighbor, the greater the repulsion
    }

    if (distance < rule3Distance) {
      averageVelocity += vel[i];
      rule3Count++;
    }
  }

  glm::vec3 velocityChange(0.0f, 0.0f, 0.0f);

  if (rule1Count > 0) {
    centerOfMass /= (float)rule1Count;
    velocityChange += (centerOfMass - selfPos) * rule1Scale;
  }

  velocityChange += separation * rule2Scale;

  if (rule3Count > 0) {
    averageVelocity /= (float)rule3Count;
    velocityChange += averageVelocity * rule3Scale;
  }

  return velocityChange;
}

/**
* TODO-1.2 implement basic flocking
* For each of the `N` bodies, update its position based on its current velocity.
*/
__global__ void kernUpdateVelocityBruteForce(int N, glm::vec3 *pos,
  glm::vec3 *vel1, glm::vec3 *vel2) {
  // Compute a new velocity based on pos and vel1
  // Clamp the speed
  // Record the new velocity into vel2. Question: why NOT vel1?
  int index = threadIdx.x + (blockIdx.x * blockDim.x);
  if (index >= N) {
    return;
  }

  glm::vec3 newVelocity = vel1[index] + computeVelocityChange(N, index, pos, vel1);
  float speed = sqrtf(newVelocity.x * newVelocity.x + newVelocity.y * newVelocity.y + newVelocity.z * newVelocity.z);

  if (speed > maxSpeed && speed > 0.0f) {
    newVelocity = (newVelocity / speed) * maxSpeed;
  }

  vel2[index] = newVelocity;
}

/**
* LOOK-1.2 Since this is pretty trivial, we implemented it for you.
* For each of the `N` bodies, update its position based on its current velocity.
*/
__global__ void kernUpdatePos(int N, float dt, glm::vec3 *pos, glm::vec3 *vel) {
  // Update position by velocity
  int index = threadIdx.x + (blockIdx.x * blockDim.x);
  if (index >= N) {
    return;
  }
  glm::vec3 thisPos = pos[index];
  thisPos += vel[index] * dt;

  // Wrap the boids around so we don't lose them
  thisPos.x = thisPos.x < -scene_scale ? scene_scale : thisPos.x;
  thisPos.y = thisPos.y < -scene_scale ? scene_scale : thisPos.y;
  thisPos.z = thisPos.z < -scene_scale ? scene_scale : thisPos.z;

  thisPos.x = thisPos.x > scene_scale ? -scene_scale : thisPos.x;
  thisPos.y = thisPos.y > scene_scale ? -scene_scale : thisPos.y;
  thisPos.z = thisPos.z > scene_scale ? -scene_scale : thisPos.z;

  pos[index] = thisPos;
}

// LOOK-2.1 Consider this method of computing a 1D index from a 3D grid index.
// LOOK-2.3 Looking at this method, what would be the most memory efficient
//          order for iterating over neighboring grid cells?
//          for(x)
//            for(y)
//             for(z)? Or some other order?
__device__ int gridIndex3Dto1D(int x, int y, int z, int gridResolution) {
  return x + y * gridResolution + z * gridResolution * gridResolution;
}

__global__ void kernComputeIndices(int N, int gridResolution,
  glm::vec3 gridMin, float inverseCellWidth,
  glm::vec3 *pos, int *indices, int *gridIndices) {
    // TODO-2.1
    // - Label each boid with the index of its grid cell.
    // - Set up a parallel array of integer indices as pointers to the actual
    //   boid data in pos and vel1/vel2
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (index >= N) {
        return;
    }

    // Compute grid cell position for this boid
    glm::vec3 gridPos = (pos[index] - gridMin) * inverseCellWidth;
    
    // Convert to integer cell coordinates
    int gridX = (int)gridPos.x;
    int gridY = (int)gridPos.y;
    int gridZ = (int)gridPos.z;

    // Clamp to valid grid range
    gridX = imax(0, imin(gridX, gridResolution - 1));
    gridY = imax(0, imin(gridY, gridResolution - 1));
    gridZ = imax(0, imin(gridZ, gridResolution - 1));

    // Convert 3D grid index to 1D grid index
    int gridIndex = gridIndex3Dto1D(gridX, gridY, gridZ, gridResolution);

    // Store grid index for this particle
    gridIndices[index] = gridIndex;
    
    // Store the particle's array index (this will be sorted along with gridIndices)
    indices[index] = index;
}

// LOOK-2.1 Consider how this could be useful for indicating that a cell
//          does not enclose any boids
__global__ void kernResetIntBuffer(int N, int *intBuffer, int value) {
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (index < N) {
    intBuffer[index] = value;
  }
}

__global__ void kernIdentifyCellStartEnd(int N, int *particleGridIndices,
  int *gridCellStartIndices, int *gridCellEndIndices, int totalCellCount) {
  // TODO-2.1
  // Identify the start point of each cell in the gridIndices array.
  // This is basically a parallel unrolling of a loop that goes
  // "this index doesn't match the one before it, must be a new cell!"
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (index >= N) {
      return;
  }

  int currentCellIndex = particleGridIndices[index];
  
  // DEFENSIVE: Validate currentCellIndex to prevent out-of-bounds writes
  if (currentCellIndex < 0 || currentCellIndex >= totalCellCount) {
      return;  // Invalid cell index, skip
  }

  //boid index:   [  0  ] [  1  ] [  2  ] [  3  ]
  //gridindexes:  [  2  ] [  2  ] [  5  ] [  5  ]

  // Check if this is the start of a new cell
  //i.e. if the left bird has a different grid index with me ,i am the leftmost guy
  if (index == 0 || particleGridIndices[index - 1] != currentCellIndex) {
      gridCellStartIndices[currentCellIndex] = index;
  }

  // Check if this is the end of a cell
  //i.e. if the right bird has a different grid index with me,i am the rightmost guy
  if (index == N - 1 || particleGridIndices[index + 1] != currentCellIndex) {
      gridCellEndIndices[currentCellIndex] = index;
  }
}

__global__ void kernUpdateVelNeighborSearchScattered(
  int N, int gridResolution, glm::vec3 gridMin,
  float inverseCellWidth, float cellWidth,
  int *gridCellStartIndices, int *gridCellEndIndices,
  int *particleArrayIndices,
  glm::vec3 *pos, glm::vec3 *vel1, glm::vec3 *vel2) {
  // TODO-2.1 - Update a boid's velocity using the uniform grid to reduce
  // the number of boids that need to be checked.
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (index >= N) {
      return;
  }

  // - Identify the grid cell that this particle is in
  glm::vec3 selfPos = pos[index];
  
  // Compute grid cell position for this boid
  glm::vec3 gridPos = (selfPos - gridMin) * inverseCellWidth;
  int gridX = (int)gridPos.x;
  int gridY = (int)gridPos.y;
  int gridZ = (int)gridPos.z;

  // Apply the three boid rules
  glm::vec3 centerOfMass(0.0f);
  glm::vec3 separation(0.0f);
  glm::vec3 averageVelocity(0.0f);
  int rule1Count = 0;
  int rule3Count = 0;

  // Determine the search radius in grid cells
  float maxSearchDistance = fmaxf(fmaxf(rule1Distance, rule2Distance), rule3Distance);
  int cellRadius = (int)(maxSearchDistance * inverseCellWidth) + 1;

  // - Identify which cells may contain neighbors. This isn't always 8.
  // Search neighboring cells
  for (int dz = -cellRadius; dz <= cellRadius; dz++) {
    for (int dy = -cellRadius; dy <= cellRadius; dy++) {
      for (int dx = -cellRadius; dx <= cellRadius; dx++) {
        int neighborX = gridX + dx;
        int neighborY = gridY + dy;
        int neighborZ = gridZ + dz;

        // Check if neighbor cell is within grid bounds
        if (neighborX < 0 || neighborX >= gridResolution ||
            neighborY < 0 || neighborY >= gridResolution ||
            neighborZ < 0 || neighborZ >= gridResolution) {
          continue;
        }

        int neighborCellIndex = gridIndex3Dto1D(neighborX, neighborY, neighborZ, gridResolution);
        
        // DEFENSIVE: Additional bounds check for neighborCellIndex
        if (neighborCellIndex < 0 || neighborCellIndex >= gridResolution * gridResolution * gridResolution) {
          continue;
        }
        
        // - For each cell, read the start/end indices in the boid pointer array.
        int startIdx = gridCellStartIndices[neighborCellIndex];
        
        // Skip empty cells (check for sentinel value -1)
        if (startIdx == -1) {
          continue;
        }

        int endIdx = gridCellEndIndices[neighborCellIndex];
        
        // DEFENSIVE: Validate endIdx
        if (endIdx < startIdx || endIdx >= N) {
          continue;
        }

        // Check all boids in this cell
        for (int i = startIdx; i <= endIdx; i++) {
          // DEFENSIVE: Bounds check on i before accessing particleArrayIndices
          if (i < 0 || i >= N) {
            continue;
          }
          
          //indirect addressing,neighborIdx might jump randomly,leading to consistent cache miss
          int neighborIdx = particleArrayIndices[i];
          
          // DEFENSIVE: Validate neighborIdx before using it to access arrays
          if (neighborIdx < 0 || neighborIdx >= N) {
            continue;
          }
          
          // Skip self
          if (neighborIdx == index) {
            continue;
          }

          // - Access each boid in the cell and compute velocity change from
          //   the boids rules, if this boid is within the neighborhood distance.

          glm::vec3 offset = pos[neighborIdx] - selfPos;
          float distance = glm::length(offset);

          // Rule 1: Cohesion
          if (distance < rule1Distance) {
            centerOfMass += pos[neighborIdx];
            rule1Count++;
          }

          // Rule 2: Separation
          if (distance < rule2Distance) {
            separation -= offset;
          }

          // Rule 3: Alignment
          if (distance < rule3Distance) {
            averageVelocity += vel1[neighborIdx];
            rule3Count++;
          }
        }
      }
    }
  }

  // Compute velocity change
  glm::vec3 velocityChange(0.0f);

  if (rule1Count > 0) {
    centerOfMass /= (float)rule1Count;
    velocityChange += (centerOfMass - selfPos) * rule1Scale;
  }

  velocityChange += separation * rule2Scale;

  if (rule3Count > 0) {
    averageVelocity /= (float)rule3Count;
    velocityChange += averageVelocity * rule3Scale;
  }

  // - Clamp the speed change before putting the new speed in vel2
  // Compute new velocity and clamp speed
  glm::vec3 newVelocity = vel1[index] + velocityChange;
  float speed = glm::length(newVelocity);

  if (speed > maxSpeed && speed > 0.0f) {  // DEFENSIVE: Check speed > 0 before division
    newVelocity = (newVelocity / speed) * maxSpeed;
  }

  vel2[index] = newVelocity;
}

__global__ void kernUpdateVelNeighborSearchCoherent(
  int N, int gridResolution, glm::vec3 gridMin,
  float inverseCellWidth, float cellWidth,
  int *gridCellStartIndices, int *gridCellEndIndices,
  glm::vec3 *pos, glm::vec3 *vel1, glm::vec3 *vel2) {
  // TODO-2.3 - This should be very similar to kernUpdateVelNeighborSearchScattered,
  // except with one less level of indirection.
  // This should expect gridCellStartIndices and gridCellEndIndices to refer
  // directly to pos and vel1.
  // - Identify the grid cell that this particle is in
  // - Identify which cells may contain neighbors. This isn't always 8.
  // - For each cell, read the start/end indices in the boid pointer array.
  //   DIFFERENCE: For best results, consider what order the cells should be
  //   checked in to maximize the memory benefits of reordering the boids data.
  // - Access each boid in the cell and compute velocity change from
  //   the boids rules, if this boid is within the neighborhood distance.
  // - Clamp the speed change before putting the new speed in vel2
  
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (index >= N) {
      return;
  }

  glm::vec3 selfPos = pos[index];
  
  // Compute grid cell position for this boid
  glm::vec3 gridPos = (selfPos - gridMin) * inverseCellWidth;
  int gridX = (int)gridPos.x;
  int gridY = (int)gridPos.y;
  int gridZ = (int)gridPos.z;

  // Apply the three boid rules
  glm::vec3 centerOfMass(0.0f);
  glm::vec3 separation(0.0f);
  glm::vec3 averageVelocity(0.0f);
  int rule1Count = 0;
  int rule3Count = 0;

  // Determine the search radius in grid cells
  float maxSearchDistance = fmaxf(fmaxf(rule1Distance, rule2Distance), rule3Distance);
  int cellRadius = (int)(maxSearchDistance * inverseCellWidth) + 1;

  // Search neighboring cells
  // Iterate in z-y-x order for better memory access patterns (matches gridIndex3Dto1D)
  for (int dz = -cellRadius; dz <= cellRadius; dz++) {
    for (int dy = -cellRadius; dy <= cellRadius; dy++) {
      for (int dx = -cellRadius; dx <= cellRadius; dx++) {
        int neighborX = gridX + dx;
        int neighborY = gridY + dy;
        int neighborZ = gridZ + dz;

        // Check if neighbor cell is within grid bounds
        if (neighborX < 0 || neighborX >= gridResolution ||
            neighborY < 0 || neighborY >= gridResolution ||
            neighborZ < 0 || neighborZ >= gridResolution) {
          continue;
        }

        int neighborCellIndex = gridIndex3Dto1D(neighborX, neighborY, neighborZ, gridResolution);
        
        // DEFENSIVE: Additional bounds check for neighborCellIndex
        if (neighborCellIndex < 0 || neighborCellIndex >= gridResolution * gridResolution * gridResolution) {
          continue;
        }
        
        int startIdx = gridCellStartIndices[neighborCellIndex];
        
        // Skip empty cells (check for sentinel value -1)
        if (startIdx == -1) {
          continue;
        }

        int endIdx = gridCellEndIndices[neighborCellIndex];
        
        // DEFENSIVE: Validate endIdx
        if (endIdx < startIdx || endIdx >= N) {
          continue;
        }

        // COHERENT DIFFERENCE: Direct access to pos and vel1 arrays
        // No indirection through particleArrayIndices
        for (int neighborIdx = startIdx; neighborIdx <= endIdx; neighborIdx++) {
          // DEFENSIVE: Bounds check
          if (neighborIdx < 0 || neighborIdx >= N) {
            continue;
          }
          
          // Skip self
          if (neighborIdx == index) {
            continue;
          }
          //direct addressing
          glm::vec3 offset = pos[neighborIdx] - selfPos;
          float distance = glm::length(offset);

          // Rule 1: Cohesion
          if (distance < rule1Distance) {
            centerOfMass += pos[neighborIdx];
            rule1Count++;
          }

          // Rule 2: Separation
          if (distance < rule2Distance) {
            separation -= offset;
          }

          // Rule 3: Alignment
          if (distance < rule3Distance) {
            averageVelocity += vel1[neighborIdx];
            rule3Count++;
          }
        }
      }
    }
  }

  // Compute velocity change
  glm::vec3 velocityChange(0.0f);

  if (rule1Count > 0) {
    centerOfMass /= (float)rule1Count;
    velocityChange += (centerOfMass - selfPos) * rule1Scale;
  }

  velocityChange += separation * rule2Scale;

  if (rule3Count > 0) {
    averageVelocity /= (float)rule3Count;
    velocityChange += averageVelocity * rule3Scale;
  }

  // Compute new velocity and clamp speed
  glm::vec3 newVelocity = vel1[index] + velocityChange;
  float speed = glm::length(newVelocity);

  if (speed > maxSpeed && speed > 0.0f) {  // DEFENSIVE: Check speed > 0 before division
    newVelocity = (newVelocity / speed) * maxSpeed;
  }

  vel2[index] = newVelocity;
}

// TODO-2.3 - Kernel to reshuffle boid data to be coherent within cells
__global__ void kernReshuffleData(int N, int *particleArrayIndices,
  glm::vec3 *oldPos, glm::vec3 *oldVel,
  glm::vec3 *newPos, glm::vec3 *newVel) {
  // Reshuffle position and velocity data according to sorted particle indices
  // This makes data access coherent within grid cells
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (index >= N) {
    return;
  }

  int originalIndex = particleArrayIndices[index];
  
  // DEFENSIVE: Validate originalIndex
  if (originalIndex < 0 || originalIndex >= N) {
    // If index is invalid, just copy current position (fallback)
    newPos[index] = oldPos[index];
    newVel[index] = oldVel[index];
    return;
  }

  // Copy data from original scattered position to new coherent position
  // memory friendly
  newPos[index] = oldPos[originalIndex];
  newVel[index] = oldVel[originalIndex];
}

/**
* Step the entire N-body simulation by `dt` seconds.
*/
void Boids::stepSimulationNaive(float dt) {
  int effBS = getEffectiveBlockSize();
  dim3 fullBlocksPerGrid((numObjects + effBS - 1) / effBS);

  // Zero out grid-specific metrics (not used by naive)
  g_perfMetrics.kernComputeIndices_ms = 0.0f;
  g_perfMetrics.kernResetBuffer_ms = 0.0f;
  g_perfMetrics.kernIdentifyCellStartEnd_ms = 0.0f;
  g_perfMetrics.thrustSort_ms = 0.0f;
  g_perfMetrics.kernReshuffleData_ms = 0.0f;

  //non-blocking cpu calling,returns immediately
  //send a RECORD_EVENT hardware instruction to the stream.
  cudaEventRecord(perfEvent_start, 0);

  // Measure velocity update kernel
  cudaEventRecord(perfEvent_kernelStart, 0);
  kernUpdateVelocityBruteForce<<<fullBlocksPerGrid, effBS>>>(numObjects, dev_pos, dev_vel1, dev_vel2);
  cudaEventRecord(perfEvent_kernelStop, 0);
  cudaEventSynchronize(perfEvent_kernelStop);
  cudaEventElapsedTime(&g_perfMetrics.kernUpdateVelocity_ms, perfEvent_kernelStart, perfEvent_kernelStop);
  checkCUDAErrorWithLine("kernUpdateVelocityBruteForce failed!");

  // Measure position update kernel
  cudaEventRecord(perfEvent_kernelStart, 0);
  kernUpdatePos<<<fullBlocksPerGrid, effBS>>>(numObjects, dt, dev_pos, dev_vel2);
  cudaEventRecord(perfEvent_kernelStop, 0);
  cudaEventSynchronize(perfEvent_kernelStop);
  cudaEventElapsedTime(&g_perfMetrics.kernUpdatePos_ms, perfEvent_kernelStart, perfEvent_kernelStop);
  checkCUDAErrorWithLine("kernUpdatePos failed!");

  cudaEventRecord(perfEvent_stop, 0);
  //blocking cpu calling,cpu thread will be suspended by os
  //card will erupt a hardware interrupt to CPU,which is handled by driver
  cudaEventSynchronize(perfEvent_stop);
  cudaEventElapsedTime(&g_perfMetrics.totalStepTime_ms, perfEvent_start, perfEvent_stop);

  //swap to update velocity
  glm::vec3 *temp = dev_vel1;
  dev_vel1 = dev_vel2;
  dev_vel2 = temp;
}

void Boids::stepSimulationScatteredGrid(float dt) {
  int effBS = getEffectiveBlockSize();
  dim3 fullBlocksPerGrid((numObjects + effBS - 1) / effBS);
  dim3 gridBlocksPerGrid((gridCellCount + effBS - 1) / effBS);

  // Reset reshuffle metric (not used by scattered)
  g_perfMetrics.kernReshuffleData_ms = 0.0f;

  cudaEventRecord(perfEvent_start, 0);

  // 1. Compute grid indices for each particle
  cudaEventRecord(perfEvent_kernelStart, 0);
  kernComputeIndices<<<fullBlocksPerGrid, effBS>>>(
      numObjects, gridSideCount, gridMinimum, gridInverseCellWidth,
      dev_pos, dev_particleArrayIndices, dev_particleGridIndices);
  cudaEventRecord(perfEvent_kernelStop, 0);
  cudaEventSynchronize(perfEvent_kernelStop);
  cudaEventElapsedTime(&g_perfMetrics.kernComputeIndices_ms, perfEvent_kernelStart, perfEvent_kernelStop);
  checkCUDAErrorWithLine("kernComputeIndices failed!");

  // 2. Sort particles by grid cell using thrust
  cudaEventRecord(perfEvent_kernelStart, 0);
  thrust::sort_by_key(dev_thrust_particleGridIndices, 
                      dev_thrust_particleGridIndices + numObjects,
                      dev_thrust_particleArrayIndices);
  cudaEventRecord(perfEvent_kernelStop, 0);
  cudaEventSynchronize(perfEvent_kernelStop);
  cudaEventElapsedTime(&g_perfMetrics.thrustSort_ms, perfEvent_kernelStart, perfEvent_kernelStop);
  checkCUDAErrorWithLine("thrust::sort_by_key failed!");

  // 3. Reset grid cell start/end indices to sentinel value
  cudaEventRecord(perfEvent_kernelStart, 0);
  kernResetIntBuffer<<<gridBlocksPerGrid, effBS>>>(
      gridCellCount, dev_gridCellStartIndices, -1);
  kernResetIntBuffer<<<gridBlocksPerGrid, effBS>>>(
      gridCellCount, dev_gridCellEndIndices, -1);
  cudaEventRecord(perfEvent_kernelStop, 0);
  cudaEventSynchronize(perfEvent_kernelStop);
  cudaEventElapsedTime(&g_perfMetrics.kernResetBuffer_ms, perfEvent_kernelStart, perfEvent_kernelStop);
  checkCUDAErrorWithLine("kernResetIntBuffer failed!");

  // 4. Identify start and end of each cell in the sorted array
  cudaEventRecord(perfEvent_kernelStart, 0);
  kernIdentifyCellStartEnd<<<fullBlocksPerGrid, effBS>>>(
      numObjects, dev_particleGridIndices,
      dev_gridCellStartIndices, dev_gridCellEndIndices, gridCellCount);
  cudaEventRecord(perfEvent_kernelStop, 0);
  cudaEventSynchronize(perfEvent_kernelStop);
  cudaEventElapsedTime(&g_perfMetrics.kernIdentifyCellStartEnd_ms, perfEvent_kernelStart, perfEvent_kernelStop);
  checkCUDAErrorWithLine("kernIdentifyCellStartEnd failed!");

  // 5. Update velocities using scattered grid neighbor search
  cudaEventRecord(perfEvent_kernelStart, 0);
  kernUpdateVelNeighborSearchScattered<<<fullBlocksPerGrid, effBS>>>(
      numObjects, gridSideCount, gridMinimum,
      gridInverseCellWidth, gridCellWidth,
      dev_gridCellStartIndices, dev_gridCellEndIndices,
      dev_particleArrayIndices,
      dev_pos, dev_vel1, dev_vel2);
  cudaEventRecord(perfEvent_kernelStop, 0);
  cudaEventSynchronize(perfEvent_kernelStop);
  cudaEventElapsedTime(&g_perfMetrics.kernUpdateVelocity_ms, perfEvent_kernelStart, perfEvent_kernelStop);
  checkCUDAErrorWithLine("kernUpdateVelNeighborSearchScattered failed!");

  // 6. Update positions
  cudaEventRecord(perfEvent_kernelStart, 0);
  kernUpdatePos<<<fullBlocksPerGrid, effBS>>>(numObjects, dt, dev_pos, dev_vel2);
  cudaEventRecord(perfEvent_kernelStop, 0);
  cudaEventSynchronize(perfEvent_kernelStop);
  cudaEventElapsedTime(&g_perfMetrics.kernUpdatePos_ms, perfEvent_kernelStart, perfEvent_kernelStop);
  checkCUDAErrorWithLine("kernUpdatePos failed!");

  cudaEventRecord(perfEvent_stop, 0);
  cudaEventSynchronize(perfEvent_stop);
  cudaEventElapsedTime(&g_perfMetrics.totalStepTime_ms, perfEvent_start, perfEvent_stop);

  // 7. Ping-pong velocities
  glm::vec3 *temp = dev_vel1;
  dev_vel1 = dev_vel2;
  dev_vel2 = temp;
}

void Boids::stepSimulationCoherentGrid(float dt) {
  int effBS = getEffectiveBlockSize();
  dim3 fullBlocksPerGrid((numObjects + effBS - 1) / effBS);
  dim3 gridBlocksPerGrid((gridCellCount + effBS - 1) / effBS);

  cudaEventRecord(perfEvent_start, 0);

  // 1. Compute grid indices for each particle (using current dev_pos)
  cudaEventRecord(perfEvent_kernelStart, 0);
  kernComputeIndices<<<fullBlocksPerGrid, effBS>>>(
      numObjects, gridSideCount, gridMinimum, gridInverseCellWidth,
      dev_pos, dev_particleArrayIndices, dev_particleGridIndices);
  cudaEventRecord(perfEvent_kernelStop, 0);
  cudaEventSynchronize(perfEvent_kernelStop);
  cudaEventElapsedTime(&g_perfMetrics.kernComputeIndices_ms, perfEvent_kernelStart, perfEvent_kernelStop);
  checkCUDAErrorWithLine("kernComputeIndices failed!");

  // 2. Sort particles by grid cell using thrust
  cudaEventRecord(perfEvent_kernelStart, 0);
  thrust::sort_by_key(dev_thrust_particleGridIndices, 
                      dev_thrust_particleGridIndices + numObjects,
                      dev_thrust_particleArrayIndices);
  cudaEventRecord(perfEvent_kernelStop, 0);
  cudaEventSynchronize(perfEvent_kernelStop);
  cudaEventElapsedTime(&g_perfMetrics.thrustSort_ms, perfEvent_kernelStart, perfEvent_kernelStop);
  checkCUDAErrorWithLine("thrust::sort_by_key failed!");

  // 3. Reset grid cell start/end indices
  cudaEventRecord(perfEvent_kernelStart, 0);
  kernResetIntBuffer<<<gridBlocksPerGrid, effBS>>>(
      gridCellCount, dev_gridCellStartIndices, -1);
  kernResetIntBuffer<<<gridBlocksPerGrid, effBS>>>(
      gridCellCount, dev_gridCellEndIndices, -1);
  cudaEventRecord(perfEvent_kernelStop, 0);
  cudaEventSynchronize(perfEvent_kernelStop);
  cudaEventElapsedTime(&g_perfMetrics.kernResetBuffer_ms, perfEvent_kernelStart, perfEvent_kernelStop);
  checkCUDAErrorWithLine("kernResetIntBuffer failed!");

  // 4. Identify start and end of each cell
  cudaEventRecord(perfEvent_kernelStart, 0);
  kernIdentifyCellStartEnd<<<fullBlocksPerGrid, effBS>>>(
      numObjects, dev_particleGridIndices,
      dev_gridCellStartIndices, dev_gridCellEndIndices, gridCellCount);
  cudaEventRecord(perfEvent_kernelStop, 0);
  cudaEventSynchronize(perfEvent_kernelStop);
  cudaEventElapsedTime(&g_perfMetrics.kernIdentifyCellStartEnd_ms, perfEvent_kernelStart, perfEvent_kernelStop);
  checkCUDAErrorWithLine("kernIdentifyCellStartEnd failed!");

  // 5. Reshuffle boid data to be coherent with sorted order
  cudaEventRecord(perfEvent_kernelStart, 0);
  kernReshuffleData<<<fullBlocksPerGrid, effBS>>>(
      numObjects, dev_particleArrayIndices,
      dev_pos, dev_vel1,
      dev_coherentPos, dev_coherentVel1);
  cudaEventRecord(perfEvent_kernelStop, 0);
  cudaEventSynchronize(perfEvent_kernelStop);
  cudaEventElapsedTime(&g_perfMetrics.kernReshuffleData_ms, perfEvent_kernelStart, perfEvent_kernelStop);
  checkCUDAErrorWithLine("kernReshuffleData failed!");

  // 6. Update velocities using coherent grid neighbor search
  cudaEventRecord(perfEvent_kernelStart, 0);
  kernUpdateVelNeighborSearchCoherent<<<fullBlocksPerGrid, effBS>>>(
      numObjects, gridSideCount, gridMinimum,
      gridInverseCellWidth, gridCellWidth,
      dev_gridCellStartIndices, dev_gridCellEndIndices,
      dev_coherentPos, dev_coherentVel1, dev_coherentVel2);
  cudaEventRecord(perfEvent_kernelStop, 0);
  cudaEventSynchronize(perfEvent_kernelStop);
  cudaEventElapsedTime(&g_perfMetrics.kernUpdateVelocity_ms, perfEvent_kernelStart, perfEvent_kernelStop);
  checkCUDAErrorWithLine("kernUpdateVelNeighborSearchCoherent failed!");

  // 7. Update positions using coherent data
  cudaEventRecord(perfEvent_kernelStart, 0);
  kernUpdatePos<<<fullBlocksPerGrid, effBS>>>(numObjects, dt, dev_coherentPos, dev_coherentVel2);
  cudaEventRecord(perfEvent_kernelStop, 0);
  cudaEventSynchronize(perfEvent_kernelStop);
  cudaEventElapsedTime(&g_perfMetrics.kernUpdatePos_ms, perfEvent_kernelStart, perfEvent_kernelStop);
  checkCUDAErrorWithLine("kernUpdatePos failed!");

  // 8. Copy coherent data back to main buffers (for rendering & next frame)
  //    This is included in totalStepTime as it is part of the full pipeline cost.
  cudaMemcpy(dev_pos, dev_coherentPos, numObjects * sizeof(glm::vec3), cudaMemcpyDeviceToDevice);
  cudaMemcpy(dev_vel1, dev_coherentVel2, numObjects * sizeof(glm::vec3), cudaMemcpyDeviceToDevice);
  checkCUDAErrorWithLine("cudaMemcpy coherent to main failed!");

  cudaEventRecord(perfEvent_stop, 0);
  cudaEventSynchronize(perfEvent_stop);
  cudaEventElapsedTime(&g_perfMetrics.totalStepTime_ms, perfEvent_start, perfEvent_stop);

  // 9. Ping-pong coherent velocities for next iteration
  glm::vec3 *temp = dev_coherentVel1;
  dev_coherentVel1 = dev_coherentVel2;
  dev_coherentVel2 = temp;
}

void Boids::endSimulation() {
  cudaFree(dev_vel1);
  cudaFree(dev_vel2);
  cudaFree(dev_pos);

  // TODO-2.1 TODO-2.3 - Free any additional buffers here.
  cudaFree(dev_particleArrayIndices);
  cudaFree(dev_particleGridIndices);
  cudaFree(dev_gridCellStartIndices);
  cudaFree(dev_gridCellEndIndices);
  
  // Free coherent buffers
  cudaFree(dev_coherentPos);
  cudaFree(dev_coherentVel1);
  cudaFree(dev_coherentVel2);
  
  // Destroy CUDA Events
  cudaEventDestroy(perfEvent_start);
  cudaEventDestroy(perfEvent_stop);
  cudaEventDestroy(perfEvent_kernelStart);
  cudaEventDestroy(perfEvent_kernelStop);
}

void Boids::resetPerformanceMetrics() {
  memset(&g_perfMetrics, 0, sizeof(Boids::PerformanceMetrics));
}

Boids::PerformanceMetrics Boids::getPerformanceMetrics() {
  return g_perfMetrics;
}

int Boids::getBlockSize() {
  return getEffectiveBlockSize();
}

void Boids::stepSimulation(SimulationMethod method, float dt) {
  switch (method) {
    case SimulationMethod::NAIVE:
      stepSimulationNaive(dt);
      break;
    case SimulationMethod::SCATTERED_GRID:
      stepSimulationScatteredGrid(dt);
      break;
    case SimulationMethod::COHERENT_GRID:
      stepSimulationCoherentGrid(dt);
      break;
  }
}

void Boids::unitTest() {
  // LOOK-1.2 Feel free to write additional tests here.

  // test unstable sort
  int *dev_intKeys;
  int *dev_intValues;
  int N = 10;

  std::unique_ptr<int[]>intKeys{ new int[N] };
  std::unique_ptr<int[]>intValues{ new int[N] };

  intKeys[0] = 0; intValues[0] = 0;
  intKeys[1] = 1; intValues[1] = 1;
  intKeys[2] = 0; intValues[2] = 2;
  intKeys[3] = 3; intValues[3] = 3;
  intKeys[4] = 0; intValues[4] = 4;
  intKeys[5] = 2; intValues[5] = 5;
  intKeys[6] = 2; intValues[6] = 6;
  intKeys[7] = 0; intValues[7] = 7;
  intKeys[8] = 5; intValues[8] = 8;
  intKeys[9] = 6; intValues[9] = 9;

  cudaMalloc((void**)&dev_intKeys, N * sizeof(int));
  checkCUDAErrorWithLine("cudaMalloc dev_intKeys failed!");

  cudaMalloc((void**)&dev_intValues, N * sizeof(int));
  checkCUDAErrorWithLine("cudaMalloc dev_intValues failed!");

  dim3 fullBlocksPerGrid((N + blockSize - 1) / blockSize);

  std::cout << "before unstable sort: " << std::endl;
  for (int i = 0; i < N; i++) {
    std::cout << "  key: " << intKeys[i];
    std::cout << " value: " << intValues[i] << std::endl;
  }

  // How to copy data to the GPU
  cudaMemcpy(dev_intKeys, intKeys.get(), sizeof(int) * N, cudaMemcpyHostToDevice);
  cudaMemcpy(dev_intValues, intValues.get(), sizeof(int) * N, cudaMemcpyHostToDevice);

  // Wrap device vectors in thrust iterators for use with thrust.
  thrust::device_ptr<int> dev_thrust_keys(dev_intKeys);
  thrust::device_ptr<int> dev_thrust_values(dev_intValues);
  // LOOK-2.1 Example for using thrust::sort_by_key
  thrust::sort_by_key(dev_thrust_keys, dev_thrust_keys + N, dev_thrust_values);

  // How to copy data back to the CPU side from the GPU
  cudaMemcpy(intKeys.get(), dev_intKeys, sizeof(int) * N, cudaMemcpyDeviceToHost);
  cudaMemcpy(intValues.get(), dev_intValues, sizeof(int) * N, cudaMemcpyDeviceToHost);
  checkCUDAErrorWithLine("memcpy back failed!");

  std::cout << "after unstable sort: " << std::endl;
  for (int i = 0; i < N; i++) {
    std::cout << "  key: " << intKeys[i];
    std::cout << " value: " << intValues[i] << std::endl;
  }

  // cleanup
  cudaFree(dev_intKeys);
  cudaFree(dev_intValues);
  checkCUDAErrorWithLine("cudaFree failed!");
  return;
}
