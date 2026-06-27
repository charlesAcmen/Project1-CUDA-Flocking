#pragma once

namespace Boids {
    // Primary init (uses compile-time blockSize=128)
    void initSimulation(int N);
    // Parameterized init - allows runtime block size override
    void initSimulation(int N, int blockSz);

    void stepSimulationNaive(float dt);
    void stepSimulationScatteredGrid(float dt);
    void stepSimulationCoherentGrid(float dt);
    void copyBoidsToVBO(float *vbodptr_positions, float *vbodptr_velocities);

    void endSimulation();
    void unitTest();

    // Performance measurement interface
    struct PerformanceMetrics {
        float kernUpdateVelocity_ms;
        float kernUpdatePos_ms;
        float kernComputeIndices_ms;
        float kernResetBuffer_ms;
        float kernIdentifyCellStartEnd_ms;
        float thrustSort_ms;
        float kernReshuffleData_ms;
        float totalStepTime_ms;
    };

    void resetPerformanceMetrics();
    PerformanceMetrics getPerformanceMetrics();
    int getBlockSize();

    // Simulation method enum for unified dispatch
    enum class SimulationMethod { NAIVE, SCATTERED_GRID, COHERENT_GRID };
    void stepSimulation(SimulationMethod method, float dt);
}
