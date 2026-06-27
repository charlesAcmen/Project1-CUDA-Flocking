/**
* @file      main_parameterized.cpp
* @brief     Parameterized Boids flocking simulation for performance testing
* @authors   Modified for automated benchmarking
* @date      2026
*/

#include "main.hpp"
#include "kernel.h"

#include <iostream>
#include <memory>
#include <sstream>
#include <fstream>
#include <iomanip>
#include <string>
#include <cstring>
#include <chrono>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <glm/gtc/matrix_transform.hpp>

// ================
// Configuration
// ================

// LOOK-2.1 LOOK-2.3 - toggles for UNIFORM_GRID and COHERENT_GRID
// These are now defaults, can be overridden by command line
#ifndef VISUALIZE
#define VISUALIZE 1
#endif
#ifndef UNIFORM_GRID
#define UNIFORM_GRID 0
#endif
#ifndef COHERENT_GRID
#define COHERENT_GRID 0
#endif

// LOOK-1.2 - change this to adjust particle count in the simulation
#ifndef N_FOR_VIS
#define N_FOR_VIS 5000
#endif

const float DT = 0.2f;

// Runtime configuration structure
struct RuntimeConfig {
    int numBoids = N_FOR_VIS;
    int blockSize = 128;  // Default block size
    int method = 0;       // 0=naive, 1=scattered, 2=coherent
    bool visualize = (VISUALIZE != 0);
    int maxFrames = 0;    // 0 = unlimited
    std::string outputFile = "";
    
    void parseCommandLine(int argc, char** argv) {
        for (int i = 1; i < argc; i++) {
            std::string arg = argv[i];
            
            if (arg == "-n" || arg == "--nboids") {
                if (i + 1 < argc) numBoids = std::atoi(argv[++i]);
            }
            else if (arg == "-b" || arg == "--blocksize") {
                if (i + 1 < argc) blockSize = std::atoi(argv[++i]);
            }
            else if (arg == "-m" || arg == "--method") {
                if (i + 1 < argc) {
                    std::string methodStr = argv[++i];
                    if (methodStr == "naive" || methodStr == "0") method = 0;
                    else if (methodStr == "scattered" || methodStr == "1") method = 1;
                    else if (methodStr == "coherent" || methodStr == "2") method = 2;
                }
            }
            else if (arg == "-v" || arg == "--visualize") {
                visualize = true;
            }
            else if (arg == "--no-vis" || arg == "--novis") {
                visualize = false;
            }
            else if (arg == "-f" || arg == "--frames") {
                if (i + 1 < argc) maxFrames = std::atoi(argv[++i]);
            }
            else if (arg == "-o" || arg == "--output") {
                if (i + 1 < argc) outputFile = argv[++i];
            }
            else if (arg == "-h" || arg == "--help") {
                printHelp();
                exit(0);
            }
        }
        
        // Auto-generate output filename if not specified
        if (outputFile.empty()) {
            std::ostringstream oss;
            oss << "perf_";
            if (method == 0) oss << "naive";
            else if (method == 1) oss << "scattered";
            else oss << "coherent";
            oss << "_N" << numBoids;
            oss << "_B" << blockSize;
            oss << "_vis" << (visualize ? 1 : 0);
            oss << ".csv";
            outputFile = oss.str();
        }
    }
    
    void printHelp() {
        std::cout << "CUDA Boids Flocking Simulation - Parameterized Version\n\n";
        std::cout << "Usage: cis5650_boids [options]\n\n";
        std::cout << "Options:\n";
        std::cout << "  -n, --nboids <N>      Number of boids (default: " << N_FOR_VIS << ")\n";
        std::cout << "  -b, --blocksize <B>   CUDA block size (default: 128)\n";
        std::cout << "  -m, --method <M>      Simulation method:\n";
        std::cout << "                          0/naive - Naive O(N^2) method\n";
        std::cout << "                          1/scattered - Scattered uniform grid\n";
        std::cout << "                          2/coherent - Coherent uniform grid\n";
        std::cout << "  -v, --visualize       Enable visualization (default)\n";
        std::cout << "  --no-vis, --novis     Disable visualization for pure performance\n";
        std::cout << "  -f, --frames <F>      Max frames to run (0 = unlimited)\n";
        std::cout << "  -o, --output <file>   Output CSV filename\n";
        std::cout << "  -h, --help            Show this help\n\n";
        std::cout << "Examples:\n";
        std::cout << "  cis5650_boids -n 10000 -m naive --no-vis -f 1000\n";
        std::cout << "  cis5650_boids -n 5000 -b 256 -m scattered -v\n";
        std::cout << "  cis5650_boids -n 20000 -m coherent --no-vis -o results.csv\n";
    }
    
    void print() const {
        std::cout << "Configuration:\n";
        std::cout << "  Number of boids: " << numBoids << "\n";
        std::cout << "  Block size: " << blockSize << "\n";
        std::cout << "  Method: ";
        if (method == 0) std::cout << "Naive\n";
        else if (method == 1) std::cout << "Scattered Grid\n";
        else std::cout << "Coherent Grid\n";
        std::cout << "  Visualization: " << (visualize ? "ON" : "OFF") << "\n";
        std::cout << "  Max frames: " << (maxFrames > 0 ? std::to_string(maxFrames) : "unlimited") << "\n";
        std::cout << "  Output file: " << outputFile << "\n";
    }
};

RuntimeConfig g_config;
std::ofstream g_perfCSV;
cudaEvent_t g_frameStart;
cudaEvent_t g_frameStop;
float g_lastFrameTime_ms = 0.0f;

/**
* C main function.
*/
int main(int argc, char* argv[]) {
    projectName = "5650 CUDA Intro: Boids (Parameterized)";
    
    g_config.parseCommandLine(argc, argv);
    std::cout << "========================================\n";
    g_config.print();
    std::cout << "========================================\n\n";
    
    if (init(argc, argv)) {
        mainLoop();
        Boids::endSimulation();
        if (g_perfCSV.is_open()) {
            g_perfCSV.close();
        }
        if (g_frameStart) {
            cudaEventDestroy(g_frameStart);
            cudaEventDestroy(g_frameStop);
        }
        return 0;
    } else {
        return 1;
    }
}

//-------------------------------
//---------RUNTIME STUFF---------
//-------------------------------

std::string deviceName;
GLFWwindow *window;

/**
* Initialization of CUDA and GLFW.
*/
bool init(int argc, char **argv) {
    // Set window title to "Student Name: [SM 2.0] GPU Name"
    cudaDeviceProp deviceProp;
    int gpuDevice = 0;
    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    if (gpuDevice > device_count) {
        std::cout
        << "Error: GPU device number is greater than the number of devices!"
        << " Perhaps a CUDA-capable GPU is not installed?"
        << std::endl;
        return false;
    }
    cudaGetDeviceProperties(&deviceProp, gpuDevice);
    int major = deviceProp.major;
    int minor = deviceProp.minor;

    std::ostringstream ss;
    ss << projectName << " [SM " << major << "." << minor << " " << deviceProp.name << "]";
    deviceName = ss.str();

    // Window setup stuff
    if (g_config.visualize) {
        glfwSetErrorCallback(errorCallback);

        if (!glfwInit()) {
            std::cout
            << "Error: Could not initialize GLFW!"
            << " Perhaps OpenGL 3.3 isn't available?"
            << std::endl;
            return false;
        }

        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
        glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
        glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

        window = glfwCreateWindow(width, height, deviceName.c_str(), NULL, NULL);
        if (!window) {
            glfwTerminate();
            return false;
        }
        glfwMakeContextCurrent(window);
        glfwSetKeyCallback(window, keyCallback);
        glfwSetCursorPosCallback(window, mousePositionCallback);
        glfwSetMouseButtonCallback(window, mouseButtonCallback);

        glewExperimental = GL_TRUE;
        if (glewInit() != GLEW_OK) {
            return false;
        }

        // Default to device ID 0. If you have more than one GPU and want to test a non-default one,
        // change the device ID.
        cudaGLSetGLDevice(0);

        // Initialize drawing state
        initVAO();

        cudaGLRegisterBufferObject(boidVBO_positions);
        cudaGLRegisterBufferObject(boidVBO_velocities);
    }

    // Initialize N-body simulation with custom block size
    Boids::initSimulation(g_config.numBoids, g_config.blockSize);

    // Create CUDA events for frame timing
    cudaEventCreate(&g_frameStart);
    cudaEventCreate(&g_frameStop);

    // Open CSV file for performance logging
    g_perfCSV.open(g_config.outputFile, std::ios::out | std::ios::trunc);
    if (g_perfCSV.is_open()) {
        g_perfCSV << "frame,fps,frame_ms,";
        g_perfCSV << "kern_update_velocity_ms,kern_update_pos_ms,";
        g_perfCSV << "kern_compute_indices_ms,kern_reset_buffer_ms,";
        g_perfCSV << "kern_identify_cell_ms,thrust_sort_ms,";
        g_perfCSV << "kern_reshuffle_ms,total_step_ms,";
        g_perfCSV << "method,num_boids,block_size,visualize\n";
        g_perfCSV << std::fixed << std::setprecision(4);
    }

    if (g_config.visualize) {
        updateCamera();
        initShaders(program);
        glEnable(GL_DEPTH_TEST);
    }

    return true;
}

void initVAO() {
    std::unique_ptr<GLfloat[]> bodies{ new GLfloat[4 * g_config.numBoids] };
    std::unique_ptr<GLuint[]> bindices{ new GLuint[g_config.numBoids] };

    glm::vec4 ul(-1.0, -1.0, 1.0, 1.0);
    glm::vec4 lr(1.0, 1.0, 0.0, 0.0);

    for (int i = 0; i < g_config.numBoids; i++) {
        bodies[4 * i + 0] = 0.0f;
        bodies[4 * i + 1] = 0.0f;
        bodies[4 * i + 2] = 0.0f;
        bodies[4 * i + 3] = 1.0f;
        bindices[i] = i;
    }

    glGenVertexArrays(1, &boidVAO);
    glGenBuffers(1, &boidVBO_positions);
    glGenBuffers(1, &boidVBO_velocities);
    glGenBuffers(1, &boidIBO);

    glBindVertexArray(boidVAO);

    glBindBuffer(GL_ARRAY_BUFFER, boidVBO_positions);
    glBufferData(GL_ARRAY_BUFFER, 4 * g_config.numBoids * sizeof(GLfloat), bodies.get(), GL_DYNAMIC_DRAW);
    glEnableVertexAttribArray(positionLocation);
    glVertexAttribPointer((GLuint)positionLocation, 4, GL_FLOAT, GL_FALSE, 0, 0);

    glBindBuffer(GL_ARRAY_BUFFER, boidVBO_velocities);
    glBufferData(GL_ARRAY_BUFFER, 4 * g_config.numBoids * sizeof(GLfloat), bodies.get(), GL_DYNAMIC_DRAW);
    glEnableVertexAttribArray(velocitiesLocation);
    glVertexAttribPointer((GLuint)velocitiesLocation, 4, GL_FLOAT, GL_FALSE, 0, 0);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, boidIBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, g_config.numBoids * sizeof(GLuint), bindices.get(), GL_STATIC_DRAW);

    glBindVertexArray(0);
}

void initShaders(GLuint * program) {
    GLint location;

    program[PROG_BOID] = glslUtility::createProgram(
        "shaders/boid.vert.glsl",
        "shaders/boid.geom.glsl",
        "shaders/boid.frag.glsl", attributeLocations, 2);
    glUseProgram(program[PROG_BOID]);

    if ((location = glGetUniformLocation(program[PROG_BOID], "u_projMatrix")) != -1) {
        glUniformMatrix4fv(location, 1, GL_FALSE, &projection[0][0]);
    }
    if ((location = glGetUniformLocation(program[PROG_BOID], "u_cameraPos")) != -1) {
        glUniform3fv(location, 1, &cameraPosition[0]);
    }
}

//====================================
// Main loop
//====================================
void runCUDA() {
    float *dptrVertPositions = NULL;
    float *dptrVertVelocities = NULL;

    if (g_config.visualize) {
        cudaGLMapBufferObject((void**)&dptrVertPositions, boidVBO_positions);
        cudaGLMapBufferObject((void**)&dptrVertVelocities, boidVBO_velocities);
    }

    // Record frame start time
    cudaEventRecord(g_frameStart, 0);

    // Execute the kernel based on method
    if (g_config.method == 0) {
        Boids::stepSimulationNaive(DT);
    } else if (g_config.method == 1) {
        Boids::stepSimulationScatteredGrid(DT);
    } else {
        Boids::stepSimulationCoherentGrid(DT);
    }

    if (g_config.visualize) {
        Boids::copyBoidsToVBO(dptrVertPositions, dptrVertVelocities);
    }

    // Record frame end time
    cudaEventRecord(g_frameStop, 0);
    cudaEventSynchronize(g_frameStop);
    cudaEventElapsedTime(&g_lastFrameTime_ms, g_frameStart, g_frameStop);

    if (g_config.visualize) {
        cudaGLUnmapBufferObject(boidVBO_positions);
        cudaGLUnmapBufferObject(boidVBO_velocities);
    }
}

void mainLoop() {
    double fps = 0;
    double timebase = 0;
    int frame = 0;
    int totalFrames = 0;

    Boids::unitTest();

    bool running = true;
    double lastTime = glfwGetTime();

    while (running) {
        if (g_config.visualize) {
            glfwPollEvents();
            if (glfwWindowShouldClose(window)) break;
        }

        frame++;
        totalFrames++;
        double time;
        if (g_config.visualize) {
            time = glfwGetTime();
        } else {
            // Use chrono for headless mode (GLFW not initialized)
            static auto startPoint = std::chrono::steady_clock::now();
            auto now = std::chrono::steady_clock::now();
            time = std::chrono::duration<double>(now - startPoint).count();
        }

        if (time - timebase > 1.0) {
            fps = frame / (time - timebase);
            timebase = time;
            frame = 0;
        }

        runCUDA();

        // Get detailed performance metrics from kernel
        Boids::PerformanceMetrics metrics = Boids::getPerformanceMetrics();

        // Write performance data to CSV
        if (g_perfCSV.is_open()) {
            std::string methodStr = (g_config.method == 0) ? "naive" : 
                                   (g_config.method == 1) ? "scattered" : "coherent";
            g_perfCSV << totalFrames << ","
                      << fps << ","
                      << g_lastFrameTime_ms << ","
                      << metrics.kernUpdateVelocity_ms << ","
                      << metrics.kernUpdatePos_ms << ","
                      << metrics.kernComputeIndices_ms << ","
                      << metrics.kernResetBuffer_ms << ","
                      << metrics.kernIdentifyCellStartEnd_ms << ","
                      << metrics.thrustSort_ms << ","
                      << metrics.kernReshuffleData_ms << ","
                      << metrics.totalStepTime_ms << ","
                      << methodStr << ","
                      << g_config.numBoids << ","
                      << g_config.blockSize << ","
                      << (g_config.visualize ? 1 : 0) << "\n";
        }

        if (g_config.visualize) {
            std::ostringstream ss;
            ss << "[";
            ss.precision(1);
            ss << std::fixed << fps;
            ss << " fps | ";
            ss.precision(3);
            ss << "frame: " << g_lastFrameTime_ms << "ms";
            ss << " | N=" << g_config.numBoids;
            ss << " | B=" << g_config.blockSize;
            ss << "] " << deviceName;
            glfwSetWindowTitle(window, ss.str().c_str());

            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

            glUseProgram(program[PROG_BOID]);
            glBindVertexArray(boidVAO);
            glPointSize((GLfloat)pointSize);
            glDrawElements(GL_POINTS, g_config.numBoids, GL_UNSIGNED_INT, 0);
            glPointSize(1.0f);

            glUseProgram(0);
            glBindVertexArray(0);

            glfwSwapBuffers(window);
        } else {
            // No visualization mode - print progress periodically
            if (totalFrames % 100 == 0) {
                std::cout << "Frame " << totalFrames << " | " 
                         << std::fixed << std::setprecision(2) << fps << " FPS | "
                         << std::setprecision(3) << g_lastFrameTime_ms << " ms\n";
            }
        }

        // Check max frames limit
        if (g_config.maxFrames > 0 && totalFrames >= g_config.maxFrames) {
            std::cout << "\nReached max frames limit: " << g_config.maxFrames << "\n";
            running = false;
        }
    }

    // Cleanup
    if (g_perfCSV.is_open()) {
        g_perfCSV.close();
        std::cout << "\nPerformance data saved to: " << g_config.outputFile << "\n";
    }
    if (g_frameStart) {
        cudaEventDestroy(g_frameStart);
        cudaEventDestroy(g_frameStop);
    }

    if (g_config.visualize) {
        glfwDestroyWindow(window);
        glfwTerminate();
    }
}

void errorCallback(int error, const char *description) {
    fprintf(stderr, "error %d: %s\n", error, description);
}

void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods) {
    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) {
        glfwSetWindowShouldClose(window, GL_TRUE);
    }
}

void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods) {
    leftMousePressed = (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS);
    rightMousePressed = (button == GLFW_MOUSE_BUTTON_RIGHT && action == GLFW_PRESS);
}

void mousePositionCallback(GLFWwindow* window, double xpos, double ypos) {
    if (leftMousePressed) {
        phi += (xpos - lastX) / width;
        theta -= (ypos - lastY) / height;
        theta = std::fmax(0.01f, std::fmin(theta, 3.14f));
        updateCamera();
    }
    else if (rightMousePressed) {
        zoom += (ypos - lastY) / height;
        zoom = std::fmax(0.1f, std::fmin(zoom, 5.0f));
        updateCamera();
    }

    lastX = xpos;
    lastY = ypos;
}

void updateCamera() {
    cameraPosition.x = zoom * sin(phi) * sin(theta);
    cameraPosition.z = zoom * cos(theta);
    cameraPosition.y = zoom * cos(phi) * sin(theta);
    cameraPosition += lookAt;

    projection = glm::perspective(fovy, float(width) / float(height), zNear, zFar);
    glm::mat4 view = glm::lookAt(cameraPosition, lookAt, glm::vec3(0, 0, 1));
    projection = projection * view;

    GLint location;

    glUseProgram(program[PROG_BOID]);
    if ((location = glGetUniformLocation(program[PROG_BOID], "u_projMatrix")) != -1) {
        glUniformMatrix4fv(location, 1, GL_FALSE, &projection[0][0]);
    }
}
