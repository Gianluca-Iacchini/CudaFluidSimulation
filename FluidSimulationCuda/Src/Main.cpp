#include <iostream>
#include "test.cuh"
#include "cpuFluidSim.h"
#include <string>

#define GPU_SIM 0


// settings
#if GPU_SIM
const unsigned int SCR_WIDTH = 1920;
const unsigned int SCR_HEIGHT = 1080;
const int SCALE = 2;
#else
const unsigned int SCR_WIDTH = 1024;
const unsigned int SCR_HEIGHT = 1024;
const int SCALE = 8;
#endif

float vertices[] = {
	-1.0f, -1.0f, 1.0f, -1.0f,
	-1.0f, 1.0f, 1.0f, 1.0f
};

unsigned int indices[] = {  // note that we start from 0!
	0, 1, 2,  // first Triangle
	1, 3, 2   // second Triangle
};

unsigned int VBO, VAO, EBO;
unsigned int texture;

bool isPressed = false;
bool firstClick = true;
double xPos = 0.0f;
double yPos = 0.0f;
double lastXPos = 0.0f;
double lastYPos = 0.0f;

int main()
{
#if GPU_SIM
	cudaInit(SCR_WIDTH, SCR_HEIGHT, SCALE, texture);
#else
	init(SCR_WIDTH, SCR_HEIGHT, SCALE);
#endif // GPU_SIM

	float deltaTime = 0.f;
	float lastTime = 0.0f;

	int nbFrames = 0;
	double lasttFrameTime;

//	while (!glfwWindowShouldClose(window))
//	{
//		deltaTime = glfwGetTime() - lastTime;
//		lastTime = glfwGetTime();
//
//		
//
//			// Measure speed
//		double currentTime = glfwGetTime();
//		nbFrames++;
//		if (currentTime - lasttFrameTime >= 1.0) { // If last prinf() was more than 1 sec ago
//				// printf and reset timer
//			std::string frames = "FPS: " + std::to_string(nbFrames);
//			glfwSetWindowTitle(window, frames.c_str());
//			nbFrames = 0;
//			lasttFrameTime += 1.0;
//		}
//
//
//		processInput(window);
//
//#if GPU_SIM
//		computeField(deltaTime, xPos/SCALE, (SCR_HEIGHT - yPos)/SCALE, lastXPos/SCALE, (SCR_HEIGHT - (lastYPos))/SCALE, isPressed);
//#else
//		//on_frame(texture, deltaTime, isPressed);
//#endif
//		glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
//		glClear(GL_COLOR_BUFFER_BIT);
//
//		shader.Use();
//		shader.SetInt("dyeTexture", 0);
//
//		glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
//
//		GLenum err;
//		while ((err = glGetError()) != GL_NO_ERROR)
//		{
//			std::cout << err << std::endl;
//		}
//		glfwSwapBuffers(window);
//		glfwPollEvents();
//
//	}

	uint32_t* data = (uint32_t*)malloc((SCR_WIDTH / SCALE) * (SCR_HEIGHT / SCALE) * sizeof(uint32_t));
	on_frame(data, deltaTime, isPressed);
	std::cout.write(reinterpret_cast<char*>(data), (SCR_WIDTH / SCALE)* (SCR_HEIGHT / SCALE) * 4);


	cudaExit();
	return 0;
}

