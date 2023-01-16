#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <Shaders/Shader.h>
#include <iostream>
#include "GPUFluidSim.cuh"
#include "CPUFluidSim.h"
#include <string>

#define GPU_SIM 1

void framebuffer_size_callback(GLFWwindow* window, int width, int height);
void processInput(GLFWwindow* window);

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

/* OpenGL data*/
float vertices[] = {
	-1.0f, -1.0f, 1.0f, -1.0f,
	-1.0f, 1.0f, 1.0f, 1.0f
};

unsigned int indices[] = { 
	0, 1, 2,  // first Triangle
	1, 3, 2   // second Triangle
};

unsigned int VBO, VAO, EBO;
unsigned int texture;

/* Mouse input data */
bool isPressed = false;
bool firstClick = true;
double xPos = 0.0f;
double yPos = 0.0f;
double lastXPos = 0.0f;
double lastYPos = 0.0f;

/* Timer data */
double computeTimeStart = 0.0f;
double computeTimeEnd = 0.0f;
uint64_t totalFrames = 0;
double minComputeTime = 999;
double maxComputeTime = 0.0f;
double averageComputeTime = 0.0f;

/* Rounds float and converts it to string */
std::string roundedFloatToString(float val)
{ 
	float nearest = roundf(val * 100) / 100;
	std::string valStr = std::to_string(nearest);
	return valStr.substr(0, valStr.find(".") + 3);
}

int main()
{
	/* Initialize GLFW */
	glfwInit();
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);


	GLFWwindow* window = glfwCreateWindow(SCR_WIDTH, SCR_HEIGHT, "LearnOpenGL", NULL, NULL);
	if (window == NULL)
	{
		std::cout << "Failed to create GLFW window" << std::endl;
		glfwTerminate();
		return -1;
	}
	glfwMakeContextCurrent(window);
	glfwSwapInterval(0);
	glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);

	/* Initialize GLAD */
	if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
	{
		std::cout << "Failed to initialize GLAD" << std::endl;
		return -1;
	}

	/* Create simple shader to display a quad with a texture */
	Shader shader = Shader("Programs/test_shader.vert", "Programs/test_shader.frag");

	/* Generate vertex */
	glGenVertexArrays(1, &VAO);
	glGenBuffers(1, &VBO);
	glGenBuffers(1, &EBO);

	glBindVertexArray(VAO);

	/* Bind quad */
	glBindBuffer(GL_ARRAY_BUFFER, VBO);
	glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);


	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);

	glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), (void*)0);
	glEnableVertexAttribArray(0);

	/* Generate and setup texture */
	glGenTextures(1, &texture);
	glBindTexture(GL_TEXTURE_2D, texture);

	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, SCR_WIDTH / SCALE, SCR_HEIGHT / SCALE, 0, GL_RGBA, GL_UNSIGNED_BYTE, 0);

#if GPU_SIM
	g_fluidSimInit(SCR_WIDTH, SCR_HEIGHT, SCALE, texture);
#else
	c_fluidSimInit(SCR_WIDTH, SCR_HEIGHT, SCALE, texture);
#endif // GPU_SIM

	float deltaTime = 0.f;
	float lastTime = 0.0f;

	int nbFrames = 0;
	double lasttFrameTime = glfwGetTime();

	while (!glfwWindowShouldClose(window))
	{
		deltaTime = glfwGetTime() - lastTime;
		lastTime = glfwGetTime();

		

		// FPS and frame time counters.
		double currentTime = glfwGetTime();
		nbFrames++;
		if (currentTime - lasttFrameTime >= 0.5) { 
			std::string frames = "FPS: " + std::to_string(nbFrames) + "     FRAME TIME: " + roundedFloatToString(1000.f / nbFrames) + "ms";
			glfwSetWindowTitle(window, frames.c_str());
			nbFrames = 0;
			lasttFrameTime += 1.0;
		}


		processInput(window);
		computeTimeStart = glfwGetTime();

#if GPU_SIM
		g_OnSimulationStep(deltaTime, xPos/SCALE, (SCR_HEIGHT - yPos)/SCALE, lastXPos/SCALE, (SCR_HEIGHT - (lastYPos))/SCALE, isPressed);
#else
		c_OnSimulationStep(deltaTime, xPos, yPos, isPressed);
#endif
		computeTimeEnd = glfwGetTime() - computeTimeStart;

		averageComputeTime += computeTimeEnd;
		minComputeTime = std::min(minComputeTime, computeTimeEnd);
		maxComputeTime = std::max(maxComputeTime, computeTimeEnd);
		totalFrames++;

		glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
		glClear(GL_COLOR_BUFFER_BIT);

		shader.Use();
		shader.SetInt("dyeTexture", 0);

		glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);

		GLenum err;
		while ((err = glGetError()) != GL_NO_ERROR)
		{
			std::cout << err << std::endl;
		}
		glfwSwapBuffers(window);
		glfwPollEvents();

	}
	std::string steps[] = { "Advect", "Vorticity", "Diffuse", "Force", "Pressure", "Project", "Paint", "Bloom" };
	
	double* stepTimes = c_getAverageTimes();
	int maxIter = 6;

#if GPU_SIM
	stepTimes = g_getAverageTimes();
	maxIter = 8;
	g_fluidSimFree();
#else
	c_fluidSimFree();
#endif // GPU_SIM
	std::cout << "================ SIMULATION END ======================" << std::endl;
	std::cout << "Total application time: " << roundedFloatToString(glfwGetTime()) << " seconds" << std::endl;
	std::cout << "------------------------------------------------------" << std::endl;
	std::cout << "Average compute time: " << roundedFloatToString(averageComputeTime * 1000 / totalFrames) << "ms" << std::endl;
	std::cout << "Max compute time: " << roundedFloatToString(maxComputeTime * 1000) << "ms" << std::endl;
	std::cout << "Min compute time: " << roundedFloatToString(minComputeTime * 1000) << "ms" << std::endl;
	std::cout << "------------------------------------------------------" << std::endl;
	for (int i = 0; i < maxIter; i++)
	{
		std::cout << steps[i] << " compute time: " << roundedFloatToString(1000 * stepTimes[i]) << "ms" << std::endl;
	}
	std::cout << "======================================================" << std::endl;

	glfwTerminate();
	return 0;
}


void processInput(GLFWwindow* window)
{
	if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
		glfwSetWindowShouldClose(window, true);

	int state = glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT);
	if (state == GLFW_PRESS)
	{
		if (firstClick)
		{
			firstClick = false;
			glfwGetCursorPos(window, &xPos, &yPos);
			lastXPos = xPos;
			lastYPos = yPos;
		}
		else
		{
			lastXPos = xPos;
			lastYPos = yPos;
			glfwGetCursorPos(window, &xPos, &yPos);
		}

		isPressed = true;
	}
	else if (state == GLFW_RELEASE)
	{
		isPressed = false;
		firstClick = true;
	}

}

/* OpenGL callback function */
void framebuffer_size_callback(GLFWwindow* window, int width, int height)
{
	glViewport(0, 0, width, height);
}