#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <Shaders/Shader.h>
#include <iostream>
#include <test.cuh>
#include <cpuFluidSim.h>
#include <vector>
#include <string>

void framebuffer_size_callback(GLFWwindow* window, int width, int height);
void processInput(GLFWwindow* window);

// settings
const unsigned int SCR_WIDTH = 512;
const unsigned int SCR_HEIGHT = 512;
const int SCALE = 2;

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


	if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
	{
		std::cout << "Failed to initialize GLAD" << std::endl;
		return -1;
	}

	Shader shader = Shader("Programs/test_shader.vert", "Programs/test_shader.frag");


	glGenVertexArrays(1, &VAO);
	glGenBuffers(1, &VBO);
	glGenBuffers(1, &EBO);

	glBindVertexArray(VAO);

	glBindBuffer(GL_ARRAY_BUFFER, VBO);
	glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);


	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);

	glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), (void*)0);
	glEnableVertexAttribArray(0);

	glGenTextures(1, &texture);
	glBindTexture(GL_TEXTURE_2D, texture);
	// set the texture wrapping/filtering options (on the currently bound texture object)
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, SCR_WIDTH / SCALE, SCR_HEIGHT / SCALE, 0, GL_RGBA, GL_UNSIGNED_BYTE, 0);

	//cudaInit(SCR_WIDTH, SCR_HEIGHT, SCALE, texture);
	init(SCR_WIDTH, SCR_HEIGHT, SCALE);

	float deltaTime = 0.f;
	float lastTime = 0.0f;

	int nbFrames = 0;
	double lasttFrameTime = glfwGetTime();

	while (!glfwWindowShouldClose(window))
	{
		deltaTime = glfwGetTime() - lastTime;
		lastTime = glfwGetTime();

		

			// Measure speed
		double currentTime = glfwGetTime();
		nbFrames++;
		if (currentTime - lasttFrameTime >= 1.0) { // If last prinf() was more than 1 sec ago
				// printf and reset timer
			std::string frames = "FPS: " + std::to_string(nbFrames);
			glfwSetWindowTitle(window, frames.c_str());
			nbFrames = 0;
			lasttFrameTime += 1.0;
		}


		processInput(window);

		
		//computeField(deltaTime, xPos/SCALE, (SCR_HEIGHT - yPos)/SCALE, lastXPos/SCALE, (SCR_HEIGHT - (lastYPos))/SCALE, isPressed);
		on_frame(texture);

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

	cudaExit();
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
			on_mouse_button(xPos, yPos);
		}
		else
		{
			lastXPos = xPos;
			lastYPos = yPos;
			glfwGetCursorPos(window, &xPos, &yPos);
			on_mouse_button(xPos, yPos);
		}

		isPressed = true;
	}
	else if (state == GLFW_RELEASE)
	{
		isPressed = false;
		firstClick = true;
	}

}


void framebuffer_size_callback(GLFWwindow* window, int width, int height)
{
	// make sure the viewport matches the new window dimensions; note that width and 
	// height will be significantly larger than specified on retina displays.
	glViewport(0, 0, width, height);
}