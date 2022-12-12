#pragma once
/* 
	Shader class from learnopengl.com
	https://learnopengl.com/Getting-started/Shaders
*/
#include "../../../Include/glad/glad.h"
#include <iostream>


class Shader {
public:
	unsigned int ID;

	Shader(const char* vertexPath, const char* framgentPath, const char* geometryPath = nullptr);
	Shader() {}

	void Use()
	{
		glUseProgram(ID);
	}

	void SetInt(const std::string& name, int value) const
	{
		glUniform1i(glGetUniformLocation(ID, name.c_str()), (int)value);
	}

private:
	void checkCompileErrors(GLuint shader, std::string type);

};