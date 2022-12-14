#include "test.cuh"
#include <math.h>
#include <cuda_gl_interop.h>
#include <surface_functions.h>
#include <helper_math.h>

#define CLAMP(val, minv, maxv) fminf(maxv, fmaxf(minv, val))
#define MIX(v0, v1, t) v0 * (1.f - t) + v1 * t 

#define CUDA_CALL(x) cudaError_t error = cudaGetLastError(); if (error != cudaSuccess) { std::cout << cudaGetErrorName(error) << std::endl; std::abort(); } x
#

static struct Config
{
	float velocityDiffusion;
	float pressure;
	float vorticity;
	float colorDiffusion;
	float densityDiffusion;
	float forceScale;
	float bloomIntense;
	int radius;
	bool bloomEnabled;
} config;

static struct SystemConfig
{
	int velocityIterations = 20;
	int pressureIterations = 30;
	int xThreads = 16;
	int yThreads = 32;
} sConfig;


static const int colorArraySize = 7;
static float3 colorArray[colorArraySize];



static float3* oldColor;
static float3* newColor;

static float2* oldVel;
static float2* newVel;

static uchar4* colorField;
static size_t xSize, ySize;
static float* pressureOld;
static float* pressureNew;
static float* vorticityField;
static float* divergenceField;
static float3 currentColor;
static float elapsedTime = 0.0f;
static float timeSincePress = 0.0f;


void setConfig(
	float vDiffusion = 0.8f,
	float pressure = 1.5f,
	float vorticity = 50.f,
	float cDiffuion = 0.8f,
	float dDiffuion = 1.2f,
	float force = 5000.0f,
	float bloomIntense = 0.1f,
	int radius = 1600,
	bool bloom = true
)

{
	config.velocityDiffusion = vDiffusion;
	config.pressure = pressure;
	config.vorticity = vorticity;
	config.colorDiffusion = cDiffuion;
	config.densityDiffusion = dDiffuion;
	config.forceScale = force;
	config.bloomIntense = bloomIntense;
	config.radius = radius;
	config.bloomEnabled = bloom;
}

cudaGraphicsResource_t textureResource = 0;
cudaArray* textureArray = 0;

cudaSurfaceObject_t surfObj;
cudaResourceDesc resourceDesc;

// inits all buffers, must be called before computeField function call
void cudaInit(size_t x, size_t y, int scale, GLuint texture)
{
	setConfig();

	colorArray[0] = { 1.0f, 0.0f, 0.0f };
	colorArray[1] = { 0.0f, 1.0f, 0.0f };
	colorArray[2] = { 1.0f, 0.0f, 1.0f };
	colorArray[3] = { 1.0f, 1.0f, 0.0f };
	colorArray[4] = { 0.0f, 1.0f, 1.0f };
	colorArray[5] = { 1.0f, 0.0f, 1.0f };
	colorArray[6] = { 1.0f, 0.5f, 0.3f };

	int idx = rand() % colorArraySize;
	currentColor = colorArray[idx];

	xSize = x/scale, ySize = y/scale;
	config.radius /= (scale * scale);

	cudaSetDevice(0);
	cudaGLSetGLDevice(0);

	cudaMalloc(&colorField, xSize * ySize * sizeof(uchar4));

	cudaMalloc(&oldColor, xSize * ySize * sizeof(float3));
	cudaMalloc(&newColor, xSize * ySize * sizeof(float3));
	cudaMalloc(&oldVel, xSize * ySize * sizeof(float2));
	cudaMalloc(&newVel, xSize * ySize * sizeof(float2));


	cudaMalloc(&pressureOld, xSize * ySize * sizeof(float));
	cudaMalloc(&pressureNew, xSize * ySize * sizeof(float));
	cudaMalloc(&vorticityField, xSize * ySize * sizeof(float));
	cudaMalloc(&divergenceField, xSize * ySize * sizeof(float));

	

	cudaError_t cgError = cudaGraphicsGLRegisterImage(&textureResource, texture, GL_TEXTURE_2D, cudaGraphicsMapFlagsWriteDiscard);


	memset(&resourceDesc, 0, sizeof(resourceDesc));
	resourceDesc.resType = cudaResourceTypeArray;

	cudaGraphicsMapResources(1, &textureResource);
	cudaGraphicsSubResourceGetMappedArray(&textureArray, textureResource, 0, 0);

	resourceDesc.res.array.array = textureArray;
	cudaCreateSurfaceObject(&surfObj, &resourceDesc);
}

// releases all buffers, must be called on program exit
void cudaExit()
{
	cudaFree(oldVel);
	cudaFree(newVel);
	cudaFree(oldColor);
	cudaFree(newColor);
	cudaFree(colorField);
	cudaFree(pressureOld);
	cudaFree(pressureNew);
	cudaFree(vorticityField);
	cudaFree(divergenceField);
}

// interpolates quantity of grid cells
__device__ float2 interpolate(float2 v, float2* vel, size_t xSize, size_t ySize)
{
	float x1 = (int)v.x;
	float y1 = (int)v.y;
	float x2 = (int)v.x + 1;
	float y2 = (int)v.y + 1;
	
	float2 v1, v2, v3, v4;

	v1 = vel[int(CLAMP(y1, 0.0f, ySize - 1.0f)) * xSize + int(CLAMP(x1, 0.0f, xSize - 1.0f))];
	v2 = vel[int(CLAMP(y1, 0.0f, ySize - 1.0f)) * xSize + int(CLAMP(x2, 0.0f, xSize - 1.0f))];
	v3 = vel[int(CLAMP(y2, 0.0f, ySize - 1.0f)) * xSize + int(CLAMP(x1, 0.0f, xSize - 1.0f))];
	v4 = vel[int(CLAMP(y2, 0.0f, ySize - 1.0f)) * xSize + int(CLAMP(x2, 0.0f, xSize - 1.0f))];


	float tx = (v.x - x1) / (x2 - x1);
	float ty = (v.y - y1) / (y2 - y1);

	float2 u1 = MIX(v1, v2, tx); 
	float2 u2 = MIX(v3, v4, tx);

	return MIX(u1, u2, ty);
}


// interpolates quantity of grid cells
__device__ float3 interpolate(float2 v, float3* col, size_t xSize, size_t ySize)
{
	float x1 = (int)v.x;
	float y1 = (int)v.y;
	float x2 = (int)v.x + 1;
	float y2 = (int)v.y + 1;

	float3 c1, c2, c3, c4;

	c1 = col[int(CLAMP(y1, 0.0f, ySize - 1.0f)) * xSize + int(CLAMP(x1, 0.0f, xSize - 1.0f))];
	c2 = col[int(CLAMP(y1, 0.0f, ySize - 1.0f)) * xSize + int(CLAMP(x2, 0.0f, xSize - 1.0f))];
	c3 = col[int(CLAMP(y2, 0.0f, ySize - 1.0f)) * xSize + int(CLAMP(x1, 0.0f, xSize - 1.0f))];
	c4 = col[int(CLAMP(y2, 0.0f, ySize - 1.0f)) * xSize + int(CLAMP(x2, 0.0f, xSize - 1.0f))];


	float tx = (v.x - x1) / (x2 - x1);
	float ty = (v.y - y1) / (y2 - y1);


	float3 col1 = MIX(c1, c2, tx);
	float3 col2 = MIX(c3, c4, tx);

	return MIX(col1, col2, ty);

}

// computes divergency of velocity field
__global__ void computeDivergence(float* divergenceField, float2* vel, size_t xSize, size_t ySize)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	float2 vL, vR, vB, vT;

	vL = vel[int(CLAMP(y, 0.0f, ySize - 1.0f)) * xSize + int(CLAMP(x - 1, 0.0f, xSize - 1.0f))];
	vR = vel[int(CLAMP(y, 0.0f, ySize - 1.0f)) * xSize + int(CLAMP(x + 1, 0.0f, xSize - 1.0f))];
	vB = vel[int(CLAMP(y - 1, 0.0f, ySize - 1.0f)) * xSize + int(CLAMP(x, 0.0f, xSize - 1.0f))];
	vT = vel[int(CLAMP(y + 1, 0.0f, ySize - 1.0f)) * xSize + int(CLAMP(x, 0.0f, xSize - 1.0f))];

	divergenceField[y * xSize + x] = 0.5f * (vR.x - vL.x + vT.y - vB.y);
}


// adds quantity to particles using bilinear interpolation
__global__ void advect(float2* newVel, float2* oldVel, size_t xSize, size_t ySize, float dDiffusion, float dt)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	float decay = 1.0f / (1.0f + dDiffusion * dt);
	float2 pos = { x * 1.0f, y * 1.0f };
	float2& oldV = oldVel[y * xSize + x];
	// find new particle tracing where it came from
	float2 vLerp = interpolate(pos - oldV * dt, oldVel, xSize, ySize);
	vLerp  = vLerp * decay;
	newVel[y * xSize + x] = vLerp;
}

// adds quantity to particles using bilinear interpolation
__global__ void advect(float3* newColor, float3* oldColor, float2* vel, size_t xSize, size_t ySize, float dDiffusion, float dt)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	float decay = 1.0f / (1.0f + dDiffusion * dt);
	float2 pos = { x * 1.0f, y * 1.0f };
	float2& oldV = vel[y * xSize + x];
	// find new particle tracing where it came from
	float3 cLerp = interpolate(pos - oldV * dt, oldColor,  xSize, ySize);
	
	cLerp.x = fminf(1.0f, pow(cLerp.x, 1.005f) * decay);
	cLerp.y = fminf(1.0f, pow(cLerp.y, 1.005f) * decay);
	cLerp.z = fminf(1.0f, pow(cLerp.z, 1.005f) * decay);
	newColor[y * xSize + x] = cLerp;
}

// calculates color field diffusion
__global__ void diffuseCol(float3* newColor, float3* oldColor, size_t xSize, size_t ySize, float cDiffusion, float dt)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	float alpha = cDiffusion * cDiffusion / dt;
	float beta = 4.0f + alpha;

	float3 cL, cR, cB, cT, cC;
	cL = oldColor[int(CLAMP(y, 0.0f, ySize - 1.0f)) * xSize + int(CLAMP(x - 1, 0.0f, xSize - 1.0f))];
	cR = oldColor[int(CLAMP(y, 0.0f, ySize - 1.0f)) * xSize + int(CLAMP(x + 1, 0.0f, xSize - 1.0f))];
	cB = oldColor[int(CLAMP(y - 1, 0.0f, ySize - 1.0f)) * xSize + int(CLAMP(x, 0.0f, xSize - 1.0f))];
	cT = oldColor[int(CLAMP(y + 1, 0.0f, ySize - 1.0f)) * xSize + int(CLAMP(x, 0.0f, xSize - 1.0f))];
	cC = oldColor[y * xSize + x];

	newColor[y * xSize + x] = (cL + cR + cB + cT + cC * alpha) * (1.f / beta);

}

// fills output image with corresponding color
__global__ void paint(uchar4* colorField, float3* oldColor, size_t xSize, size_t ySize)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	float R = oldColor[y * xSize + x].x;
	float G = oldColor[y * xSize + x].y;
	float B = oldColor[y * xSize + x].z;

	colorField[y * xSize + x] = make_uchar4(fminf(255.0f, 255.0f * R), fminf(255.0f, 255.0f * G), fminf(255.0f, 255.0f * B), 255);
}

// calculates nonzero divergency velocity field u
__global__ void diffuseVel(float2* newVel, float2* oldVel, size_t xSize, size_t ySize, float vDiffusion, float dt)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	float2 pos = { x * 1.0f, y * 1.0f };

	// perfoms one iteration of jacobi method (diffuse method should be called 20-50 times per cell)
	float alpha = vDiffusion * vDiffusion / dt;
	float beta = 4.0f + alpha;

	float2 uL, uR, uB, uT, uC;

	uL = oldVel[int(CLAMP(y, 0.0f, ySize - 1.0f)) * xSize + int(CLAMP(x - 1, 0.0f, xSize - 1.0f))];
	uR = oldVel[int(CLAMP(y, 0.0f, ySize - 1.0f)) * xSize + int(CLAMP(x + 1, 0.0f, xSize - 1.0f))];
	uB = oldVel[int(CLAMP(y - 1, 0.0f, ySize - 1.0f)) * xSize + int(CLAMP(x, 0.0f, xSize - 1.0f))];
	uT = oldVel[int(CLAMP(y + 1, 0.0f, ySize - 1.0f)) * xSize + int(CLAMP(x, 0.0f, xSize - 1.0f))];
	uC = oldVel[y * xSize + x];

	newVel[y * xSize + x] = (uT + uB + uL + uR + uC * alpha) * (1.f / beta);
}

// performs iteration of jacobi method on pressure field
__global__ void computePressureImpl(float* divergenceField, size_t xSize, size_t ySize, float* pNew, float* pOld, float dt)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	float div = divergenceField[y * xSize + x];

	float pL, pR, pB, pT;

	pL = pOld[int(CLAMP(y, 0.0f, ySize - 1.0f)) * xSize + int(CLAMP(x - 1, 0.0f, xSize - 1.0f))];
	pR = pOld[int(CLAMP(y, 0.0f, ySize - 1.0f)) * xSize + int(CLAMP(x + 1, 0.0f, xSize - 1.0f))];
	pB = pOld[int(CLAMP(y - 1, 0.0f, ySize - 1.0f)) * xSize + int(CLAMP(x, 0.0f, xSize - 1.0f))];
	pT = pOld[int(CLAMP(y + 1, 0.0f, ySize - 1.0f)) * xSize + int(CLAMP(x, 0.0f, xSize - 1.0f))];
	

	float pressure = (pL + pR + pB + pT - div) * 0.25f;


	pNew[y * xSize + x] = pressure;
}

// projects pressure field on velocity field
__global__ void project(float2* oldVel, size_t xSize, size_t ySize, float* pField)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	float2& u = oldVel[y * xSize + x];

	float pL, pR, pB, pT;

	pL = pField[int(CLAMP(y, 0.0f, ySize - 1.0f)) * xSize + int(CLAMP(x - 1, 0.0f, xSize - 1.0f))];
	pR = pField[int(CLAMP(y, 0.0f, ySize - 1.0f)) * xSize + int(CLAMP(x + 1, 0.0f, xSize - 1.0f))];
	pB = pField[int(CLAMP(y - 1, 0.0f, ySize - 1.0f)) * xSize + int(CLAMP(x, 0.0f, xSize - 1.0f))];
	pT = pField[int(CLAMP(y + 1, 0.0f, ySize - 1.0f)) * xSize + int(CLAMP(x, 0.0f, xSize - 1.0f))];

	float2 subtractVel = { (pR - pL) * 0.5f, (pT - pB) * 0.5f };


	u = u - subtractVel;
}

// applies force and add color dye to the particle field
__global__ void applyForce(float2* oldVel, float3* oldColor, size_t xSize, size_t ySize, float3 color, float2 F, float2 pos, int r, float dt)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	float e = expf(-((x - pos.x) * (x - pos.x) + (y - pos.y) * (y - pos.y)) / r);
	float2 uF = F * dt * e;
	
	float2& u = oldVel[y * xSize + x];
	float3& c = oldColor[y * xSize + x];
	
	u = u + uF;
	c += color * e;
}


// applies vorticity to velocity field
__global__ void computeVorticity(float2* newVel, float2* oldVel, float* vField, size_t xSize, size_t ySize, float vorticity, float dt)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;


	float2 vL, vR, vB, vT;

	vL = oldVel[int(CLAMP(y, 0.0f, ySize - 1.0f)) * xSize + int(CLAMP(x - 1, 0.0f, xSize - 1.0f))];
	vR = oldVel[int(CLAMP(y, 0.0f, ySize - 1.0f)) * xSize + int(CLAMP(x + 1, 0.0f, xSize - 1.0f))];
	vB = oldVel[int(CLAMP(y - 1, 0.0f, ySize - 1.0f)) * xSize + int(CLAMP(x, 0.0f, xSize - 1.0f))];
	vT = oldVel[int(CLAMP(y + 1, 0.0f, ySize - 1.0f)) * xSize + int(CLAMP(x, 0.0f, xSize - 1.0f))];

	vField[y * xSize + x] = 0.5f * (vR.y - vL.y - vT.x + vB.x);

	__syncthreads();

	float2& pOld = oldVel[y * xSize + x];
	float2& pNew = newVel[y * xSize + x];


	float vortL = vField[int(CLAMP(y, 0.0f, ySize - 1.0f)) * xSize + int(CLAMP(x - 1, 0.0f, xSize - 1.0f))];
	float vortR = vField[int(CLAMP(y, 0.0f, ySize - 1.0f)) * xSize + int(CLAMP(x + 1, 0.0f, xSize - 1.0f))];
	float vortB = vField[int(CLAMP(y - 1, 0.0f, ySize - 1.0f)) * xSize + int(CLAMP(x, 0.0f, xSize - 1.0f))];
	float vortT = vField[int(CLAMP(y + 1, 0.0f, ySize - 1.0f)) * xSize + int(CLAMP(x, 0.0f, xSize - 1.0f))];
	float vortC = vField[int(CLAMP(y, 0.0f, ySize - 1.0f)) * xSize + int(CLAMP(x, 0.0f, xSize - 1.0f))];

	float2 v = { (abs(vortT) - abs(vortB)) * 0.5f, (abs(vortL) - abs(vortR)) * 0.5f };

	float length = sqrtf(v.x * v.x + v.y * v.y) + 0.001f;

	v = v * (1.0f / length);

	v = v * vortC * vorticity;
	pOld = pOld + v * dt;
	pNew = pOld;
}

// adds flashlight effect near the mouse position
__global__ void applyBloom(uchar4* colorField, size_t xSize, size_t ySize, int xpos, int ypos, float radius, float bloomIntense)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;


	float e = bloomIntense * expf(-((x - xpos) * (x - xpos) + (y - ypos) * (y - ypos) + 1.0f) / (radius * radius));

	unsigned char R = colorField[y * xSize + x].x;
	unsigned char G = colorField[y * xSize + x].y;
	unsigned char B = colorField[y * xSize + x].z;

	float maxval = fmaxf(R, fmaxf(G, B));

	colorField[y * xSize + x] = make_uchar4(fminf(255.0f, R + maxval * e), fminf(255.0f, G + maxval * e), fminf(255.0f, B + maxval * e), 255);
}

__global__ void writeToTexture(cudaSurfaceObject_t surface, uchar4* colorField, size_t xSize, size_t ySize)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x < xSize && y < ySize) {
		surf2Dwrite(colorField[y * xSize + x], surface, x*sizeof(uchar4), y);
	}
}

// performs several iterations over velocity and color fields
void computeDiffusion(dim3 numBlocks, dim3 threadsPerBlock, float dt)
{
	// diffuse velocity and color
	for (int i = 0; i < sConfig.velocityIterations; i++)
	{
		diffuseVel <<<numBlocks, threadsPerBlock >> > (newVel, oldVel, xSize, ySize, config.velocityDiffusion, dt);
		diffuseCol <<<numBlocks, threadsPerBlock >> > (newColor, oldColor, xSize, ySize, config.colorDiffusion, dt);
		std::swap(newVel, oldVel);
		std::swap(oldColor, newColor);
	}
}

// performs several iterations over pressure field
void computePressure(dim3 numBlocks, dim3 threadsPerBlock, float dt)
{
	computeDivergence << <numBlocks, threadsPerBlock >> > (divergenceField, oldVel, xSize, ySize);

	for (int i = 0; i < sConfig.pressureIterations; i++)
	{
		computePressureImpl << <numBlocks, threadsPerBlock >> > (divergenceField, xSize, ySize, pressureNew, pressureOld, dt);
		std::swap(pressureOld, pressureNew);
	}
}

// main function, calls vorticity -> diffusion -> force -> pressure -> project -> advect -> paint -> bloom
void computeField(float dt, int x1pos, int y1pos, int x2pos, int y2pos, bool isPressed)
{
	dim3 threadsPerBlock(sConfig.xThreads, sConfig.yThreads);
	dim3 numBlocks(xSize / threadsPerBlock.x, ySize / threadsPerBlock.y);

	// advect
	advect << <numBlocks, threadsPerBlock >> > (newVel, oldVel, xSize, ySize, config.densityDiffusion, dt);
	std::swap(newVel, oldVel);
	advect << <numBlocks, threadsPerBlock >> > (newColor, oldColor, oldVel, xSize, ySize, config.densityDiffusion, dt);
	std::swap(newColor, oldColor);

	// curls and vortisity
	computeVorticity <<<numBlocks, threadsPerBlock >> > (newVel, oldVel, vorticityField, xSize, ySize, config.vorticity, dt);
	std::swap(newVel, oldVel);

	// diffuse velocity and color
	computeDiffusion(numBlocks, threadsPerBlock, dt);

	// apply force
	if (isPressed)
	{

		timeSincePress = 0.0f;
		elapsedTime += dt;
		// apply gradient to color
		int roundT = int(elapsedTime) % colorArraySize;
		int ceilT = int((elapsedTime)+1) % colorArraySize;
		float w = elapsedTime - int(elapsedTime);
		currentColor = colorArray[roundT] * (1 - w) + colorArray[ceilT] * w;

		float2 F;
		float scale = config.forceScale;
		F.x = (x2pos - x1pos) * scale;
		F.y = (y2pos - y1pos) * scale;
		float2 pos = { x2pos * 1.0f, y2pos * 1.0f };
		applyForce << <numBlocks, threadsPerBlock >> > (oldVel, oldColor, xSize, ySize, currentColor, F, pos, config.radius, dt);
	}
	else
	{
		timeSincePress += dt;
	}

	// compute pressure
	computePressure(numBlocks, threadsPerBlock, dt);

	// project
	project <<<numBlocks, threadsPerBlock >> > (oldVel, xSize, ySize, pressureOld);
	cudaMemset(pressureOld, 0.0f, xSize * ySize * sizeof(float));



	// paint image
	paint <<<numBlocks, threadsPerBlock >> > (colorField, oldColor, xSize, ySize);

	// apply bloom in mouse pos
	if (config.bloomEnabled && timeSincePress < 5.0f)
	{
		applyBloom <<<numBlocks, threadsPerBlock >> > (colorField, xSize, ySize, x2pos, y2pos, config.radius, config.bloomIntense);
	}

	writeToTexture << <numBlocks, threadsPerBlock >> > (surfObj, colorField, xSize, ySize);

	cudaError_t error = cudaGetLastError();
	if (error != cudaSuccess)
	{
		std::cout << cudaGetErrorName(error) << std::endl;
	}

	cudaDeviceSynchronize();
}