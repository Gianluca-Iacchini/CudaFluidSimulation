#include "test.cuh"
#include <math.h>
#include <surface_functions.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda_runtime_api.h"
#include "cudart_platform.h"

#define CLAMP(val, minv, maxv) fminf(maxv, fmaxf(minv, val))
#define MIX(v0, v1, t) v0 * (1.f - t) + v1 * t 

#define CUDA_CALL(x) cudaError_t error = cudaGetLastError(); if (error != cudaSuccess) { std::cout << cudaGetErrorName(error) << std::endl; std::abort(); } x
#

struct Particle
{
	float2 u; // velocity
	float3 color;
};

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

static Particle* newField;
static Particle* oldField;
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
void cudaInit(size_t x, size_t y, int scale)
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

	cudaMalloc(&colorField, xSize * ySize * sizeof(uchar4));
	cudaMalloc(&oldField, xSize * ySize * sizeof(Particle));
	cudaMalloc(&newField, xSize * ySize * sizeof(Particle));
	cudaMalloc(&pressureOld, xSize * ySize * sizeof(float));
	cudaMalloc(&pressureNew, xSize * ySize * sizeof(float));
	cudaMalloc(&vorticityField, xSize * ySize * sizeof(float));
	cudaMalloc(&divergenceField, xSize * ySize * sizeof(float));
}

// releases all buffers, must be called on program exit
void cudaExit()
{
	cudaFree(colorField);
	cudaFree(oldField);
	cudaFree(newField);
	cudaFree(pressureOld);
	cudaFree(pressureNew);
	cudaFree(vorticityField);
	cudaFree(divergenceField);
}

// interpolates quantity of grid cells
__device__ Particle interpolate(float2 v, Particle* field, size_t xSize, size_t ySize)
{
	float x1 = (int)v.x;
	float y1 = (int)v.y;
	float x2 = (int)v.x + 1;
	float y2 = (int)v.y + 1;
	
	Particle q1, q2, q3, q4;

	q1 = field[int(CLAMP(y1, 0.0f, ySize - 1.0f)) * xSize + int(CLAMP(x1, 0.0f, xSize - 1.0f))];
	q2 = field[int(CLAMP(y1, 0.0f, ySize - 1.0f)) * xSize + int(CLAMP(x2, 0.0f, xSize - 1.0f))];
	q3 = field[int(CLAMP(y2, 0.0f, ySize - 1.0f)) * xSize + int(CLAMP(x1, 0.0f, xSize - 1.0f))];
	q4 = field[int(CLAMP(y2, 0.0f, ySize - 1.0f)) * xSize + int(CLAMP(x2, 0.0f, xSize - 1.0f))];


	float tx = (v.x - x1) / (x2 - x1);
	float ty = (v.y - y1) / (y2 - y1);

	float2 u1 = MIX(q1.u, q2.u, tx); 
	float2 u2 = MIX(q3.u, q4.u, tx);

	float3 c1 = MIX(q1.color, q2.color, tx);
	float3 c2 = MIX(q3.color, q4.color, tx); 

	Particle res;
	res.u = MIX(u1, u2, ty);
	res.color = MIX(c1, c2, ty); 
	return res;
}


// computes divergency of velocity field
__global__ void computeDivergence(float* divergenceField, Particle* field, size_t xSize, size_t ySize)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	Particle qL, qR, qB, qT;

	qL = field[int(CLAMP(y, 0.0f, ySize - 1.0f)) * xSize + int(CLAMP(x - 1, 0.0f, xSize - 1.0f))];
	qR = field[int(CLAMP(y, 0.0f, ySize - 1.0f)) * xSize + int(CLAMP(x + 1, 0.0f, xSize - 1.0f))];
	qB = field[int(CLAMP(y - 1, 0.0f, ySize - 1.0f)) * xSize + int(CLAMP(x, 0.0f, xSize - 1.0f))];
	qT = field[int(CLAMP(y + 1, 0.0f, ySize - 1.0f)) * xSize + int(CLAMP(x, 0.0f, xSize - 1.0f))];

	divergenceField[y * xSize + x] = 0.5f * (qR.u.x - qL.u.x + qT.u.y - qB.u.y);
}


// adds quantity to particles using bilinear interpolation
__global__ void advect(Particle* newField, Particle* oldField, size_t xSize, size_t ySize, float dDiffusion, float dt)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	float decay = 1.0f / (1.0f + dDiffusion * dt);
	float2 pos = { x * 1.0f, y * 1.0f };
	Particle& Pold = oldField[y * xSize + x];
	// find new particle tracing where it came from
	Particle p = interpolate(pos - Pold.u * dt, oldField, xSize, ySize);
	p.u = p.u * decay;
	p.color.x = fminf(1.0f, pow(p.color.x, 1.005f) * decay);
	p.color.y = fminf(1.0f, pow(p.color.y, 1.005f) * decay);
	p.color.z = fminf(1.0f, pow(p.color.z, 1.005f) * decay);
	newField[y * xSize + x] = p;
}

// calculates color field diffusion
__global__ void computeColor(Particle* newField, Particle* oldField, size_t xSize, size_t ySize, float cDiffusion, float dt)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	float alpha = cDiffusion * cDiffusion / dt;
	float beta = 4.0f + alpha;

	float3 cL, cR, cB, cT, cC;
	cL = oldField[int(CLAMP(y, 0.0f, ySize - 1.0f)) * xSize + int(CLAMP(x - 1, 0.0f, xSize - 1.0f))].color;
	cR = oldField[int(CLAMP(y, 0.0f, ySize - 1.0f)) * xSize + int(CLAMP(x + 1, 0.0f, xSize - 1.0f))].color;
	cB = oldField[int(CLAMP(y - 1, 0.0f, ySize - 1.0f)) * xSize + int(CLAMP(x, 0.0f, xSize - 1.0f))].color;
	cT = oldField[int(CLAMP(y + 1, 0.0f, ySize - 1.0f)) * xSize + int(CLAMP(x, 0.0f, xSize - 1.0f))].color;
	cC = oldField[y * xSize + x].color;

	newField[y * xSize + x].color = (cL + cR + cB + cT + cC * alpha) * (1.f / beta);

}

// fills output image with corresponding color
__global__ void paint(uchar4* colorField, Particle* field, size_t xSize, size_t ySize)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	float R = field[y * xSize + x].color.x;
	float G = field[y * xSize + x].color.y;
	float B = field[y * xSize + x].color.z;

	colorField[y * xSize + x] = make_uchar4(fminf(255.0f, 255.0f * R), fminf(255.0f, 255.0f * G), fminf(255.0f, 255.0f * B), 255);
}

// calculates nonzero divergency velocity field u
__global__ void diffuse(Particle* newField, Particle* oldField, size_t xSize, size_t ySize, float vDiffusion, float dt)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	float2 pos = { x * 1.0f, y * 1.0f };

	// perfoms one iteration of jacobi method (diffuse method should be called 20-50 times per cell)
	float alpha = vDiffusion * vDiffusion / dt;
	float beta = 4.0f + alpha;

	float2 uL, uR, uB, uT, uC;

	uL = oldField[int(CLAMP(y, 0.0f, ySize - 1.0f)) * xSize + int(CLAMP(x - 1, 0.0f, xSize - 1.0f))].u;
	uR = oldField[int(CLAMP(y, 0.0f, ySize - 1.0f)) * xSize + int(CLAMP(x + 1, 0.0f, xSize - 1.0f))].u;
	uB = oldField[int(CLAMP(y - 1, 0.0f, ySize - 1.0f)) * xSize + int(CLAMP(x, 0.0f, xSize - 1.0f))].u;
	uT = oldField[int(CLAMP(y + 1, 0.0f, ySize - 1.0f)) * xSize + int(CLAMP(x, 0.0f, xSize - 1.0f))].u;
	uC = oldField[y * xSize + x].u;

	newField[y * xSize + x].u = (uT + uB + uL + uR + uC * alpha) * (1.f / beta);
}

// performs iteration of jacobi method on pressure field
__global__ void computePressureImpl(Particle* field, float* divergenceField, size_t xSize, size_t ySize, float* pNew, float* pOld, float dt)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	float div = divergenceField[y * xSize + x];

	float pL, pR, pB, pT, pC;

	pL = pOld[int(CLAMP(y, 0.0f, ySize - 1.0f)) * xSize + int(CLAMP(x - 1, 0.0f, xSize - 1.0f))];
	pR = pOld[int(CLAMP(y, 0.0f, ySize - 1.0f)) * xSize + int(CLAMP(x + 1, 0.0f, xSize - 1.0f))];
	pB = pOld[int(CLAMP(y - 1, 0.0f, ySize - 1.0f)) * xSize + int(CLAMP(x, 0.0f, xSize - 1.0f))];
	pT = pOld[int(CLAMP(y + 1, 0.0f, ySize - 1.0f)) * xSize + int(CLAMP(x, 0.0f, xSize - 1.0f))];
	

	float pressure = (pL + pR + pB + pT - div) * 0.25f;


	pNew[y * xSize + x] = pressure;
}

// projects pressure field on velocity field
__global__ void project(Particle* newField, size_t xSize, size_t ySize, float* pField)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	float2& u = newField[y * xSize + x].u;

	float pL, pR, pB, pT, pC;

	pL = pField[int(CLAMP(y, 0.0f, ySize - 1.0f)) * xSize + int(CLAMP(x - 1, 0.0f, xSize - 1.0f))];
	pR = pField[int(CLAMP(y, 0.0f, ySize - 1.0f)) * xSize + int(CLAMP(x + 1, 0.0f, xSize - 1.0f))];
	pB = pField[int(CLAMP(y - 1, 0.0f, ySize - 1.0f)) * xSize + int(CLAMP(x, 0.0f, xSize - 1.0f))];
	pT = pField[int(CLAMP(y + 1, 0.0f, ySize - 1.0f)) * xSize + int(CLAMP(x, 0.0f, xSize - 1.0f))];

	float2 subtractVel = { (pR - pL) * 0.5f, (pT - pB) * 0.5f };


	u = u - subtractVel;
}

// applies force and add color dye to the particle field
__global__ void applyForce(Particle* field, size_t xSize, size_t ySize, float3 color, float2 F, float2 pos, int r, float dt)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	float e = expf(-((x - pos.x) * (x - pos.x) + (y - pos.y) * (y - pos.y)) / r);
	float2 uF = F * dt * e;
	Particle& p = field[y * xSize + x];
	p.u = p.u + uF;
	color = color * e + p.color;
	p.color.x = color.x;
	p.color.y = color.y;
	p.color.z = color.z;
}


// applies vorticity to velocity field
__global__ void computeVorticity(Particle* newField, Particle* oldField, float* vField, size_t xSize, size_t ySize, float vorticity, float dt)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;


	Particle qL, qR, qB, qT;

	qL = oldField[int(CLAMP(y, 0.0f, ySize - 1.0f)) * xSize + int(CLAMP(x - 1, 0.0f, xSize - 1.0f))];
	qR = oldField[int(CLAMP(y, 0.0f, ySize - 1.0f)) * xSize + int(CLAMP(x + 1, 0.0f, xSize - 1.0f))];
	qB = oldField[int(CLAMP(y - 1, 0.0f, ySize - 1.0f)) * xSize + int(CLAMP(x, 0.0f, xSize - 1.0f))];
	qT = oldField[int(CLAMP(y + 1, 0.0f, ySize - 1.0f)) * xSize + int(CLAMP(x, 0.0f, xSize - 1.0f))];

	vField[y * xSize + x] = 0.5f * (qR.u.y - qL.u.y - qT.u.x + qB.u.x);

	__syncthreads();

	Particle& pOld = oldField[y * xSize + x];
	Particle& pNew = newField[y * xSize + x];


	float vortL = vField[int(CLAMP(y, 0.0f, ySize - 1.0f)) * xSize + int(CLAMP(x - 1, 0.0f, xSize - 1.0f))];
	float vortR = vField[int(CLAMP(y, 0.0f, ySize - 1.0f)) * xSize + int(CLAMP(x + 1, 0.0f, xSize - 1.0f))];
	float vortB = vField[int(CLAMP(y - 1, 0.0f, ySize - 1.0f)) * xSize + int(CLAMP(x, 0.0f, xSize - 1.0f))];
	float vortT = vField[int(CLAMP(y + 1, 0.0f, ySize - 1.0f)) * xSize + int(CLAMP(x, 0.0f, xSize - 1.0f))];
	float vortC = vField[int(CLAMP(y, 0.0f, ySize - 1.0f)) * xSize + int(CLAMP(x, 0.0f, xSize - 1.0f))];

	float2 v = { (abs(vortT) - abs(vortB)) * 0.5f, (abs(vortL) - abs(vortR)) * 0.5f };

	float length = sqrtf(v.x * v.x + v.y * v.y) + 0.001f;

	v = v * (1.0f / length);

	v = v * vortC * vorticity;
	pOld.u = pOld.u + v * dt;
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

// performs several iterations over velocity and color fields
void computeDiffusion(dim3 numBlocks, dim3 threadsPerBlock, float dt)
{
	// diffuse velocity and color
	for (int i = 0; i < sConfig.velocityIterations; i++)
	{
		diffuse <<<numBlocks, threadsPerBlock >> > (newField, oldField, xSize, ySize, config.velocityDiffusion, dt);
		computeColor <<<numBlocks, threadsPerBlock >> > (newField, oldField, xSize, ySize, config.colorDiffusion, dt);
		std::swap(newField, oldField);
	}
}

// performs several iterations over pressure field
void computePressure(dim3 numBlocks, dim3 threadsPerBlock, float dt)
{
	computeDivergence << <numBlocks, threadsPerBlock >> > (divergenceField, oldField, xSize, ySize);

	for (int i = 0; i < sConfig.pressureIterations; i++)
	{
		computePressureImpl << <numBlocks, threadsPerBlock >> > (oldField, divergenceField, xSize, ySize, pressureNew, pressureOld, dt);
		std::swap(pressureOld, pressureNew);
	}
}

// main function, calls vorticity -> diffusion -> force -> pressure -> project -> advect -> paint -> bloom
void computeField(uchar4* result, float dt, int x1pos, int y1pos, int x2pos, int y2pos, bool isPressed)
{
	dim3 threadsPerBlock(sConfig.xThreads, sConfig.yThreads);
	dim3 numBlocks(xSize / threadsPerBlock.x, ySize / threadsPerBlock.y);

	// advect
	advect << <numBlocks, threadsPerBlock >> > (newField, oldField, xSize, ySize, config.densityDiffusion, dt);
	std::swap(newField, oldField);

	// curls and vortisity
	//computeVorticity <<<numBlocks, threadsPerBlock >> > (vorticityField, oldField, xSize, ySize);
	computeVorticity <<<numBlocks, threadsPerBlock >> > (newField, oldField, vorticityField, xSize, ySize, config.vorticity, dt);
	std::swap(oldField, newField);

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
		applyForce << <numBlocks, threadsPerBlock >> > (newField, xSize, ySize, currentColor, F, pos, config.radius, dt);
		std::swap(oldField, newField);
	}
	else
	{
		timeSincePress += dt;
	}

	// compute pressure
	computePressure(numBlocks, threadsPerBlock, dt);

	// project
	project <<<numBlocks, threadsPerBlock >> > (oldField, xSize, ySize, pressureOld);
	cudaMemset(pressureOld, 0.0f, xSize * ySize * sizeof(float));



	// paint image
	paint <<<numBlocks, threadsPerBlock >> > (colorField, oldField, xSize, ySize);

	// apply bloom in mouse pos
	if (config.bloomEnabled && timeSincePress < 5.0f)
	{
		applyBloom <<<numBlocks, threadsPerBlock >> > (colorField, xSize, ySize, x2pos, y2pos, config.radius, config.bloomIntense);
	}

	
	cudaDeviceSynchronize();
	cudaMemcpy(result, colorField, xSize * ySize * sizeof(uchar4), cudaMemcpyDeviceToHost);

	cudaError_t error = cudaGetLastError();
	if (error != cudaSuccess)
	{
		std::cout << cudaGetErrorName(error) << std::endl;
	}

	cudaDeviceSynchronize();
}