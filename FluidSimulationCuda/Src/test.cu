#include "test.cuh"
#include <math.h>
#include <cuda_gl_interop.h>
#include <surface_functions.h>
#include <helper_math.h>
#include <GLFW/glfw3.h>

#define CLAMP(val, minv, maxv) fminf(maxv, fmaxf(minv, val))
#define MIX(v0, v1, t) v0 * (1.f - t) + v1 * t 

#define CUDA_CALL(call)                                          \
do {                                                                  \
    cudaError_t err = call;                                           \
    if (cudaSuccess != err) {                                         \
        fprintf (stderr, "Cuda error in file '%s' in line %i : %s.\n",\
                 __FILE__, __LINE__, cudaGetErrorString(err) );       \
        exit(EXIT_FAILURE);                                           \
    }                                                                 \
} while (0)

double averageKernelTimes[8];
uint64_t g_totalFrames = 0;

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
	int yThreads = 16;
} sConfig;


static const int colorArraySize = 7;
static float3 colorArray[colorArraySize];



static float2* velocityField;
static float* pressureField;
static float3* dyeColorField;

static float* vorticityField;
static float* divergenceField;

static uchar4* textureColorField;

static size_t xSize, ySize;


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

__constant__ int xSize_d;
__constant__ int ySize_d;

__constant__ struct Config devConstants;

cudaStream_t stream_0;
cudaStream_t stream_1;

double* g_getAverageTimes()
{
	return averageKernelTimes;
}

// inits all buffers, must be called before computeField function call
void cudaInit(size_t x, size_t y, int scale, GLuint texture)
{
	setConfig();

	for (int i = 0; i < 8; i++)
	{
		averageKernelTimes[i] = 0;
	}

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


	
	CUDA_CALL(cudaSetDevice(0));
	CUDA_CALL(cudaGLSetGLDevice(0));

	CUDA_CALL(cudaMalloc(&textureColorField, xSize * ySize * sizeof(uchar4)));

	CUDA_CALL(cudaMalloc(&dyeColorField, xSize * ySize * sizeof(float3)));
	CUDA_CALL(cudaMalloc(&velocityField, xSize * ySize * sizeof(float2)));


	CUDA_CALL(cudaMalloc(&pressureField, xSize * ySize * sizeof(float)));
	CUDA_CALL(cudaMalloc(&vorticityField, xSize * ySize * sizeof(float)));
	CUDA_CALL(cudaMalloc(&divergenceField, xSize * ySize * sizeof(float)));

	int xs = xSize;
	int ys = ySize;

	CUDA_CALL(cudaStreamCreate(&stream_0));
	CUDA_CALL(cudaStreamCreate(&stream_1));

	CUDA_CALL(cudaMemcpyToSymbol(xSize_d, &xs, sizeof(int)));
	CUDA_CALL(cudaMemcpyToSymbol(ySize_d, &ys, sizeof(int)));
	CUDA_CALL(cudaMemcpyToSymbol(devConstants, &config, sizeof(Config)));

	CUDA_CALL(cudaGraphicsGLRegisterImage(&textureResource, texture, GL_TEXTURE_2D, cudaGraphicsMapFlagsWriteDiscard));


	memset(&resourceDesc, 0, sizeof(resourceDesc));
	resourceDesc.resType = cudaResourceTypeArray;

	CUDA_CALL(cudaGraphicsMapResources(1, &textureResource));
	CUDA_CALL(cudaGraphicsSubResourceGetMappedArray(&textureArray, textureResource, 0, 0));

	resourceDesc.res.array.array = textureArray;
	CUDA_CALL(cudaCreateSurfaceObject(&surfObj, &resourceDesc));
}

// releases all buffers, must be called on program exit
void cudaExit()
{
	CUDA_CALL(cudaFree(velocityField));
	CUDA_CALL(cudaFree(dyeColorField));
	CUDA_CALL(cudaFree(textureColorField));
	CUDA_CALL(cudaFree(pressureField));
	CUDA_CALL(cudaFree(vorticityField));
	CUDA_CALL(cudaFree(divergenceField));
}

// interpolates quantity of grid cells
__device__ float2 interpolate(float2 v, float2* vel)
{
	float x1 = (int)v.x;
	float y1 = (int)v.y;
	float x2 = (int)v.x + 1;
	float y2 = (int)v.y + 1;
	
	float2 v1, v2, v3, v4;

	v1 = vel[int(CLAMP(y1, 0.0f, ySize_d - 1.0f)) * xSize_d + int(CLAMP(x1, 0.0f, xSize_d - 1.0f))];
	v2 = vel[int(CLAMP(y1, 0.0f, ySize_d - 1.0f)) * xSize_d + int(CLAMP(x2, 0.0f, xSize_d - 1.0f))];
	v3 = vel[int(CLAMP(y2, 0.0f, ySize_d - 1.0f)) * xSize_d + int(CLAMP(x1, 0.0f, xSize_d - 1.0f))];
	v4 = vel[int(CLAMP(y2, 0.0f, ySize_d - 1.0f)) * xSize_d + int(CLAMP(x2, 0.0f, xSize_d - 1.0f))];


	float tx = (v.x - x1) / (x2 - x1);
	float ty = (v.y - y1) / (y2 - y1);

	float2 u1 = MIX(v1, v2, tx); 
	float2 u2 = MIX(v3, v4, tx);

	return MIX(u1, u2, ty);
}


// interpolates quantity of grid cells
__device__ float3 interpolate(float2 v, float3* col)
{
	float x1 = (int)v.x;
	float y1 = (int)v.y;
	float x2 = (int)v.x + 1;
	float y2 = (int)v.y + 1;

	float3 c1, c2, c3, c4;

	c1 = col[int(CLAMP(y1, 0.0f, ySize_d - 1.0f)) * xSize_d + int(CLAMP(x1, 0.0f, xSize_d - 1.0f))];
	c2 = col[int(CLAMP(y1, 0.0f, ySize_d - 1.0f)) * xSize_d + int(CLAMP(x2, 0.0f, xSize_d - 1.0f))];
	c3 = col[int(CLAMP(y2, 0.0f, ySize_d - 1.0f)) * xSize_d + int(CLAMP(x1, 0.0f, xSize_d - 1.0f))];
	c4 = col[int(CLAMP(y2, 0.0f, ySize_d - 1.0f)) * xSize_d + int(CLAMP(x2, 0.0f, xSize_d - 1.0f))];


	float tx = (v.x - x1) / (x2 - x1);
	float ty = (v.y - y1) / (y2 - y1);


	float3 col1 = MIX(c1, c2, tx);
	float3 col2 = MIX(c3, c4, tx);

	return MIX(col1, col2, ty);

}

// computes divergency of velocity field
__global__ void computeDivergence(float* divergenceField, float2* vel)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	float2 vL, vR, vB, vT;

	vL = vel[y * xSize_d + int(CLAMP(x - 1, 0.0f, xSize_d - 1.0f))];
	vR = vel[y * xSize_d + int(CLAMP(x + 1, 0.0f, xSize_d - 1.0f))];
	vB = vel[int(CLAMP(y - 1, 0.0f, ySize_d - 1.0f)) * xSize_d + x];
	vT = vel[int(CLAMP(y + 1, 0.0f, ySize_d - 1.0f)) * xSize_d + x];

	divergenceField[y * xSize_d + x] = 0.5f * (vR.x - vL.x + vT.y - vB.y);
}


// adds quantity to particles using bilinear interpolation
__global__ void advect(float2* oldVel, float dt)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	float decay = 1.0f / (1.0f + devConstants.densityDiffusion * dt);
	float2 pos = { x * 1.0f, y * 1.0f };
	float2& oldV = oldVel[y * xSize_d + x];
	// find new particle tracing where it came from
	float2 vLerp = interpolate(pos - oldV * dt, oldVel);
	vLerp  = vLerp * decay;

	__syncthreads();
	oldVel[y * xSize_d + x] = vLerp;
}

// adds quantity to particles using bilinear interpolation
__global__ void advect(float3* oldColor, float2* vel, float dt)
{

	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	float decay = 1.0f / (1.0f + devConstants.densityDiffusion * dt);
	float2 pos = { x * 1.0f, y * 1.0f };
	float2& oldV = vel[y * xSize_d + x];
	// find new particle tracing where it came from
	float3 cLerp = interpolate(pos - oldV * dt, oldColor);
	
	cLerp.x = fminf(1.0f, pow(cLerp.x, 1.005f) * decay);
	cLerp.y = fminf(1.0f, pow(cLerp.y, 1.005f) * decay);
	cLerp.z = fminf(1.0f, pow(cLerp.z, 1.005f) * decay);

	__syncthreads();
	oldColor[y * xSize_d + x] = cLerp;
}

// calculates color field diffusion
__global__ void diffuseCol(float3* oldColor, float dt)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int RADIUS = 1;
	int BmR = 15;
	__shared__ float3 colorShared[18][18];

	if (threadIdx.y < RADIUS)
	{
		colorShared[threadIdx.y][threadIdx.x + RADIUS] = oldColor[int(CLAMP(y - 1, 0.0f, ySize_d - 1.0f)) * xSize_d + x];
	}
	else if (threadIdx.y >= BmR)
	{
		colorShared[threadIdx.y + 2 * RADIUS][threadIdx.x + RADIUS] = oldColor[int(CLAMP(y + 1, 0.0f, ySize_d - 1.0f)) * xSize_d + x];
	}
	if (threadIdx.x < RADIUS)
	{
		colorShared[threadIdx.y + RADIUS][threadIdx.x] = oldColor[y * xSize_d + int(CLAMP(x - 1, 0.0f, xSize_d - 1.0f))];
	}
	else if (threadIdx.x >= BmR)
	{
		colorShared[threadIdx.y + RADIUS][threadIdx.x + 2 * RADIUS] = oldColor[y * xSize_d + int(CLAMP(x + 1, 0.0f, xSize_d - 1.0f))];
	}

	colorShared[threadIdx.y + RADIUS][threadIdx.x + RADIUS] = oldColor[y * xSize_d + x];

	__syncthreads();

	float alpha = devConstants.colorDiffusion * devConstants.colorDiffusion/ dt;
	float beta = 4.0f + alpha;

	float3 cL, cR, cB, cT, cC;

	for (int i = 0; i < 20; i++)
	{
		cL = colorShared[threadIdx.y + RADIUS][threadIdx.x + RADIUS - 1];
		cR = colorShared[threadIdx.y + RADIUS][threadIdx.x + RADIUS + 1];
		cB = colorShared[threadIdx.y + RADIUS - 1][threadIdx.x + RADIUS];
		cT = colorShared[threadIdx.y + RADIUS + 1][threadIdx.x + RADIUS];
		cC = colorShared[threadIdx.y + RADIUS][threadIdx.x + RADIUS];


		float3 newValue = (cL + cR + cB + cT + cC * alpha) * (1.f / beta);


		__syncthreads();

		colorShared[threadIdx.y + RADIUS][threadIdx.x + RADIUS] = newValue;

		__syncthreads();
	}

	oldColor[y * xSize_d + x] = colorShared[threadIdx.y + RADIUS][threadIdx.x + RADIUS];

}

// calculates nonzero divergency velocity field u
__global__ void diffuseVel(float2* oldVel, float dt)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int RADIUS = 1;
	int BmR = 15;
	__shared__ float2 velShared[18][18];

	if (threadIdx.y < RADIUS)
	{
		velShared[threadIdx.y][threadIdx.x + RADIUS] = oldVel[int(CLAMP(y - 1, 0.0f, ySize_d - 1.0f)) * xSize_d + x];
	}
	else if (threadIdx.y >= BmR)
	{
		velShared[threadIdx.y + 2 * RADIUS][threadIdx.x + RADIUS] = oldVel[int(CLAMP(y + 1, 0.0f, ySize_d - 1.0f)) * xSize_d + x];
	}
	if (threadIdx.x < RADIUS)
	{
		velShared[threadIdx.y + RADIUS][threadIdx.x] = oldVel[y * xSize_d + int(CLAMP(x - 1, 0.0f, xSize_d - 1.0f))];
	}
	else if (threadIdx.x >= BmR)
	{
		velShared[threadIdx.y + RADIUS][threadIdx.x + 2 * RADIUS] = oldVel[y * xSize_d + int(CLAMP(x + 1, 0.0f, xSize_d - 1.0f))];
	}

	velShared[threadIdx.y + RADIUS][threadIdx.x + RADIUS] = oldVel[y * xSize_d + x];

	__syncthreads();

	float2 pos = { x * 1.0f, y * 1.0f };

	// perfoms one iteration of jacobi method (diffuse method should be called 20-50 times per cell)
	float alpha = devConstants.velocityDiffusion * devConstants.velocityDiffusion / dt;
	float beta = 4.0f + alpha;

	float2 uL, uR, uB, uT, uC;

	for (int i = 0; i < 20; i++)
	{
		uL = velShared[threadIdx.y + RADIUS][threadIdx.x + RADIUS - 1];
		uR = velShared[threadIdx.y + RADIUS][threadIdx.x + RADIUS + 1];
		uB = velShared[threadIdx.y + RADIUS - 1][threadIdx.x + RADIUS];
		uT = velShared[threadIdx.y + RADIUS + 1][threadIdx.x + RADIUS];
		uC = velShared[threadIdx.y + RADIUS][threadIdx.x + RADIUS];

		float2 newValue = (uT + uB + uL + uR + uC * alpha) * (1.f / beta);


		__syncthreads();

		velShared[threadIdx.y + RADIUS][threadIdx.x + RADIUS] = newValue;

		__syncthreads();
	}


	oldVel[y * xSize_d + x] = velShared[threadIdx.y + RADIUS][threadIdx.x + RADIUS];
}

// fills output image with corresponding color
__global__ void paint(uchar4* colorField, float3* oldColor)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	float R = oldColor[y * xSize_d + x].x;
	float G = oldColor[y * xSize_d + x].y;
	float B = oldColor[y * xSize_d + x].z;

	colorField[y * xSize_d + x] = make_uchar4(fminf(255.0f, 255.0f * R), fminf(255.0f, 255.0f * G), fminf(255.0f, 255.0f * B), 255);
}

// performs iteration of jacobi method on pressure field
__global__ void computePressureImpl(float* divergenceField, float* pOld, float dt)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	float div = divergenceField[y * xSize_d + x];

	float pL, pR, pB, pT;

	int RADIUS = 1;
	int BmR = 15;
	__shared__ float pressureShared[18][18];

	if (threadIdx.y < RADIUS)
	{
		pressureShared[threadIdx.y][threadIdx.x + RADIUS] = pOld[int(CLAMP(y - 1, 0.0f, ySize_d - 1.0f)) * xSize_d + x];
	}
	else if (threadIdx.y >= BmR)
	{
		pressureShared[threadIdx.y + 2 * RADIUS][threadIdx.x + RADIUS] = pOld[int(CLAMP(y + 1, 0.0f, ySize_d - 1.0f)) * xSize_d + x];
	}
	if (threadIdx.x < RADIUS)
	{
		pressureShared[threadIdx.y + RADIUS][threadIdx.x] = pOld[y * xSize_d + int(CLAMP(x - 1, 0.0f, xSize_d - 1.0f))];
	}
	else if (threadIdx.x >= BmR)
	{
		pressureShared[threadIdx.y + RADIUS][threadIdx.x + 2 * RADIUS] = pOld[y * xSize_d + int(CLAMP(x + 1, 0.0f, xSize_d - 1.0f))];
	}

	pressureShared[threadIdx.y + RADIUS][threadIdx.x + RADIUS] = pOld[y * xSize_d + x];

	__syncthreads();

	for (int i = 0; i < 30; i++)
	{

		pL = pressureShared[threadIdx.y + RADIUS][threadIdx.x + RADIUS - 1];
		pR = pressureShared[threadIdx.y + RADIUS][threadIdx.x + RADIUS + 1];
		pB = pressureShared[threadIdx.y + RADIUS - 1][threadIdx.x + RADIUS];
		pT = pressureShared[threadIdx.y + RADIUS + 1][threadIdx.x + RADIUS];

		float pressure = (pL + pR + pB + pT - div) * 0.25f;

		__syncthreads();

		pressureShared[threadIdx.y + RADIUS][threadIdx.x + RADIUS] = pressure;

		__syncthreads();
	}

	pOld[y * xSize_d + x] = pressureShared[threadIdx.y + RADIUS][threadIdx.x + RADIUS];
}

// projects pressure field on velocity field
__global__ void project(float2* oldVel, float* pField)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	float2& u = oldVel[y * xSize_d + x];

	float pL, pR, pB, pT;

	pL = pField[y * xSize_d + int(CLAMP(x - 1, 0.0f, xSize_d - 1.0f))];
	pR = pField[y * xSize_d + int(CLAMP(x + 1, 0.0f, xSize_d - 1.0f))];
	pB = pField[int(CLAMP(y - 1, 0.0f, ySize_d - 1.0f)) * xSize_d + x];
	pT = pField[int(CLAMP(y + 1, 0.0f, ySize_d - 1.0f)) * xSize_d + x];

	float2 subtractVel = { (pR - pL) * 0.5f, (pT - pB) * 0.5f };


	u = u - subtractVel;
}

// applies force and add color dye to the particle field
__global__ void applyForce(float2* oldVel, float3* oldColor, float3 color, float2 F, float2 pos, float dt)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	float e = expf(-((x - pos.x) * (x - pos.x) + (y - pos.y) * (y - pos.y)) / devConstants.radius);
	float2 uF = F * dt * e;
	
	float2& u = oldVel[y * xSize_d + x];
	float3& c = oldColor[y * xSize_d + x];
	
	u = u + uF;
	c += color * e;
}


// applies vorticity to velocity field
__global__ void computeVorticity(float2* oldVel, float *vField, float dt)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	float2 vL, vR, vB, vT;


	vL = oldVel[y * xSize_d + int(CLAMP(x - 1, 0.0f, xSize_d - 1.0f))];
	vR = oldVel[y * xSize_d + int(CLAMP(x + 1, 0.0f, xSize_d - 1.0f))];
	vB = oldVel[int(CLAMP(y - 1, 0.0f, ySize_d - 1.0f)) * xSize_d + x];
	vT = oldVel[int(CLAMP(y + 1, 0.0f, ySize_d - 1.0f)) * xSize_d + x];

	vField[y * xSize_d + x] = 0.5f * (vR.y - vL.y - vT.x + vB.x);

	__syncthreads();



	float vortL = vField[y * xSize_d + int(CLAMP(x - 1, 0.0f, xSize_d - 1.0f))];
	float vortR = vField[y * xSize_d + int(CLAMP(x + 1, 0.0f, xSize_d - 1.0f))];
	float vortB = vField[int(CLAMP(y - 1, 0.0f, ySize_d - 1.0f)) * xSize_d + x];
	float vortT = vField[int(CLAMP(y + 1, 0.0f, ySize_d - 1.0f)) * xSize_d + x];
	float vortC = vField[y * xSize_d + x];

	float2 v = { (abs(vortT) - abs(vortB)) * 0.5f, (abs(vortL) - abs(vortR)) * 0.5f };

	float length = sqrtf(v.x * v.x + v.y * v.y) + 0.001f;

	v = v * (1.0f / length);

	v = v * vortC * devConstants.vorticity;
	float2 newVelValue = oldVel[y * xSize_d + x] + v * dt;
	
	__syncthreads();

	oldVel[y * xSize_d + x] = newVelValue;
}

// adds flashlight effect near the mouse position
__global__ void applyBloom(uchar4* colorField, int xpos, int ypos)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;


	float e = devConstants.bloomIntense * expf(-((x - xpos) * (x - xpos) + (y - ypos) * (y - ypos) + 1.0f) / (devConstants.radius * devConstants.radius));

	unsigned char R = colorField[y * xSize_d + x].x;
	unsigned char G = colorField[y * xSize_d + x].y;
	unsigned char B = colorField[y * xSize_d + x].z;

	float maxval = fmaxf(R, fmaxf(G, B));

	colorField[y * xSize_d + x] = make_uchar4(fminf(255.0f, R + maxval * e), fminf(255.0f, G + maxval * e), fminf(255.0f, B + maxval * e), 255);
}

__global__ void writeToTexture(cudaSurfaceObject_t surface, uchar4* colorField)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x < xSize_d && y < ySize_d) {
		surf2Dwrite(colorField[y * xSize_d + x], surface, x*sizeof(uchar4), y);
	}
}

// main function, calls vorticity -> diffusion -> force -> pressure -> project -> advect -> paint -> bloom
void computeField(float dt, int x1pos, int y1pos, int x2pos, int y2pos, bool isPressed)
{
	dim3 threadsPerBlock(sConfig.xThreads, sConfig.yThreads);
	dim3 numBlocks(xSize / threadsPerBlock.x, ySize / threadsPerBlock.y);

	for (int i = 0; i < 8; i++)
	{
		averageKernelTimes[i] = averageKernelTimes[i] * g_totalFrames;
	}

	g_totalFrames++;

	double startTime = glfwGetTime();
	double endKernelTime = 0;

	// advect
	advect << <numBlocks, threadsPerBlock >> > (velocityField, dt);
	
	advect << <numBlocks, threadsPerBlock >> > (dyeColorField, velocityField, dt);
	CUDA_CALL(cudaDeviceSynchronize());
	endKernelTime = glfwGetTime() - startTime;
	averageKernelTimes[0] += endKernelTime;
	startTime = glfwGetTime();

	// curls and vortisity
	computeVorticity <<<numBlocks, threadsPerBlock >> > (velocityField, vorticityField, dt);
	CUDA_CALL(cudaDeviceSynchronize());
	endKernelTime = glfwGetTime() - startTime;
	averageKernelTimes[1] += endKernelTime;
	startTime = glfwGetTime();

	// diffuse velocity and color
	diffuseVel << <numBlocks, threadsPerBlock, 0, stream_0 >> > (velocityField, dt);
	diffuseCol << <numBlocks, threadsPerBlock, 0, stream_1 >> > (dyeColorField, dt);
	CUDA_CALL(cudaStreamSynchronize(stream_0));
	CUDA_CALL(cudaStreamSynchronize(stream_1));
	endKernelTime = glfwGetTime() - startTime;
	averageKernelTimes[2] += endKernelTime;
	startTime = glfwGetTime();

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
		CUDA_CALL(cudaStreamSynchronize(stream_0));
		CUDA_CALL(cudaStreamSynchronize(stream_1));
		applyForce << <numBlocks, threadsPerBlock >> > (velocityField, dyeColorField, currentColor, F, pos, dt);

	}
	else
	{
		timeSincePress += dt;
	}
	CUDA_CALL(cudaDeviceSynchronize());
	endKernelTime = glfwGetTime() - startTime;
	averageKernelTimes[3] += endKernelTime;
	startTime = glfwGetTime();

	// compute pressure
	computeDivergence << <numBlocks, threadsPerBlock >> > (divergenceField, velocityField);
	computePressureImpl << <numBlocks, threadsPerBlock >> > (divergenceField, pressureField, dt);
	CUDA_CALL(cudaDeviceSynchronize());
	endKernelTime = glfwGetTime() - startTime;
	averageKernelTimes[4] += endKernelTime;
	startTime = glfwGetTime();

	// project
	project <<<numBlocks, threadsPerBlock >> > (velocityField, pressureField);
	CUDA_CALL(cudaMemsetAsync(pressureField, 0.0f, xSize * ySize * sizeof(float)));
	CUDA_CALL(cudaDeviceSynchronize());
	endKernelTime = glfwGetTime() - startTime;
	averageKernelTimes[5] += endKernelTime;
	startTime = glfwGetTime();

	// paint image
	paint <<<numBlocks, threadsPerBlock >> > (textureColorField, dyeColorField);
	CUDA_CALL(cudaDeviceSynchronize());
	endKernelTime = glfwGetTime() - startTime;
	averageKernelTimes[6] += endKernelTime;
	startTime = glfwGetTime();

	// apply bloom in mouse pos
	if (config.bloomEnabled && timeSincePress < 5.0f)
	{
		applyBloom <<<numBlocks, threadsPerBlock >> > (textureColorField, x2pos, y2pos);
		CUDA_CALL(cudaDeviceSynchronize());
		endKernelTime = glfwGetTime() - startTime;
		averageKernelTimes[7] += endKernelTime;
	}

	writeToTexture << <numBlocks, threadsPerBlock >> > (surfObj, textureColorField);

	for (int i = 0; i < 8; i++)
	{
		averageKernelTimes[i] = averageKernelTimes[i] / g_totalFrames;
	}

	CUDA_CALL(cudaDeviceSynchronize());
}