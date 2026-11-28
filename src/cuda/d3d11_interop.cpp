#include "d3d11_interop.h"

#include <d3d11.h>
#include <dxgi.h>
#include <wrl/client.h>
#include <cuda_runtime.h>
#include <cuda_d3d11_interop.h>
#include <SDL3/SDL.h>

bool D3D11InteropTarget::init(SDL_Renderer* renderer, SDL_Texture* texture) {
    release();  // Allows calling init() again, e.g. after SDL recreated its textures

    // Both pointers are owned by SDL: no AddRef was done for us, so they must not be Released.
    auto* device = static_cast<ID3D11Device*>(SDL_GetPointerProperty(
        SDL_GetRendererProperties(renderer), SDL_PROP_RENDERER_D3D11_DEVICE_POINTER, nullptr));
    auto* d3dTex = static_cast<ID3D11Texture2D*>(SDL_GetPointerProperty(
        SDL_GetTextureProperties(texture), SDL_PROP_TEXTURE_D3D11_TEXTURE_POINTER, nullptr));
    if (device == nullptr || d3dTex == nullptr) {
        SDL_Log("Interop: no D3D11 device or texture (renderer is not direct3d11?)");
        return false;
    }

    D3D11_TEXTURE2D_DESC desc{};
    d3dTex->GetDesc(&desc);
    // Expected: 768x768, format 28 (R8G8B8A8_UNORM), usage 0 (DEFAULT), bind flags 8 (SHADER_RESOURCE)
    SDL_Log("Interop texture: %ux%u, format %d, usage %d, bind flags %u",
        desc.Width, desc.Height, static_cast<int>(desc.Format), static_cast<int>(desc.Usage), desc.BindFlags);
    if (desc.Usage == D3D11_USAGE_DYNAMIC) {
        SDL_Log("Interop: dynamic (streaming) textures cannot be registered with CUDA");
        return false;
    }

    // CUDA and D3D11 must run on the same adapter. These two COM pointers are ours; ComPtr Releases them.
    Microsoft::WRL::ComPtr<IDXGIDevice> dxgiDevice;
    Microsoft::WRL::ComPtr<IDXGIAdapter> adapter;
    if (FAILED(device->QueryInterface(IID_PPV_ARGS(&dxgiDevice)))
        || FAILED(dxgiDevice->GetAdapter(&adapter))) {
        SDL_Log("Interop: could not query the D3D11 adapter");
        return false;
    }
    int d3dCudaDevice = -1;
    int currentDevice = -1;
    // Fails when the adapter has no CUDA device, e.g. D3D11 runs on an integrated GPU
    const cudaError_t deviceResult = cudaD3D11GetDevice(&d3dCudaDevice, adapter.Get());
    if (deviceResult != cudaSuccess || cudaGetDevice(&currentDevice) != cudaSuccess
        || d3dCudaDevice != currentDevice) {
        cudaGetLastError();  // Clear the error so a later CUDA_CHECK does not report it
        SDL_Log("Interop: D3D11 adapter is not the current CUDA device (%s)", cudaGetErrorString(deviceResult));
        return false;
    }

    const cudaError_t registerResult = cudaGraphicsD3D11RegisterResource(
        &resource_, d3dTex, cudaGraphicsRegisterFlagsNone);
    if (registerResult != cudaSuccess) {
        cudaGetLastError();
        resource_ = nullptr;
        SDL_Log("Interop: register failed: %s", cudaGetErrorString(registerResult));
        return false;
    }
    // Every frame overwrites the whole texture, so CUDA need not preserve its previous contents.
    cudaGraphicsResourceSetMapFlags(resource_, cudaGraphicsMapFlagsWriteDiscard);
    return true;
}

void D3D11InteropTarget::release() {
    if (resource_ == nullptr) {
        return;
    }
    unmap();
    cudaGraphicsUnregisterResource(resource_);
    resource_ = nullptr;
}

cudaArray* D3D11InteropTarget::map() {
    if (resource_ == nullptr || mapped_) {
        return nullptr;
    }
    if (cudaGraphicsMapResources(1, &resource_, 0) != cudaSuccess) {  // stream 0: the default stream
        cudaGetLastError();
        return nullptr;
    }
    mapped_ = true;
    cudaArray* array = nullptr;
    if (cudaGraphicsSubResourceGetMappedArray(&array, resource_, 0, 0) != cudaSuccess) {  // array index 0, mip level 0
        cudaGetLastError();
        unmap();
        return nullptr;
    }
    return array;
}

void D3D11InteropTarget::unmap() {
    if (mapped_) {
        cudaGraphicsUnmapResources(1, &resource_, 0);
        mapped_ = false;
    }
}
