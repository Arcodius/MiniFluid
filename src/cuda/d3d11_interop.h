#pragma once

struct SDL_Renderer;
struct SDL_Texture;
struct cudaGraphicsResource;
struct cudaArray;

// Lets CUDA write directly into the D3D11 texture behind an SDL texture, so the frame never leaves the GPU.
// Requires the SDL "direct3d11" renderer, a non-streaming texture (USAGE_DEFAULT) and CUDA running on the same adapter as D3D11.
class D3D11InteropTarget {
public:
    // Returns false (and stays invalid) when interop is not possible; the caller falls back to the readback path.
    bool init(SDL_Renderer* renderer, SDL_Texture* texture);
    // Unregisters the texture. Must run before the SDL texture or renderer is destroyed.
    void release();
    bool valid() const { return resource_ != nullptr; }

    // Hands the texture to CUDA. Returns nullptr on failure. D3D must not touch the texture until unmap().
    cudaArray* map();
    // Hands the texture back to D3D. Must run before SDL_RenderTexture draws it.
    void unmap();
private:
    cudaGraphicsResource* resource_ = nullptr;
    bool mapped_ = false;
};
