#include <string.h>
#include <math.h>
#include <stdlib.h>
#include <string>

#include <SDL3/SDL.h>
#include <SDL3/SDL_main.h>

#include "cuda/cuda_fluid_solver.h"

std::vector<uint8_t> gFrameBuffer;
SDL_Window* gSDLWindow;
SDL_Renderer* gSDLRenderer;
SDL_Texture* gSimTexture;
static int gDone;
static bool gScreenshotRequested;
const int WINDOW_WIDTH = 1024 / 2;
const int WINDOW_HEIGHT = 1024 / 2;
const int SIM_WIDTH = 64;
const int SIM_HEIGHT = 64;
const int SIM_DEPTH = 32;

CudaFluidSolver solver = CudaFluidSolver(SIM_WIDTH, SIM_HEIGHT, SIM_DEPTH, 1.0f / SIM_WIDTH, 1.0f);
std::vector<float> gDisplaySlice(SIM_WIDTH * SIM_HEIGHT);

bool SMOOTH_RENDERING = true;
bool SHOW_DIV = false;
bool PAUSE = false;
int SLICE_DEPTH = SIM_DEPTH / 2;

enum class DisplayQuantity {
    Density,
    Divergence
};
DisplayQuantity gDisplayQuantity = DisplayQuantity::Density;

SDL_FRect calDstRect(int outputWidth, int outputHeight, int simWidth, int simHeight) {
    const float scaleX = static_cast<float>(outputWidth) / simWidth;
    const float scaleY = static_cast<float>(outputHeight) / simHeight;
    const float scale = std::min(scaleX, scaleY);
    const float width  = simWidth  * scale;
    const float height = simHeight * scale;

    SDL_FRect dst_rect;
    dst_rect.x = (outputWidth  - width)  * 0.5f;
    dst_rect.y = (outputHeight - height) * 0.5f;
    dst_rect.w = width;
    dst_rect.h = height;
    return dst_rect;
}

bool update() {
    SDL_Event e;
    if (SDL_PollEvent(&e)) {
        if (e.type == SDL_EVENT_QUIT) {
            return false;
        }
        if (e.type == SDL_EVENT_KEY_UP && e.key.key == SDLK_ESCAPE) {
            return false;
        }
        if (e.type == SDL_EVENT_KEY_UP && e.key.key == SDLK_SPACE) {
            PAUSE = !PAUSE;
            return true;
        }
        if (e.type == SDL_EVENT_KEY_UP && e.key.key == SDLK_S) {
            SMOOTH_RENDERING = !SMOOTH_RENDERING;
            SDL_SetTextureScaleMode(gSimTexture, SMOOTH_RENDERING ? SDL_SCALEMODE_LINEAR : SDL_SCALEMODE_NEAREST);
            return true;
        }
        if (e.type == SDL_EVENT_KEY_UP && e.key.key == SDLK_D) {
            gDisplayQuantity = (gDisplayQuantity == DisplayQuantity::Divergence) ? DisplayQuantity::Density : DisplayQuantity::Divergence;
            return true;
        }
        if (e.type == SDL_EVENT_KEY_UP && e.key.key == SDLK_P) {
            gScreenshotRequested = true;
            return true;
        }
        if (e.type == SDL_EVENT_KEY_DOWN && e.key.key == SDLK_UP) {
            SLICE_DEPTH++;
            SLICE_DEPTH = std::clamp(SLICE_DEPTH, 0, SIM_DEPTH - 1);
        }
        if (e.type == SDL_EVENT_KEY_DOWN && e.key.key == SDLK_DOWN) {
            SLICE_DEPTH--;
            SLICE_DEPTH = std::clamp(SLICE_DEPTH, 0, SIM_DEPTH - 1);
        }
    }
    return true;
}



void render(Uint64 aTicks) {
    const int nx = solver.nx();
    const int ny = solver.ny();
    if (gDisplayQuantity == DisplayQuantity::Density) {
        solver.copyDensitySlice(SLICE_DEPTH, gDisplaySlice.data(), gDisplaySlice.size());
    } else {
        solver.copyDivergenceSlice(SLICE_DEPTH, gDisplaySlice.data(), gDisplaySlice.size());
    }

    for (int j = 0, c = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i, ++c) {
            const float value = gDisplaySlice[c];
            float intensity;
            if (gDisplayQuantity == DisplayQuantity::Density) {
                intensity = normalize(value, 0.0f, 1.0f);
            } else {
                constexpr float divergenceRange = 1.0f;
                intensity = normalize(value, -divergenceRange, divergenceRange);
            }
            const uint8_t byteValue = toByte(intensity);
            gFrameBuffer[c * 4 + 0] = byteValue;
            gFrameBuffer[c * 4 + 1] = byteValue;
            gFrameBuffer[c * 4 + 2] = byteValue;
            gFrameBuffer[c * 4 + 3] = 255;
        }
    }

    if (!SDL_UpdateTexture(gSimTexture, nullptr, gFrameBuffer.data(), SIM_WIDTH * 4 * sizeof(uint8_t))) {
        SDL_Log("Failed to update texture: %s", SDL_GetError());
    }

    int out_width = 0, out_height = 0;
    SDL_GetCurrentRenderOutputSize(gSDLRenderer, &out_width, &out_height);
    SDL_FRect dst_rect = calDstRect(out_width, out_height, SIM_WIDTH, SIM_HEIGHT);
    SDL_SetRenderDrawColor(gSDLRenderer, 0, 0, 0, 255);
    SDL_RenderClear(gSDLRenderer);
    SDL_RenderTexture(gSDLRenderer, gSimTexture, nullptr, &dst_rect); // use all texture

    if (gScreenshotRequested) {
        gScreenshotRequested = false;
        SDL_Surface* screenshot = SDL_RenderReadPixels(gSDLRenderer, nullptr);
        if (!screenshot) {
            SDL_Log("Failed to capture screenshot: %s", SDL_GetError());
        } else {
            const char* basePath = SDL_GetBasePath();
            if (!basePath) {
                SDL_Log("Failed to get application directory: %s", SDL_GetError());
            } else {
                std::string screenshotPath;
                for (Uint64 index = 1;; ++index) {
                    screenshotPath = std::string(basePath) + "screenshot_" + std::to_string(index) + ".bmp";
                    SDL_PathInfo pathInfo;
                    if (!SDL_GetPathInfo(screenshotPath.c_str(), &pathInfo)) {
                        break;
                    }
                }

                if (!SDL_SaveBMP(screenshot, screenshotPath.c_str())) {
                    SDL_Log("Failed to save screenshot to %s: %s", screenshotPath.c_str(), SDL_GetError());
                } else {
                    SDL_Log("Screenshot saved to %s", screenshotPath.c_str());
                }
            }
            SDL_DestroySurface(screenshot);
        }
    }

    SDL_RenderPresent(gSDLRenderer); // Future need: change to SDL_LockTexture
}

void loop() {
    static Uint64 prev_ticks = SDL_GetTicks();
    static double accumulator = 0.0;
    constexpr double simulation_dt = 1.0 / 60.0;
    constexpr int max_step_per_frame = 4;

    const Uint64 cur_ticks = SDL_GetTicks();
    double frame_dt = static_cast<double>(cur_ticks - prev_ticks) / 1000.0;
    prev_ticks = cur_ticks;

    frame_dt = std::min(frame_dt, 0.1);
    if (!update()) {
        gDone = 1;
        return;
    }

    if (PAUSE) {
        accumulator = 0.0;
    } else {
        accumulator += frame_dt;
        int steps = 0;
        while (accumulator >= simulation_dt && steps < max_step_per_frame) {
            solver.step(static_cast<float>(simulation_dt));
            accumulator -= simulation_dt;
            steps++;
        }
        if (steps == max_step_per_frame) {
            accumulator = 0.0;
        }
    }

    render(cur_ticks);
}

int main(int argc, char** argv) {
    if (!SDL_Init(SDL_INIT_VIDEO | SDL_INIT_EVENTS)) {
        return -1;
    }

    gFrameBuffer = std::vector<uint8_t>(SIM_WIDTH * SIM_HEIGHT * 4); // ABGR8888
    gSDLWindow = SDL_CreateWindow("MiniFluid", WINDOW_WIDTH, WINDOW_HEIGHT, 0);
    gSDLRenderer = SDL_CreateRenderer(gSDLWindow, nullptr);
    if (!SDL_SetRenderVSync(gSDLRenderer, 1)) {
        SDL_Log("Failed to enable VSync: %s", SDL_GetError());
    }
    gSimTexture = SDL_CreateTexture(gSDLRenderer, SDL_PIXELFORMAT_ABGR8888, SDL_TEXTUREACCESS_STREAMING, SIM_WIDTH, SIM_HEIGHT);
    if (!gSDLWindow || !gSDLRenderer || !gSimTexture){
        SDL_Log("Initialization failed: %s", SDL_GetError());
    }
    if (!SDL_SetTextureScaleMode(
        gSimTexture, 
        SMOOTH_RENDERING ? SDL_SCALEMODE_LINEAR : SDL_SCALEMODE_NEAREST)) {
        SDL_Log("Failed to set scale mode: %s", SDL_GetError());
    }

    gDone = 0;
    while(!gDone) {
        loop();
    }

    SDL_DestroyTexture(gSimTexture);
    SDL_DestroyRenderer(gSDLRenderer);
    SDL_DestroyWindow(gSDLWindow);
    SDL_Quit();

    return 0;
}