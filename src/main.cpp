#include <string.h>
#include <math.h>
#include <stdlib.h>
#include <exception>
#include <string>

#include <SDL3/SDL.h>
#include <SDL3/SDL_main.h>

#include <imgui.h>
#include <imgui_impl_sdl3.h>
#include <imgui_impl_sdlrenderer3.h>

#include "camera.h"
#include "scene_presets.h"
#include "cuda/cuda_fluid_solver.h"
#include "cuda/cuda_renderer.h"
#include "profile/profiler.h"
#include "profile/profiler_ui.h"

std::vector<uint8_t> gFrameBuffer;
std::vector<uint8_t> gSliceBuffer;
SDL_Window* gSDLWindow;
SDL_Renderer* gSDLRenderer;
SDL_Texture* gRenderTexture;
SDL_Texture* gSliceTexture;
static int gDone;
static bool gScreenshotRequested;
static std::string gScreenshotPath; // Empty: next free screenshot_N.bmp
const int WINDOW_WIDTH = 768;
const int WINDOW_HEIGHT = 768;
const int RENDER_WIDTH = 768;
const int RENDER_HEIGHT = 768;
const int SIM_W = 128;
const int SIM_WIDTH = SIM_W;
const int SIM_HEIGHT = SIM_W * 1.5;
const int SIM_DEPTH = SIM_W;
int SLICE_DEPTH = SIM_DEPTH / 2;
constexpr double SIMULATION_DT = 1.0 / 60.0;

CudaFluidSolver solver = CudaFluidSolver(SIM_WIDTH, SIM_HEIGHT, SIM_DEPTH, 1.0f / SIM_WIDTH, 1.0f);
std::vector<float> gDisplaySlice(SIM_WIDTH * SIM_HEIGHT);
std::vector<ScenePreset> gPresets = makeScenePresets();
int gPresetIndex = 0;

bool SMOOTH_RENDERING = true;
bool PAUSE = false;

Camera* camera;
CudaRenderer* cudaRenderer;
bool gMouseLookActive;
bool gImGuiActive;          // ImGui is only set up for the interactive loop, not for --capture
std::string gOutputDirectory;
std::string gGpuName;

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

void updateProfilerMetadata() {
    profiler().setMetadata({
        {"scene", gPresets[gPresetIndex].name},
        {"grid", std::to_string(SIM_WIDTH) + "x" + std::to_string(SIM_HEIGHT) + "x" + std::to_string(SIM_DEPTH)},
        {"render", std::to_string(RENDER_WIDTH) + "x" + std::to_string(RENDER_HEIGHT)},
        {"dt", std::to_string(SIMULATION_DT)},
        {"pressure_cycles", std::to_string(solver.parameters().pressureCycles)},
        {"multigrid_levels", std::to_string(solver.multigridLevels())},
        {"gpu", gGpuName},
    });
}

void applyPreset(int index) {
    gPresetIndex = std::clamp(index, 0, static_cast<int>(gPresets.size()) - 1);
    const ScenePreset& preset = gPresets[gPresetIndex];
    solver.parameters() = preset.parameters;
    solver.setEmitters(preset.emitters);
    solver.reset();
    updateProfilerMetadata();
    SDL_Log("Scene %d: %s", gPresetIndex + 1, preset.name);
}

void OnInit() {
    const float h = solver.spacing();
    const Vector3 center(0.5f * SIM_WIDTH * h, 0.45f * SIM_HEIGHT * h, 0.5f * SIM_DEPTH * h);
    camera = new Camera();
    camera->setFovY(50.0f);
    camera->setPosition(center + Vector3(0.95f, 0.12f, 1.65f));
    camera->lookAt(center);
    cudaRenderer = new CudaRenderer(solver, *camera, {RENDER_WIDTH, RENDER_HEIGHT});
}

bool update(float deltaTime) {
    SDL_Event e;
    while (SDL_PollEvent(&e)) {
        if (gImGuiActive) {
            ImGui_ImplSDL3_ProcessEvent(&e);
        }
        if (e.type == SDL_EVENT_QUIT) {
            return false;
        }
        // While the cursor is over the profiler window, or a widget has keyboard focus,
        // the input belongs to ImGui. Releasing buttons is always handled below.
        const bool uiMouse = gImGuiActive && ImGui::GetIO().WantCaptureMouse;
        const bool uiKeyboard = gImGuiActive && ImGui::GetIO().WantCaptureKeyboard;
        LightingSettings& lighting = cudaRenderer->lighting();
        if (e.type == SDL_EVENT_KEY_UP && e.key.key == SDLK_F1) {
            profiler().setWindowVisible(!profiler().windowVisible());
        }
        if (e.type == SDL_EVENT_KEY_UP && !uiKeyboard) {
            switch (e.key.key) {
            case SDLK_ESCAPE:
                return false;
            case SDLK_SPACE:
                PAUSE = !PAUSE;
                break;
            case SDLK_M:
                SMOOTH_RENDERING = !SMOOTH_RENDERING;
                SDL_SetTextureScaleMode(gRenderTexture, SMOOTH_RENDERING ? SDL_SCALEMODE_LINEAR : SDL_SCALEMODE_NEAREST);
                SDL_SetTextureScaleMode(gSliceTexture, SMOOTH_RENDERING ? SDL_SCALEMODE_LINEAR : SDL_SCALEMODE_NEAREST);
                break;
            case SDLK_V:
                gDisplayQuantity = (gDisplayQuantity == DisplayQuantity::Divergence) ? DisplayQuantity::Density : DisplayQuantity::Divergence;
                break;
            case SDLK_P:
                gScreenshotRequested = true;
                break;
            case SDLK_R:
                solver.reset();
                break;
            case SDLK_L:
                lighting.lit = !lighting.lit;
                break;
            case SDLK_F:
                lighting.emission = !lighting.emission;
                break;
            case SDLK_G:
                lighting.floor = !lighting.floor;
                break;
            case SDLK_1:
            case SDLK_2:
            case SDLK_3:
            case SDLK_4:
                applyPreset(static_cast<int>(e.key.key - SDLK_1));
                break;
            default:
                break;
            }
        }
        if (e.type == SDL_EVENT_KEY_DOWN && !uiKeyboard && e.key.key == SDLK_UP) {
            SLICE_DEPTH++;
            SLICE_DEPTH = std::clamp(SLICE_DEPTH, 0, SIM_DEPTH - 1);
        }
        if (e.type == SDL_EVENT_KEY_DOWN && !uiKeyboard && e.key.key == SDLK_DOWN) {
            SLICE_DEPTH--;
            SLICE_DEPTH = std::clamp(SLICE_DEPTH, 0, SIM_DEPTH - 1);
        }
        if (e.type == SDL_EVENT_MOUSE_BUTTON_DOWN && !uiMouse
            && e.button.button == SDL_BUTTON_RIGHT) {
            gMouseLookActive = true;
        }
        if (e.type == SDL_EVENT_MOUSE_BUTTON_UP
            && e.button.button == SDL_BUTTON_RIGHT) {
            gMouseLookActive = false;
        }
        if (e.type == SDL_EVENT_MOUSE_MOTION && gMouseLookActive) {
            camera->processMouseMotion(e.motion.xrel, e.motion.yrel);
        }
        if (e.type == SDL_EVENT_MOUSE_WHEEL && !uiMouse) {
            const float direction = e.wheel.direction == SDL_MOUSEWHEEL_FLIPPED
                ? -1.0f
                : 1.0f;
            camera->handleScroll(direction * e.wheel.y, deltaTime);
        }
        if (e.type == SDL_EVENT_WINDOW_FOCUS_LOST) {
            gMouseLookActive = false;
        }
    }

    if (gImGuiActive && ImGui::GetIO().WantCaptureKeyboard) {
        return true;
    }
    const bool* keyboardState = SDL_GetKeyboardState(nullptr);
    const bool movingForward = keyboardState[SDL_SCANCODE_W];
    const bool movingBackward = keyboardState[SDL_SCANCODE_S];
    const bool movingLeft = keyboardState[SDL_SCANCODE_A];
    const bool movingRight = keyboardState[SDL_SCANCODE_D];
    const float movementDeltaTime =
        deltaTime * ((movingForward || movingBackward)
            && (movingLeft || movingRight) ? 0.70710678f : 1.0f);
    if (movingForward) {
        camera->handleKeyPress('w', movementDeltaTime);
    }
    if (movingBackward) {
        camera->handleKeyPress('s', movementDeltaTime);
    }
    if (movingLeft) {
        camera->handleKeyPress('a', movementDeltaTime);
    }
    if (movingRight) {
        camera->handleKeyPress('d', movementDeltaTime);
    }

    // Held keys rotate the sun: Left / Right for azimuth, Page Up / Page Down for elevation.
    constexpr float lightDegreesPerSecond = 60.0f;
    LightingSettings& lighting = cudaRenderer->lighting();
    if (keyboardState[SDL_SCANCODE_LEFT]) {
        lighting.lightAzimuth -= lightDegreesPerSecond * deltaTime;
    }
    if (keyboardState[SDL_SCANCODE_RIGHT]) {
        lighting.lightAzimuth += lightDegreesPerSecond * deltaTime;
    }
    if (keyboardState[SDL_SCANCODE_PAGEUP]) {
        lighting.lightElevation += lightDegreesPerSecond * deltaTime;
    }
    if (keyboardState[SDL_SCANCODE_PAGEDOWN]) {
        lighting.lightElevation -= lightDegreesPerSecond * deltaTime;
    }
    lighting.lightAzimuth = std::fmod(lighting.lightAzimuth + 360.0f, 360.0f);
    lighting.lightElevation = std::clamp(lighting.lightElevation, 5.0f, 89.0f);
    return true;
}

void saveScreenshot(const std::string& path) {
    SDL_Surface* screenshot = SDL_RenderReadPixels(gSDLRenderer, nullptr);
    if (!screenshot) {
        SDL_Log("Failed to capture screenshot: %s", SDL_GetError());
        return;
    }
    if (!SDL_SaveBMP(screenshot, path.c_str())) {
        SDL_Log("Failed to save screenshot to %s: %s", path.c_str(), SDL_GetError());
    } else {
        SDL_Log("Screenshot saved to %s", path.c_str());
    }
    SDL_DestroySurface(screenshot);
}

std::string nextScreenshotPath() {
    const char* basePath = SDL_GetBasePath();
    if (!basePath) {
        SDL_Log("Failed to get application directory: %s", SDL_GetError());
        return "screenshot.bmp";
    }
    std::string screenshotPath;
    for (Uint64 index = 1;; ++index) {
        screenshotPath = std::string(basePath) + "screenshot_" + std::to_string(index) + ".bmp";
        SDL_PathInfo pathInfo;
        if (!SDL_GetPathInfo(screenshotPath.c_str(), &pathInfo)) {
            return screenshotPath;
        }
    }
}

void drawUi() {
    const CpuScope scope(CpuStage::Ui);
    ImGui_ImplSDLRenderer3_NewFrame();
    ImGui_ImplSDL3_NewFrame();
    ImGui::NewFrame();
    if (profiler().windowVisible()) {
        ProfilerContext context;
        context.sceneName = gPresets[gPresetIndex].name;
        context.nx = solver.nx();
        context.ny = solver.ny();
        context.nz = solver.nz();
        context.multigridLevels = solver.multigridLevels();
        context.pressureCycles = solver.parameters().pressureCycles;
        context.simulationDt = SIMULATION_DT;
        context.paused = PAUSE;
        context.outputDirectory = gOutputDirectory;
        drawProfilerWindow(profiler(), context);
    }
    ImGui::Render();
    ImGui_ImplSDLRenderer3_RenderDrawData(ImGui::GetDrawData(), gSDLRenderer);
}

void render(Uint64 aTicks) {
    SDL_Texture* texture;
    int textureWidth;
    int textureHeight;
    if (gDisplayQuantity == DisplayQuantity::Density) {
        PROFILE_CPU(CpuStage::Render, cudaRenderer->render(gFrameBuffer.data(), gFrameBuffer.size()));
        const CpuScope upload(CpuStage::Upload);
        if (!SDL_UpdateTexture(gRenderTexture, nullptr, gFrameBuffer.data(), RENDER_WIDTH * 4 * sizeof(uint8_t))) {
            SDL_Log("Failed to update texture: %s", SDL_GetError());
        }
        texture = gRenderTexture;
        textureWidth = RENDER_WIDTH;
        textureHeight = RENDER_HEIGHT;
    } else {
        {
            const CpuScope slice(CpuStage::Render);
            const int nx = solver.nx();
            const int ny = solver.ny();
            solver.copyDivergenceSlice(SLICE_DEPTH, gDisplaySlice.data(), gDisplaySlice.size());
            // Flip vertically so that +y points up on screen.
            for (int j = 0; j < ny; ++j) {
                for (int i = 0; i < nx; ++i) {
                    constexpr float divergenceRange = 1.0f;
                    const float intensity = normalize(
                        gDisplaySlice[j * nx + i], -divergenceRange, divergenceRange);
                    const uint8_t byteValue = toByte(intensity);
                    const int c = (ny - 1 - j) * nx + i;
                    gSliceBuffer[c * 4 + 0] = byteValue;
                    gSliceBuffer[c * 4 + 1] = byteValue;
                    gSliceBuffer[c * 4 + 2] = byteValue;
                    gSliceBuffer[c * 4 + 3] = 255;
                }
            }
        }
        const CpuScope upload(CpuStage::Upload);
        if (!SDL_UpdateTexture(gSliceTexture, nullptr, gSliceBuffer.data(), SIM_WIDTH * 4 * sizeof(uint8_t))) {
            SDL_Log("Failed to update texture: %s", SDL_GetError());
        }
        texture = gSliceTexture;
        textureWidth = SIM_WIDTH;
        textureHeight = SIM_HEIGHT;
    }

    {
        const CpuScope present(CpuStage::Present);
        int out_width = 0, out_height = 0;
        SDL_GetCurrentRenderOutputSize(gSDLRenderer, &out_width, &out_height);
        SDL_FRect dst_rect = calDstRect(out_width, out_height, textureWidth, textureHeight);
        SDL_SetRenderDrawColor(gSDLRenderer, 0, 0, 0, 255);
        SDL_RenderClear(gSDLRenderer);
        SDL_RenderTexture(gSDLRenderer, texture, nullptr, &dst_rect); // use all texture
    }

    // Taken before the overlay is drawn, so screenshots never contain the profiler window.
    if (gScreenshotRequested) {
        gScreenshotRequested = false;
        saveScreenshot(gScreenshotPath.empty() ? nextScreenshotPath() : gScreenshotPath);
    }

    if (gImGuiActive) {
        drawUi();
    }

    PROFILE_CPU(CpuStage::Present, SDL_RenderPresent(gSDLRenderer)); // Future need: change to SDL_LockTexture
}

void updateWindowTitle(Uint64 ticks) {
    static Uint64 last_ticks = ticks;
    static int frames = 0;
    ++frames;
    if (ticks - last_ticks < 500) {
        return;
    }
    const double fps = frames * 1000.0 / static_cast<double>(ticks - last_ticks);
    const std::string title = "MiniFluid - " + std::string(gPresets[gPresetIndex].name)
        + " - " + std::to_string(static_cast<int>(fps + 0.5)) + " FPS"
        + (PAUSE ? " (paused)" : "")
        + (profiler().recording() ? " [REC]" : "");
    SDL_SetWindowTitle(gSDLWindow, title.c_str());
    last_ticks = ticks;
    frames = 0;
}

void loop() {
    static Uint64 prev_ticks = SDL_GetTicks();
    static double accumulator = 0.0;
    constexpr int max_step_per_frame = 4;

    FrameProfiler& frameProfiler = profiler();
    frameProfiler.beginFrame();

    const Uint64 cur_ticks = SDL_GetTicks();
    double frame_dt = static_cast<double>(cur_ticks - prev_ticks) / 1000.0;
    prev_ticks = cur_ticks;

    frame_dt = std::min(frame_dt, 0.1);
    bool running;
    PROFILE_CPU(CpuStage::Input, running = update(static_cast<float>(frame_dt)));
    if (!running) {
        gDone = 1;
        return;
    }

    int steps = 0;
    if (PAUSE) {
        accumulator = 0.0;
    } else {
        const CpuScope simulation(CpuStage::Simulation);
        if (frameProfiler.wantsDiagnostics()) {
            solver.requestDiagnostics();
        }
        accumulator += frame_dt;
        while (accumulator >= SIMULATION_DT && steps < max_step_per_frame) {
            solver.step(static_cast<float>(SIMULATION_DT));
            accumulator -= SIMULATION_DT;
            steps++;
        }
        if (steps == max_step_per_frame) {
            accumulator = 0.0;
        }
    }

    render(cur_ticks);
    updateWindowTitle(cur_ticks);

    // The render readback has already waited for the GPU, so this copy does not stall.
    SolverDiagnostics diagnostics;
    if (solver.fetchDiagnostics(diagnostics)) {
        frameProfiler.setDiagnostics(diagnostics);
    }
    frameProfiler.endFrame(steps);
}

// Offline capture: advance a fixed number of steps, render once, save, and exit.
void capture(int steps, const std::string& path) {
    constexpr float simulation_dt = static_cast<float>(SIMULATION_DT);
    const Uint64 start = SDL_GetTicksNS();
    for (int step = 0; step < steps; ++step) {
        solver.step(simulation_dt);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    const Uint64 simulated = SDL_GetTicksNS();
    render(0);
    const Uint64 rendered = SDL_GetTicksNS();
    SDL_Log("Captured %d steps: %.2f ms/step, first render %.2f ms",
        steps, (simulated - start) / 1.0e6 / std::max(steps, 1), (rendered - simulated) / 1.0e6);

    constexpr int timed_renders = 10;
    for (int i = 0; i < timed_renders; ++i) {
        render(0);
    }
    SDL_Log("Average render %.2f ms",
        (SDL_GetTicksNS() - rendered) / 1.0e6 / timed_renders);

    gScreenshotPath = path;
    gScreenshotRequested = true;
    render(0);
}

static int run(int argc, char** argv);

int main(int argc, char** argv) {
    // CUDA_CHECK throws on failure; report it instead of terminating silently.
    try {
        return run(argc, argv);
    } catch (const std::exception& e) {
        SDL_Log("Fatal error: %s", e.what());
        SDL_ShowSimpleMessageBox(SDL_MESSAGEBOX_ERROR, "MiniFluid", e.what(), nullptr);
        return 1;
    }
}

static int run(int argc, char** argv) {
    int initialPreset = 0;
    int captureSteps = -1;
    std::string capturePath = "capture.bmp";
    bool unlit = false;
    bool emission = false;
#ifdef PROFILE
    bool showProfiler = true;
#else
    bool showProfiler = false;
#endif
    std::string profileRecordPath;
    for (int i = 1; i < argc; ++i) {
        const std::string argument = argv[i];
        if (argument == "--preset" && i + 1 < argc) {
            initialPreset = std::atoi(argv[++i]) - 1;
        } else if (argument == "--capture" && i + 1 < argc) {
            captureSteps = std::atoi(argv[++i]);
            if (i + 1 < argc && argv[i + 1][0] != '-') {
                capturePath = argv[++i];
            }
        } else if (argument == "--unlit") {
            unlit = true;
        } else if (argument == "--emission") {
            emission = true;
        } else if (argument == "--profile") {
            showProfiler = true;
        } else if (argument == "--profile-record" && i + 1 < argc) {
            profileRecordPath = argv[++i];
        }
    }

    if (!SDL_Init(SDL_INIT_VIDEO | SDL_INIT_EVENTS)) {
        return -1;
    }

    gFrameBuffer = std::vector<uint8_t>(RENDER_WIDTH * RENDER_HEIGHT * 4); // ABGR8888
    gSliceBuffer = std::vector<uint8_t>(SIM_WIDTH * SIM_HEIGHT * 4);
    gSDLWindow = SDL_CreateWindow("MiniFluid", WINDOW_WIDTH, WINDOW_HEIGHT, 0);
    gSDLRenderer = SDL_CreateRenderer(gSDLWindow, nullptr);
    if (!SDL_SetRenderVSync(gSDLRenderer, 0)) {
        SDL_Log("Failed to disable VSync: %s", SDL_GetError());
    }
    gRenderTexture = SDL_CreateTexture(gSDLRenderer, SDL_PIXELFORMAT_ABGR8888, SDL_TEXTUREACCESS_STREAMING, RENDER_WIDTH, RENDER_HEIGHT);
    gSliceTexture = SDL_CreateTexture(gSDLRenderer, SDL_PIXELFORMAT_ABGR8888, SDL_TEXTUREACCESS_STREAMING, SIM_WIDTH, SIM_HEIGHT);
    if (!gSDLWindow || !gSDLRenderer || !gRenderTexture || !gSliceTexture){
        SDL_Log("Initialization failed: %s", SDL_GetError());
    }
    for (SDL_Texture* texture : {gRenderTexture, gSliceTexture}) {
        if (!SDL_SetTextureScaleMode(
            texture,
            SMOOTH_RENDERING ? SDL_SCALEMODE_LINEAR : SDL_SCALEMODE_NEAREST)) {
            SDL_Log("Failed to set scale mode: %s", SDL_GetError());
        }
    }

    if (const char* basePath = SDL_GetBasePath()) {
        gOutputDirectory = basePath;
    }
    cudaDeviceProp deviceProperties{};
    int device = 0;
    if (cudaGetDevice(&device) == cudaSuccess
        && cudaGetDeviceProperties(&deviceProperties, device) == cudaSuccess) {
        gGpuName = deviceProperties.name;
    }

    OnInit();
    applyPreset(initialPreset);
    cudaRenderer->lighting().lit = !unlit;
    cudaRenderer->lighting().emission = emission;

    if (captureSteps >= 0) {
        capture(captureSteps, capturePath);
    } else {
        IMGUI_CHECKVERSION();
        ImGui::CreateContext();
        ImGui::GetIO().IniFilename = nullptr;
        ImGui::StyleColorsDark();
        ImGui_ImplSDL3_InitForSDLRenderer(gSDLWindow, gSDLRenderer);
        ImGui_ImplSDLRenderer3_Init(gSDLRenderer);
        gImGuiActive = true;

        profiler().setWindowVisible(showProfiler);
        if (!profileRecordPath.empty()) {
            if (profiler().startRecording(profileRecordPath)) {
                SDL_Log("Recording profile to %s", profileRecordPath.c_str());
            } else {
                SDL_Log("Failed to open profile recording %s", profileRecordPath.c_str());
            }
        }

        gDone = 0;
        while(!gDone) {
            loop();
        }

        profiler().stopRecording();
        gImGuiActive = false;
        ImGui_ImplSDLRenderer3_Shutdown();
        ImGui_ImplSDL3_Shutdown();
        ImGui::DestroyContext();
    }
    profiler().releaseGpuResources();

    SDL_DestroyTexture(gSliceTexture);
    SDL_DestroyTexture(gRenderTexture);
    SDL_DestroyRenderer(gSDLRenderer);
    SDL_DestroyWindow(gSDLWindow);
    delete cudaRenderer;
    delete camera;
    SDL_Quit();

    return 0;
}
