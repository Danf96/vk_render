// vulkan_guide.h : Include file for standard system include files,
// or project specific include files.

#pragma once

#include <vk_types.h>
#include "vk_descriptors.h"
#include "vk_loader.h"
#include "vk_materials.h"
#include "camera.h"

struct DeletionQueue
{
    std::deque<std::function<void()>> deletors;

    void push_function(std::function<void()> && function) {
        deletors.push_back(function);
    }

    void flush() 
    {
        for (auto it = deletors.rbegin(); it != deletors.rend(); it++)
        {
            (*it)();
        }

        deletors.clear();
    }
};

struct FrameData 
{
    VkCommandPool _commandPool;
    VkCommandBuffer _mainCommandBuffer;

    VkSemaphore _swapchainSemaphore;
    VkSemaphore _renderSemaphore;
    VkFence _renderFence;

    DeletionQueue _deletionQueue;
    DescriptorAllocatorGrowable _frameDescriptors;
};

constexpr static uint32_t FRAME_OVERLAP = 2;

struct ComputePushConstants
{
    glm::vec4 data1;
    glm::vec4 data2;
    glm::vec4 data3;
    glm::vec4 data4;
};

struct ComputeEffect
{
    const char* name;

    VkPipeline pipeline;
    VkPipelineLayout layout;

    ComputePushConstants data;
};

struct RenderObject
{
    uint32_t indexCount;
    uint32_t firstIndex;
    VkBuffer indexBuffer;

    MaterialInstance *material;

    glm::mat4 transform;
    VkDeviceAddress vertexBufferAddress;
};

struct DrawContext
{
    std::vector<RenderObject> OpaqueSurfaces;
    std::vector<RenderObject> TransparentSurfaces;
};

struct MeshNode : public Node
{
    std::shared_ptr<MeshAsset> mesh;
    void Draw(const glm::mat4 &topMatrix, DrawContext &ctx) override;
};

struct EngineStats
{
    float frametime;
    int triangle_count;
    int drawcall_count;
    float scene_update_time;
    float mesh_draw_time;
};



class VulkanEngine 
{
public:

    bool _isInitialized{ false };
    int _frameNumber {0};
    bool stop_rendering{ false };
    VkExtent2D _windowExtent{ 1700 , 900 };

    struct SDL_Window* _window{ nullptr };

    VkInstance _instance; // vulkan lib handle
    VkDebugUtilsMessengerEXT _debug_messenger; // debug output handle
    VkPhysicalDevice _chosenGPU; // default device chosen
    VkDevice _device; // device for commands
    VkSurfaceKHR _surface; // window surface

    VkSwapchainKHR _swapchain;
    VkFormat _swapchainImageFormat;

    std::vector<VkImage> _swapchainImages;
    std::vector<VkImageView> _swapchainImageViews;
    VkExtent2D _swapchainExtent;

    FrameData _frames[FRAME_OVERLAP];

    FrameData& get_current_frame() { return _frames[_frameNumber % FRAME_OVERLAP]; };

    VkQueue _graphicsQueue;
    uint32_t _graphicsQueueFamily;

    DeletionQueue _mainDeletionQueue;

    VmaAllocator _allocator;

    DescriptorAllocatorGrowable gDescriptorAllocator;

    VkDescriptorSet _drawImageDescriptors;
    VkDescriptorSetLayout _drawImageDescriptorLayout;

    // draw resources
    AllocatedImage _drawImage;
    AllocatedImage _depthImage;
    VkExtent2D _drawExtent;
    float renderScale = 1.f;
    bool resize_requested{};

    VkFence _immFence;
    VkCommandBuffer _immCommandBuffer;
    VkCommandPool _immCommandPool;

    std::vector<ComputeEffect> backgroundEffects;
    int currentBackgroundEffect{ 0 };
    VkPipelineLayout _gradientPipelineLayout;

    VkPipelineLayout _trianglePipelineLayout;
    VkPipeline _trianglePipeline;

    VkPipelineLayout _meshPipelineLayout;
    VkPipeline _meshPipeline;

    // mesh data
    GPUMeshBuffers rectangle;
    std::vector<std::shared_ptr<MeshAsset>> testMeshes;
    GPUSceneData sceneData;
    VkDescriptorSetLayout _gpuSceneDataDescriptorLayout;

    // texture data
    AllocatedImage _whiteImage;
    AllocatedImage _blackImage;
    AllocatedImage _grayImage;
    AllocatedImage _errorCheckerboardImage;
    VkSampler _defaultSamplerLinear;
    VkSampler _defaultSamplerNearest;
    VkDescriptorSetLayout _singleImageDescriptorLayout;
    
    // material data
    MaterialInstance defaultData;
    GLTFMetallic_Roughness metalRoughMaterial;

    DrawContext mainDrawContext;
    std::unordered_map<std::string, std::shared_ptr<Node>> loadedNodes;
    std::unordered_map<std::string, std::shared_ptr<LoadedGLTF>> loadedScenes;

    Camera mainCamera;

    EngineStats stats;

    void update_scene();


    static VulkanEngine& Get();

    //initializes everything in the engine
    void init();

    //shuts down the engine
    void cleanup();

    //draw loop
    void draw();

    //run main loop
    void run();

    void immediate_submit(std::function<void(VkCommandBuffer cmd)>&& function);

    GPUMeshBuffers uploadMesh(std::span<uint32_t> indices, std::span<Vertex> vertices);

    AllocatedImage create_image(VkExtent3D size, VkFormat format, VkImageUsageFlags usage, bool mipmapped = false);
    AllocatedImage create_image(void* data, VkExtent3D size, VkFormat format, VkImageUsageFlags usage, bool mipmapped = false);
    AllocatedBuffer create_buffer(size_t allocSize, VkBufferUsageFlags usage, VmaMemoryUsage memoryUsage);
    void destroy_image(const AllocatedImage& img);
    void destroy_buffer(const AllocatedBuffer &buffer);
private:

    void init_vulkan();
    void init_swapchain();
    void init_commands();
    void init_sync_structures();
    void init_default_data();
    void create_swapchain(uint32_t width, uint32_t height);
    void destroy_swapchain();
    void draw_background(VkCommandBuffer cmd);
    void draw_main(VkCommandBuffer cmd);
    void draw_geometry(VkCommandBuffer cmd);
    void draw_imgui(VkCommandBuffer cmd, VkImageView targetImageView);
    void init_descriptors();
    void init_pipelines();
    void init_background_pipelines();
    void init_imgui();
    void init_triangle_pipeline();
    void init_mesh_pipeline();
    void resize_swapchain();
};


