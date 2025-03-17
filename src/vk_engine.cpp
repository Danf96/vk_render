//> includes
#include "vk_engine.h"

#include <SDL3/SDL.h>
#include <SDL3/SDL_vulkan.h>

#include <vk_initializers.h>
#include <vk_types.h>

#include "VkBootstrap.h"

#include <chrono>
#include <thread>

#include "vk_images.h"
#include "vk_pipelines.h"


#define VMA_IMPLEMENTATION
#if 0
#define VMA_DEBUG_LOG_FORMAT(format, ...) do { \
       printf((format), __VA_ARGS__); \
       printf("\n"); \
   } while(false)
#endif
#include "vk_mem_alloc.h"

#include "imgui.h"
#include "backends/imgui_impl_sdl3.h"
#include "backends/imgui_impl_vulkan.h"

#include "glm/gtx/transform.hpp"

constexpr bool bUseValidationLayers = true;

VulkanEngine* loadedEngine = nullptr;

bool is_visible(const RenderObject & obj, const glm::mat4 & viewProj)
{
    std::array<glm::vec3, 8> corners
    {
        glm::vec3 { 1, 1, 1 },
        glm::vec3 { 1, 1, -1 },
        glm::vec3 { 1, -1, 1 },
        glm::vec3 { 1, -1, -1 },
        glm::vec3 { -1, 1, 1 },
        glm::vec3 { -1, 1, -1 },
        glm::vec3 { -1, -1, 1 },
        glm::vec3 { -1, -1, -1 },
    };
    glm::mat4 matrix = viewProj * obj.transform;
    glm::vec3 min = { 1.5, 1.5, 1.5 };
    glm::vec3 max = { -1.5, -1.5, -1.5 };

    for ( auto &corner : corners)
    {
        // project each corner into clip space
        glm::vec4 v = matrix * glm::vec4(obj.bounds.origin + (corner * obj.bounds.extents), 1.f);

        // perspective correction
        v.x = v.x / v.w;
        v.y = v.y / v.w;
        v.z = v.z / v.w;

        min = glm::min(glm::vec3{ v.x, v.y, v.z }, min);
        max = glm::max(glm::vec3{ v.x, v.y, v.z }, max);
    }

    // check the clip space box is within the view
    if ( min.z > 1.f || max.z < 0.f || min.x > 1.f || max.x < -1.f || min.y > 1.f || max.y < -1.f )
    {
        return false;
    }
    else
    {
        return true;
    }
}


void MeshNode::Draw(const glm::mat4 & topMatrix, DrawContext & ctx)
{
    glm::mat4 nodeMatrix = topMatrix * worldTransform;

    for (auto &s : mesh->surfaces)
    {
        RenderObject def{};
        def.indexCount = s.count;
        def.firstIndex = s.startIndex;
        def.indexBuffer = mesh->meshBuffers.indexBuffer.buffer;
        def.material = &s.material->data;
        def.bounds = s.bounds;
        def.transform = nodeMatrix;
        def.vertexBufferAddress = mesh->meshBuffers.vertexBufferAddress;

        if ( s.material->data.passType == MaterialPass::Transparent )
        {
            ctx.TransparentSurfaces.push_back(def);
        }
        else
        {
            ctx.OpaqueSurfaces.push_back(def);
        }
    }

    // recurse down
    Node::Draw(topMatrix, ctx);
}

VulkanEngine& VulkanEngine::Get() { return *loadedEngine; }

void VulkanEngine::update_scene()
{
    auto start = std::chrono::system_clock::now();
    mainCamera.update();
    glm::mat4 view = mainCamera.getViewMatrix();

    mainDrawContext.OpaqueSurfaces.clear();
    mainDrawContext.TransparentSurfaces.clear();

    //loadedNodes["Suzanne"]->Draw(glm::mat4{ 1.f }, mainDrawContext);

    loadedScenes["structure"]->Draw(glm::mat4{ 1.f }, mainDrawContext);

    glm::mat4 projection = glm::perspectiveRH_ZO(glm::radians(45.f), static_cast<float>(_drawExtent.width) / static_cast<float>(_drawExtent.height), 10000.f, 0.1f);
    projection[1][1] *= -1;

    sceneData.view = view;
    sceneData.proj = projection;
    sceneData.viewproj = projection * view;

    sceneData.ambientColor = glm::vec4{ 0.1f };
    sceneData.sunlightColor = glm::vec4{ 1.f };
    sceneData.sunlightDirection = glm::vec4{ 0.3f, 1.f, 0.3f, 1.f };

    auto end = std::chrono::system_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    stats.scene_update_time = elapsed.count() / 1000.f;
}

void VulkanEngine::init()
{
    // only one engine initialization is allowed with the application.
    assert(loadedEngine == nullptr);
    loadedEngine = this;

    // We initialize SDL and create a window with it.
    SDL_Init(SDL_INIT_VIDEO);

    SDL_WindowFlags window_flags = static_cast<SDL_WindowFlags>(SDL_WINDOW_VULKAN | SDL_WINDOW_RESIZABLE);

    _window = SDL_CreateWindow(
        "Vulkan Engine",
        SDL_WINDOWPOS_UNDEFINED,
        SDL_WINDOWPOS_UNDEFINED,
        _windowExtent.width,
        _windowExtent.height,
        window_flags);

    init_vulkan();

    init_swapchain();

    init_commands();

    init_sync_structures();

    init_descriptors();

    init_pipelines();

    init_imgui();

    init_default_data();

    mainCamera.velocity = glm::vec3{ 0.f };
    mainCamera.position = glm::vec3{ 30.f, 0.f, -85.f };
    mainCamera.pitch = 0;
    mainCamera.yaw = 0;

    std::string structurePath = {"..\\..\\assets\\structure.glb"};
    auto structureFile = loadGltf(this, structurePath);

    assert(structureFile.has_value());

    loadedScenes["structure"] = *structureFile;

    // everything went fine
    _isInitialized = true;
}

void VulkanEngine::cleanup()
{
    // cleanup order should be opposite of initialization order
    if (_isInitialized) {
        // make sure GPU has stopped doing whatever it's doing
        vkDeviceWaitIdle(_device);

        loadedScenes.clear();

        for ( auto &frame : _frames )
        {
            frame._deletionQueue.flush();
        }

        _mainDeletionQueue.flush();
        

        destroy_swapchain();

        vkDestroySurfaceKHR(_instance, _surface, nullptr);
        vmaDestroyAllocator(_allocator);

        vkDestroyDevice(_device, nullptr);
        vkb::destroy_debug_utils_messenger(_instance, _debug_messenger);
        vkDestroyInstance(_instance, nullptr);
        SDL_DestroyWindow(_window);
    }

    // clear engine pointer
    loadedEngine = nullptr;
}

void VulkanEngine::draw()
{
    // wait until gpu has finished rendering previous frame, 1s timeout
    constexpr uint64_t one_second = 1'000'000'000;
    VK_CHECK(vkWaitForFences(_device, 1, &get_current_frame()._renderFence, true, one_second));

    get_current_frame()._deletionQueue.flush();
    get_current_frame()._frameDescriptors.clear_pools(_device);

    

    uint32_t swapchainImageIndex;
    VkResult e = vkAcquireNextImageKHR(_device, _swapchain, one_second, get_current_frame()._swapchainSemaphore, nullptr, &swapchainImageIndex);
    if (e == VK_ERROR_OUT_OF_DATE_KHR)
    {
        resize_requested = true;
        return;
    }
#if 1
    if (e == VK_SUBOPTIMAL_KHR)
    {
        resize_requested = true;
    }
#endif

    _drawExtent.width = static_cast<uint32_t>(std::min(_swapchainExtent.width, _drawImage.imageExtent.width) * renderScale);
    _drawExtent.height = static_cast<uint32_t>(std::min(_swapchainExtent.height, _drawImage.imageExtent.height) * renderScale);

    VK_CHECK(vkResetFences(_device, 1, &get_current_frame()._renderFence));

    // reset command buffer to begin recording again
    VK_CHECK(vkResetCommandBuffer(get_current_frame()._mainCommandBuffer, 0));

    VkCommandBuffer cmd = get_current_frame()._mainCommandBuffer;

    // begin command buffer recording, used exactly once
    VkCommandBufferBeginInfo cmdBeginInfo = vkinit::command_buffer_begin_info(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);
    VK_CHECK(vkBeginCommandBuffer(cmd, &cmdBeginInfo));

    vkutil::transition_image(cmd, _drawImage.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);
    draw_background(cmd);

    // make swapchain image into writeable mode before rendering
    vkutil::transition_image(cmd, _drawImage.image, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
    vkutil::transition_image(cmd, _depthImage.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL);


    draw_main(cmd);

    // transition draw image and swapchain image into correct transfer layouts
    vkutil::transition_image(cmd, _drawImage.image , VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
    vkutil::transition_image(cmd, _swapchainImages[swapchainImageIndex], VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);

    // copy draw image into swapchain
    vkutil::copy_image_to_image(cmd, _drawImage.image, _swapchainImages[swapchainImageIndex], _drawExtent, _swapchainExtent);

    // set swapchain image layout to attachment optimal
    vkutil::transition_image(cmd, _swapchainImages[swapchainImageIndex], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);

    // draw imgui into swapchain image
    draw_imgui(cmd, _swapchainImageViews[swapchainImageIndex]);

    // set swapchain image layout to present
    vkutil::transition_image(cmd, _swapchainImages[swapchainImageIndex], VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR);

    // finalize cmd buffer
    VK_CHECK(vkEndCommandBuffer(cmd));

    // prepare submission to queue, wait on presentsemaphore then signal rendersemaphore
    VkCommandBufferSubmitInfo cmdInfo = vkinit::command_buffer_submit_info(cmd);

    VkSemaphoreSubmitInfo waitInfo = vkinit::semaphore_submit_info(VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT_KHR, get_current_frame()._swapchainSemaphore);
    VkSemaphoreSubmitInfo signalInfo = vkinit::semaphore_submit_info(VK_PIPELINE_STAGE_2_ALL_GRAPHICS_BIT, get_current_frame()._renderSemaphore);

    VkSubmitInfo2 submit = vkinit::submit_info(&cmdInfo, &signalInfo, &waitInfo);

    // submit cmd buffer to queue and execute it
    // renderfence will block until graphic commands finish execution
    VK_CHECK(vkQueueSubmit2(_graphicsQueue, 1, &submit, get_current_frame()._renderFence));

    // prepare image to be rendered to window
    // wait on _renderSemaphore
    
    VkPresentInfoKHR presentInfo = vkinit::present_info();
    presentInfo.pSwapchains = &_swapchain;
    presentInfo.swapchainCount = 1;

    presentInfo.pWaitSemaphores = &get_current_frame()._renderSemaphore;
    presentInfo.waitSemaphoreCount = 1;

    presentInfo.pImageIndices = &swapchainImageIndex;

    VkResult presentResult = vkQueuePresentKHR(_graphicsQueue, &presentInfo);
    if (presentResult == VK_ERROR_OUT_OF_DATE_KHR)
    {
        resize_requested = true;
        return;
    }

    // increase frames drawn
    _frameNumber++;
}

void VulkanEngine::run()
{
    SDL_Event e;
    bool bQuit = false;

    // main loop
    while (!bQuit) {
        // begin clock
        auto start = std::chrono::system_clock::now();
        // Handle events on queue
        while (SDL_PollEvent(&e) != 0) {
            // close the window when user alt-f4s or clicks the X button
            if (e.type == SDL_QUIT) {
                bQuit = true;
            } 

            if (e.type == SDL_WINDOWEVENT) {
                if (e.window.event == SDL_WINDOWEVENT_MINIMIZED) {
                    stop_rendering = true;
                }
                if (e.window.event == SDL_WINDOWEVENT_RESTORED) {
                    stop_rendering = false;
                }
            }
            mainCamera.processSDLEvent(e);
            // sent SDL event to imgui for handling
            ImGui_ImplSDL2_ProcessEvent(&e);
        }

        // do not draw if we are minimized
        if (stop_rendering) {
            // throttle the speed to avoid the endless spinning
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            continue;
        }

        if (resize_requested)
        {
            resize_swapchain();
        }

        // imgui new frame
        ImGui_ImplVulkan_NewFrame();
        ImGui_ImplSDL2_NewFrame(_window);

        ImGui::NewFrame();

        if (ImGui::Begin("background")) {

            ImGui::SliderFloat("Render Scale", &renderScale, 0.3f, 1.f);

            ComputeEffect& selected = backgroundEffects[currentBackgroundEffect];

            ImGui::Text("Selected effect: ", selected.name);

            ImGui::SliderInt("Effect Index", &currentBackgroundEffect, 0, static_cast<int>(backgroundEffects.size() - 1));

            ImGui::InputFloat4("data1", reinterpret_cast<float*>(& selected.data.data1));
            ImGui::InputFloat4("data2", reinterpret_cast<float*>(& selected.data.data2));
            ImGui::InputFloat4("data3", reinterpret_cast<float*>(& selected.data.data3));
            ImGui::InputFloat4("data4", reinterpret_cast<float*>(& selected.data.data4));

            ImGui::End();
        }

        ImGui::Begin("Stats");

        ImGui::Text("frametime %f ms", stats.frametime);
        ImGui::Text("draw time %f ms", stats.mesh_draw_time);
        ImGui::Text("update time %f ms", stats.scene_update_time);
        ImGui::Text("triangles %i", stats.triangle_count);
        ImGui::Text("draws %i", stats.drawcall_count);
        ImGui::End();

        // make imgui calc internal draw structs
        ImGui::Render();

        update_scene();

        draw();
        
        // get clock again, compare with start clock
        auto end = std::chrono::system_clock::now();

        // convert to microseconds (integer), and then come back to milliseconds
        auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        stats.frametime = elapsed.count() / 1000.f;
    }
}

void VulkanEngine::immediate_submit(std::function<void(VkCommandBuffer cmd)>&& function)
{
    VK_CHECK(vkResetFences(_device, 1, &_immFence));
    VK_CHECK(vkResetCommandBuffer(_immCommandBuffer, 0));

    VkCommandBuffer cmd = _immCommandBuffer;

    VkCommandBufferBeginInfo cmdBeginInfo = vkinit::command_buffer_begin_info(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);

    VK_CHECK(vkBeginCommandBuffer(cmd, &cmdBeginInfo));

    function(cmd);

    VK_CHECK(vkEndCommandBuffer(cmd));

    VkCommandBufferSubmitInfo cmdInfo = vkinit::command_buffer_submit_info(cmd);
    VkSubmitInfo2 submit = vkinit::submit_info(&cmdInfo, nullptr, nullptr);

    // submit command buffer to queue and execute it
    // renderfence blocks until graphics command finish execution
    VK_CHECK(vkQueueSubmit2(_graphicsQueue, 1, &submit, _immFence));

    VK_CHECK(vkWaitForFences(_device, 1, &_immFence, true, 9'999'999'999));
}

GPUMeshBuffers VulkanEngine::uploadMesh(std::span<uint32_t> indices, std::span<Vertex> vertices)
{
    const size_t vertexBufferSize = vertices.size() * sizeof(Vertex);
    const size_t indexBufferSize = indices.size() * sizeof(uint32_t);

    GPUMeshBuffers newSurface{};

    // create vertex buffer
    newSurface.vertexBuffer = create_buffer(vertexBufferSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | 
                                                              VK_BUFFER_USAGE_TRANSFER_DST_BIT   | 
                                                              VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT, VMA_MEMORY_USAGE_GPU_ONLY);

    // find address of the vertex buffer
    VkBufferDeviceAddressInfo deviceAddressInfo{ .sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO };
    deviceAddressInfo.buffer = newSurface.vertexBuffer.buffer;
    newSurface.vertexBufferAddress = vkGetBufferDeviceAddress(_device, &deviceAddressInfo);

    // create index buffer
    newSurface.indexBuffer = create_buffer(indexBufferSize, VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_GPU_ONLY);

    // staging buffer used for cpu work before moving to gpu
    AllocatedBuffer staging = create_buffer(vertexBufferSize + indexBufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VMA_MEMORY_USAGE_CPU_ONLY);
    void* data = staging.allocation->GetMappedData();

    // copy vertex buffer
    std::memcpy(data, vertices.data(), vertexBufferSize);
    // copy index buffer (destination offset so nothing overwritten)
    std::memcpy(reinterpret_cast<char*>(data) + vertexBufferSize, indices.data(), indexBufferSize);

    immediate_submit([&](VkCommandBuffer cmd)
    {
            VkBufferCopy vertexCopy{};
            vertexCopy.size = vertexBufferSize;
            vkCmdCopyBuffer(cmd, staging.buffer, newSurface.vertexBuffer.buffer, 1, &vertexCopy);

            VkBufferCopy indexCopy{};
            indexCopy.srcOffset = vertexBufferSize;
            indexCopy.size = indexBufferSize;

            vkCmdCopyBuffer(cmd, staging.buffer, newSurface.indexBuffer.buffer, 1, &indexCopy);
    });
    
    destroy_buffer(staging);

    _mainDeletionQueue.push_function([=]() 
        { 
            vmaDestroyBuffer(_allocator, newSurface.indexBuffer.buffer, newSurface.indexBuffer.allocation); 
            vmaDestroyBuffer(_allocator, newSurface.vertexBuffer.buffer, newSurface.vertexBuffer.allocation);
        });

    return newSurface;
}

AllocatedImage VulkanEngine::create_image(VkExtent3D size, VkFormat format, VkImageUsageFlags usage, bool mipmapped)
{
    AllocatedImage newImage{};
    newImage.imageFormat = format;
    newImage.imageExtent = size;

    VkImageCreateInfo imgInfo = vkinit::image_create_info(format, usage, size);
    if (mipmapped) {
        imgInfo.mipLevels = static_cast<uint32_t>(std::floor(std::log2(std::max(size.width, size.height)))) + 1;
    }

    // always alloc images on gpu memory
    VmaAllocationCreateInfo allocInfo{};
    allocInfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;
    allocInfo.requiredFlags = VkMemoryPropertyFlags(VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    // alloc and create the image
    VK_CHECK(vmaCreateImage(_allocator, &imgInfo, &allocInfo, &newImage.image, &newImage.allocation, nullptr));

    // if depth format, use correct aspect flag
    VkImageAspectFlags aspectFlag = VK_IMAGE_ASPECT_COLOR_BIT;
    if (format == VK_FORMAT_D32_SFLOAT)
    {
        aspectFlag = VK_IMAGE_ASPECT_DEPTH_BIT;
    }

    // built imageView for image
    VkImageViewCreateInfo viewInfo = vkinit::imageview_create_info(format, newImage.image, aspectFlag);
    viewInfo.subresourceRange.levelCount = imgInfo.mipLevels;

    VK_CHECK(vkCreateImageView(_device, &viewInfo, nullptr, &newImage.imageView));

    return newImage;
}

AllocatedImage VulkanEngine::create_image(void* data, VkExtent3D size, VkFormat format, VkImageUsageFlags usage, bool mipmapped)
{
    // hardcoded to RGBA 8 bit format at the moment
    size_t data_size = size.depth * size.width * size.height * 4;
    AllocatedBuffer uploadBuffer = create_buffer(data_size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);

    memcpy(uploadBuffer.info.pMappedData, data, data_size);

    AllocatedImage newImage = create_image(size, format, usage | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT, mipmapped);

    immediate_submit([&](VkCommandBuffer cmd)
    {
        vkutil::transition_image(cmd, newImage.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);

        VkBufferImageCopy copyRegion{};
        copyRegion.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        copyRegion.imageSubresource.layerCount = 1;
        copyRegion.imageExtent = size;

        // copy the buffer into the image
        vkCmdCopyBufferToImage(cmd, uploadBuffer.buffer, newImage.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copyRegion);
        if ( mipmapped )
        {
            vkutil::generate_mipmaps(cmd, newImage.image, VkExtent2D{ newImage.imageExtent.width, newImage.imageExtent.height });
        }
        else
        {
            vkutil::transition_image(cmd, newImage.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        }
    });

    destroy_buffer(uploadBuffer);
    return newImage;
}

void VulkanEngine::destroy_image(const AllocatedImage& img)
{
    vkDestroyImageView(_device, img.imageView, nullptr);
    vmaDestroyImage(_allocator, img.image, img.allocation);
}

void VulkanEngine::init_vulkan()
{
    vkb::InstanceBuilder builder;
    // create vk instance with default debugging
    auto inst_ret = builder.set_app_name("Vulkan Renderer")
        .request_validation_layers(bUseValidationLayers)
        .use_default_debug_messenger()
        .require_api_version(1, 3, 0)
        .build();
    vkb::Instance vkb_inst = inst_ret.value();

    // storing instance and debug messenger
    _instance = vkb_inst.instance;
    _debug_messenger = vkb_inst.debug_messenger;

    SDL_Vulkan_CreateSurface(_window, _instance, &_surface);

    // vulkan 1.3 features
    VkPhysicalDeviceVulkan13Features features13{};
    features13.dynamicRendering = true; // no need for old render pass/frame buffer method
    features13.synchronization2 = true; // improved synchronization functions

    // vulkan 1.2 features
    VkPhysicalDeviceVulkan12Features features12{};
    features12.bufferDeviceAddress = true;  // use gpu pointers without buffer binding
    features12.descriptorIndexing = true;   // bindless textures

    // select a gpu (one that can write to the surface and supports vk 1.3)
    vkb::PhysicalDeviceSelector selector{ vkb_inst };
    vkb::PhysicalDevice physicalDevice = selector
        .set_minimum_version(1, 3)
        .set_required_features_13(features13)
        .set_required_features_12(features12)
        .set_surface(_surface)
        .select()
        .value();

    // create final vk device
    vkb::DeviceBuilder deviceBuilder{ physicalDevice };

    vkb::Device vkbDevice = deviceBuilder.build().value();

    // get VkDevice handle used in the rest of the vulkan app
    _device = vkbDevice.device;
    _chosenGPU = physicalDevice.physical_device;

    // get a graphics queue
    _graphicsQueue = vkbDevice.get_queue(vkb::QueueType::graphics).value();
    _graphicsQueueFamily = vkbDevice.get_queue_index(vkb::QueueType::graphics).value();

    // init memory allocator
    VmaAllocatorCreateInfo allocInfo{};
    allocInfo.physicalDevice = _chosenGPU;
    allocInfo.device = _device;
    allocInfo.instance = _instance;
    allocInfo.flags = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT;
    vmaCreateAllocator(&allocInfo, &_allocator);

    _mainDeletionQueue.push_function([&]() { vmaDestroyAllocator(_allocator); });
}

void VulkanEngine::init_swapchain()
{
    create_swapchain(_windowExtent.width, _windowExtent.height);

    // draw image (not drawing directly to swapchain)
    SDL_DisplayMode dMode{};
    SDL_GetCurrentDisplayMode(0, &dMode);
    VkExtent3D drawImageExtent{};
    drawImageExtent.width = static_cast<uint32_t>(dMode.w);
    drawImageExtent.height = static_cast<uint32_t>(dMode.h);
    drawImageExtent.depth = 1;

    // hardcode draw format to 32 bit float
    _drawImage.imageFormat = VK_FORMAT_R16G16B16A16_SFLOAT;
    _drawImage.imageExtent = drawImageExtent;

    VkImageUsageFlags drawImageUsages{};
    drawImageUsages |= VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    drawImageUsages |= VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    drawImageUsages |= VK_IMAGE_USAGE_STORAGE_BIT;
    drawImageUsages |= VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;

    VkImageCreateInfo rimg_info = vkinit::image_create_info(_drawImage.imageFormat, drawImageUsages, drawImageExtent);

    // for draw image, alloc from gpu local memory
    VmaAllocationCreateInfo rimg_allocinfo{};
    rimg_allocinfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;
    rimg_allocinfo.requiredFlags = VkMemoryPropertyFlags(VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    // allocate and create the image
    vmaCreateImage(_allocator, &rimg_info, &rimg_allocinfo, &_drawImage.image, &_drawImage.allocation, nullptr);

    // build image view for draw imuge to use for rendering
    VkImageViewCreateInfo rview_info = vkinit::imageview_create_info(_drawImage.imageFormat, _drawImage.image, VK_IMAGE_ASPECT_COLOR_BIT);

    VK_CHECK(vkCreateImageView(_device, &rview_info, nullptr, &_drawImage.imageView));

    // now for depth image
    _depthImage.imageFormat = VK_FORMAT_D32_SFLOAT;
    _depthImage.imageExtent = drawImageExtent;
    VkImageUsageFlags depthImageUsages{};
    depthImageUsages |= VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;

    VkImageCreateInfo dimg_info = vkinit::image_create_info(_depthImage.imageFormat, depthImageUsages, drawImageExtent);

    // allocate image
    vmaCreateImage(_allocator, &dimg_info, &rimg_allocinfo, &_depthImage.image, &_depthImage.allocation, nullptr);

    // build an imageview for draw image to use for rendering
    VkImageViewCreateInfo dview_info = vkinit::imageview_create_info(_depthImage.imageFormat, _depthImage.image, VK_IMAGE_ASPECT_DEPTH_BIT);

    VK_CHECK(vkCreateImageView(_device, &dview_info, nullptr, &_depthImage.imageView));

    // add to deletion queues
    _mainDeletionQueue.push_function([=]() 
    {
        vkDestroyImageView(_device, _drawImage.imageView, nullptr);
        vmaDestroyImage(_allocator, _drawImage.image, _drawImage.allocation);

        vkDestroyImageView(_device, _depthImage.imageView, nullptr);
        vmaDestroyImage(_allocator, _depthImage.image, _depthImage.allocation);
    });
}

void VulkanEngine::init_commands()
{
    // create command pool for submitting commands to graphics queue
    // also allow for resetting of individual command buffers
    VkCommandPoolCreateInfo commandPoolInfo{};
    commandPoolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    commandPoolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    commandPoolInfo.queueFamilyIndex = _graphicsQueueFamily;

    for (size_t i = 0; i < FRAME_OVERLAP; i++) {
        VK_CHECK(vkCreateCommandPool(_device, &commandPoolInfo, nullptr, &_frames[i]._commandPool));

        // allocate default command buffer for rendering
        VkCommandBufferAllocateInfo cmdAllocInfo = vkinit::command_buffer_allocate_info(_frames[i]._commandPool, 1);

        VK_CHECK(vkAllocateCommandBuffers(_device, &cmdAllocInfo, &_frames[i]._mainCommandBuffer));
    }

    // immediate submit sync structures
    VK_CHECK(vkCreateCommandPool(_device, &commandPoolInfo, nullptr, &_immCommandPool));

    // alloc command buffer for immediate submits
    VkCommandBufferAllocateInfo cmdAllocInfo = vkinit::command_buffer_allocate_info(_immCommandPool, 1);

    VK_CHECK(vkAllocateCommandBuffers(_device, &cmdAllocInfo, &_immCommandBuffer));

    _mainDeletionQueue.push_function([=]() 
    {
        vkDestroyCommandPool(_device, _immCommandPool, nullptr);
    });

}

void VulkanEngine::init_sync_structures()
{
    // one fence controls when gpu has finished rendering frame
    // two semaphores to synchronize rendering with swapchain
    // fence starts signalled so we wait on it on first frame
    VkFenceCreateInfo fenceCreateInfo = vkinit::fence_create_info(VK_FENCE_CREATE_SIGNALED_BIT);
    VkSemaphoreCreateInfo semaphoreCreateInfo = vkinit::semaphore_create_info();

    for (size_t i = 0; i < FRAME_OVERLAP; i++) {
        VK_CHECK(vkCreateFence(_device, &fenceCreateInfo, nullptr, &_frames[i]._renderFence));

        VK_CHECK(vkCreateSemaphore(_device, &semaphoreCreateInfo, nullptr, &_frames[i]._swapchainSemaphore));
        VK_CHECK(vkCreateSemaphore(_device, &semaphoreCreateInfo, nullptr, &_frames[i]._renderSemaphore));
    }

    VK_CHECK(vkCreateFence(_device, &fenceCreateInfo, nullptr, &_immFence));
    _mainDeletionQueue.push_function([=]() 
        {
            vkDestroyFence(_device, _immFence, nullptr);
        });
}

void VulkanEngine::init_default_data()
{
    std::array<Vertex, 4> rect_vertices{};

    rect_vertices[0].position = { 0.5, -0.5, 0.0 };
    rect_vertices[1].position = { 0.5, 0.5, 0.0 };
    rect_vertices[2].position = { -0.5, -0.5, 0.0 };
    rect_vertices[3].position = { -0.5, 0.5, 0.0 };

    rect_vertices[0].color = { 0.0, 0.0, 0.0, 1.0 };
    rect_vertices[1].color = { 0.5, 0.5, 0.5, 1.0 };
    rect_vertices[2].color = { 1.0, 0.0, 0.0, 1.0 };
    rect_vertices[3].color = { 0.0, 1.0, 0.0, 1.0 };

    rect_vertices[0].uv_x = 1;
    rect_vertices[0].uv_y = 0;
    rect_vertices[1].uv_x = 0;
    rect_vertices[1].uv_y = 0;
    rect_vertices[2].uv_x = 1;
    rect_vertices[2].uv_y = 1;
    rect_vertices[3].uv_x = 0;
    rect_vertices[3].uv_y = 1;

    std::array<uint32_t, 6> rect_indices{};

    rect_indices[0] = 0;
    rect_indices[1] = 1;
    rect_indices[2] = 2;

    rect_indices[3] = 2;
    rect_indices[4] = 1;
    rect_indices[5] = 3;

    rectangle = uploadMesh(rect_indices, rect_vertices);


    // white, grey, black default textures
    uint32_t white = glm::packUnorm4x8(glm::vec4(1, 1, 1, 1));
    
    _whiteImage = create_image( reinterpret_cast<void*>(&white), VkExtent3D{1, 1, 1}, VK_FORMAT_R8G8B8A8_UNORM,
        VK_IMAGE_USAGE_SAMPLED_BIT);

    uint32_t gray = glm::packUnorm4x8(glm::vec4(0.66f, 0.66f, 0.66f, 1));
    _grayImage = create_image(reinterpret_cast<void*>(&gray), VkExtent3D{ 1, 1, 1 }, VK_FORMAT_R8G8B8A8_UNORM,
        VK_IMAGE_USAGE_SAMPLED_BIT);

    uint32_t black = glm::packUnorm4x8(glm::vec4(0, 0, 0, 0));
    _blackImage = create_image(reinterpret_cast<void*>(&black), VkExtent3D{ 1, 1, 1 }, VK_FORMAT_R8G8B8A8_UNORM,
        VK_IMAGE_USAGE_SAMPLED_BIT);

    // checkerboard
    uint32_t magenta = glm::packUnorm4x8(glm::vec4(1, 0, 1, 1));
    std::array<uint32_t, 16 * 16> pixels;
    for (size_t x = 0; x < 16; x++) 
    {
        for (size_t y = 0; y < 16; y++)
        {
            pixels[y * 16 + x] = ((x % 2) ^ (y % 2)) ? magenta : black;
        }
    }

    _errorCheckerboardImage = create_image(pixels.data(), VkExtent3D{ 16, 16, 1 }, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_USAGE_SAMPLED_BIT);

    VkSamplerCreateInfo sampleInfo{};
    sampleInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;

    sampleInfo.magFilter = VK_FILTER_NEAREST;
    sampleInfo.minFilter = VK_FILTER_NEAREST;
    vkCreateSampler(_device, &sampleInfo, nullptr, &_defaultSamplerNearest);

    sampleInfo.magFilter = VK_FILTER_LINEAR;
    sampleInfo.minFilter = VK_FILTER_LINEAR;
    vkCreateSampler(_device, &sampleInfo, nullptr, &_defaultSamplerLinear);

    GLTFMetallic_Roughness::MaterialResources materialResources;
    //default the material textures
    materialResources.colorImage = _whiteImage;
    materialResources.colorSampler = _defaultSamplerLinear;
    materialResources.metalRoughImage = _whiteImage;
    materialResources.metalRoughSampler = _defaultSamplerLinear;

    //set the uniform buffer for the material data
    AllocatedBuffer materialConstants = create_buffer(sizeof(GLTFMetallic_Roughness::MaterialConstants), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);

    //write the buffer
    GLTFMetallic_Roughness::MaterialConstants *sceneUniformData = (GLTFMetallic_Roughness::MaterialConstants *)materialConstants.allocation->GetMappedData();
    sceneUniformData->colorFactors = glm::vec4{ 1,1,1,1 };
    sceneUniformData->metalRoughFactors = glm::vec4{ 1,0.5,0,0 };

    _mainDeletionQueue.push_function([=, this]() {
        destroy_buffer(materialConstants);
        });

    materialResources.dataBuffer = materialConstants.buffer;
    materialResources.dataBufferOffset = 0;

    defaultData = metalRoughMaterial.write_material(_device, MaterialPass::MainColor, materialResources, gDescriptorAllocator);

    _mainDeletionQueue.push_function(
    [=, this]() 
    {     
        vkDestroySampler(_device, _defaultSamplerNearest, nullptr);
        vkDestroySampler(_device, _defaultSamplerLinear, nullptr);
        destroy_image(_whiteImage);
        destroy_image(_grayImage); 
        destroy_image(_blackImage);
        destroy_image(_errorCheckerboardImage);
    });

    // monkey mesh
#if 0
    testMeshes = loadGltf(this, "..\\..\\assets\\basicmesh.glb").value();

    for (auto &m : testMeshes)
    {
        std::shared_ptr<MeshNode> newNode = std::make_shared<MeshNode>(MeshNode{});

        newNode->mesh = m;
        newNode->localTransform = glm::mat4{ 1.f };
        newNode->worldTransform = glm::mat4{ 1.f };

        for (auto &s : newNode->mesh->surfaces)
        {
            s.material = std::make_shared<GLTFMaterial>(defaultData);
        }
        loadedNodes[m->name] = std::move(newNode);
    }
#endif
}

void VulkanEngine::create_swapchain(uint32_t width, uint32_t height)
{
    vkb::SwapchainBuilder swapchainBuilder{ _chosenGPU, _device, _surface };

    _swapchainImageFormat = VK_FORMAT_B8G8R8A8_UNORM;

    vkb::Swapchain vkbSwapchain = swapchainBuilder
        .set_desired_format(VkSurfaceFormatKHR{ .format = _swapchainImageFormat, .colorSpace = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR })
        // use vsync present mode (mailbox is norm in real apps)
        .set_desired_present_mode(VK_PRESENT_MODE_FIFO_KHR)
        .set_desired_extent(width, height)
        .add_image_usage_flags(VK_IMAGE_USAGE_TRANSFER_DST_BIT)
        .build()
        .value();
    _swapchainExtent = vkbSwapchain.extent;
    // store swapchain and its related images
    _swapchain = vkbSwapchain.swapchain;
    _swapchainImages = vkbSwapchain.get_images().value();
    _swapchainImageViews = vkbSwapchain.get_image_views().value();
}

void VulkanEngine::destroy_swapchain()
{
    vkDestroySwapchainKHR(_device, _swapchain, nullptr);

    // destroy swapchain resources
    for (auto& imageView : _swapchainImageViews) {
        vkDestroyImageView(_device, imageView, nullptr);
    }

}

void VulkanEngine::draw_background(VkCommandBuffer cmd)
{
    ComputeEffect& effect = backgroundEffects[currentBackgroundEffect];

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, effect.pipeline);

    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, _gradientPipelineLayout, 0, 1, &_drawImageDescriptors, 0, nullptr);

    vkCmdPushConstants(cmd, _gradientPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(ComputePushConstants), &effect.data);

    vkCmdDispatch(cmd, static_cast<uint32_t>(std::ceil(_drawExtent.width / 16.0)), static_cast<uint32_t>(std::ceil(_drawExtent.height / 16.0)), 1);
}

void VulkanEngine::draw_main(VkCommandBuffer cmd)
{
    VkRenderingAttachmentInfo colorAttachment = vkinit::attachment_info(_drawImage.imageView, nullptr, VK_IMAGE_LAYOUT_GENERAL);
    VkRenderingAttachmentInfo depthAttachment = vkinit::depth_attachment_info(_depthImage.imageView, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL);
    VkRenderingInfo renderInfo = vkinit::rendering_info(_windowExtent, &colorAttachment, &depthAttachment);

    vkCmdBeginRendering(cmd, &renderInfo);
    auto start = std::chrono::system_clock::now();
    draw_geometry(cmd);
    auto end = std::chrono::system_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    stats.mesh_draw_time = elapsed.count() / 1000.f;

    vkCmdEndRendering(cmd);
}

void VulkanEngine::draw_geometry(VkCommandBuffer cmd)
{
    // reset counts
    stats.drawcall_count = 0;
    stats.triangle_count = 0;

    std::vector<uint32_t> opaque_draws;
    opaque_draws.reserve(mainDrawContext.OpaqueSurfaces.size());

    for ( uint32_t i = 0; i < mainDrawContext.OpaqueSurfaces.size(); ++i )
    {
        if ( is_visible(mainDrawContext.OpaqueSurfaces[i], sceneData.viewproj) )
        {
            opaque_draws.push_back(i);
        }
    }

    // sort opaque surfaces by mat and mesh
    std::sort(opaque_draws.begin(), opaque_draws.end(), [&](const auto &iA, const auto &iB) 
        {
            const RenderObject &A = mainDrawContext.OpaqueSurfaces[iA];
            const RenderObject &B = mainDrawContext.OpaqueSurfaces[iB];
            if ( A.material == B.material )
            {
                return A.indexBuffer < B.indexBuffer;
            }
            else
            {
                return A.material < B.material;
            }
        });



    // alloc new uniform buffer for scene data (skips staging step)
    AllocatedBuffer gpuSceneDataBuffer = create_buffer(sizeof(GPUSceneData), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);
    vmaSetAllocationName(_allocator, gpuSceneDataBuffer.allocation, "gpuSceneDataBuffer");

    // add to deletion queue of this frame
    get_current_frame()._deletionQueue.push_function(
    [=, this]()
    {
        destroy_buffer(gpuSceneDataBuffer);
    });

    // write buffer
    GPUSceneData *sceneUniformData = reinterpret_cast<GPUSceneData *>(gpuSceneDataBuffer.allocation->GetMappedData());
    *sceneUniformData = sceneData;

    // create descriptor set that binds buffer and updates it
    VkDescriptorSet globalDescriptor = get_current_frame()._frameDescriptors.allocate(_device, _gpuSceneDataDescriptorLayout);

    DescriptorWriter writer{};
    writer.write_buffer(0, gpuSceneDataBuffer.buffer, sizeof(GPUSceneData), 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER);
    writer.update_set(_device, globalDescriptor);

    MaterialPipeline *lastPipeline{};
    MaterialInstance *lastMaterial{};
    VkBuffer lastIndexBuffer = VK_NULL_HANDLE;

    auto draw = [&](const RenderObject &drawObj)
        {
            if ( drawObj.material != lastMaterial )
            {
                lastMaterial = drawObj.material;
                if ( drawObj.material->pipeline != lastPipeline )
                {
                    lastPipeline = drawObj.material->pipeline;
                    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, 
                        drawObj.material->pipeline->pipeline);
                    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, 
                        drawObj.material->pipeline->layout, 0, 1, &globalDescriptor, 0, nullptr);

                    // set dynamic viewport and scissor
                    VkViewport viewport{};
                    viewport.x = 0;
                    viewport.y = 0;
                    viewport.width = static_cast<float>(_drawExtent.width);
                    viewport.height = static_cast<float>(_drawExtent.height);
                    viewport.minDepth = 0.f;
                    viewport.maxDepth = 1.f;
                    vkCmdSetViewport(cmd, 0, 1, &viewport);

                    VkRect2D scissor{};
                    scissor.offset.x = 0;
                    scissor.offset.y = 0;
                    scissor.extent.width = _drawExtent.width;
                    scissor.extent.height = _drawExtent.height;
                    vkCmdSetScissor(cmd, 0, 1, &scissor);
                }
            }

            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, 
                drawObj.material->pipeline->layout, 1, 1, &drawObj.material->materialSet, 0, nullptr);

            // rebind index buffer if needed
            if ( drawObj.indexBuffer != lastIndexBuffer )
            {
                lastIndexBuffer = drawObj.indexBuffer;
                vkCmdBindIndexBuffer(cmd, drawObj.indexBuffer, 0, VK_INDEX_TYPE_UINT32);
            }
            
            // calculate final mesh matrix
            GPUDrawPushConstants pushConstants;
            pushConstants.vertexBuffer = drawObj.vertexBufferAddress;
            pushConstants.worldMatrix = drawObj.transform;

            vkCmdPushConstants(cmd, drawObj.material->pipeline->layout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(GPUDrawPushConstants), &pushConstants);

            vkCmdDrawIndexed(cmd, drawObj.indexCount, 1, drawObj.firstIndex, 0, 0);

            // add counters for triangles and draws
            stats.drawcall_count++;
            stats.triangle_count += drawObj.indexCount / 3;
        };
    for ( auto &r : opaque_draws )
    {
        draw(mainDrawContext.OpaqueSurfaces[r]);
    }
    for ( auto &r : mainDrawContext.TransparentSurfaces )
    {
        draw(r);
    }
}

void VulkanEngine::draw_imgui(VkCommandBuffer cmd, VkImageView targetImageView)
{
    VkRenderingAttachmentInfo colorAttachment = vkinit::attachment_info(targetImageView, nullptr, VK_IMAGE_LAYOUT_GENERAL);
    VkRenderingInfo renderInfo = vkinit::rendering_info(_swapchainExtent, &colorAttachment, nullptr);

    vkCmdBeginRendering(cmd, &renderInfo);

    ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), cmd);

    vkCmdEndRendering(cmd);
}

void VulkanEngine::init_descriptors()
{
    // create descriptor pool that holds 10 sets with 1 image each
    std::vector<DescriptorAllocatorGrowable::PoolSizeRatio> sizes =
    {
        { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1 },
        { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1 },
    };

    gDescriptorAllocator.init(_device, 10, sizes);

    // make descriptor set layout for compute draw
    {
        DescriptorLayoutBuilder builder{};
        builder.add_binding(0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);
        _drawImageDescriptorLayout = builder.build(_device, VK_SHADER_STAGE_COMPUTE_BIT);
    }

    // make descriptor set layout for GPU scene
    {
        DescriptorLayoutBuilder builder{};
        builder.add_binding(0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER);
        _gpuSceneDataDescriptorLayout = builder.build(_device, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT);
    }

    // descriptor set layout for textured mesh
    {
        DescriptorLayoutBuilder builder;
        builder.add_binding(0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);
        _singleImageDescriptorLayout = builder.build(_device, VK_SHADER_STAGE_FRAGMENT_BIT);
    }

    // alloc descriptor set for draw image
    _drawImageDescriptors = gDescriptorAllocator.allocate(_device, _drawImageDescriptorLayout);

    DescriptorWriter writer;
    writer.write_image(0, _drawImage.imageView, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_GENERAL, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);
    writer.update_set(_device, _drawImageDescriptors);

    for (size_t i = 0; i < FRAME_OVERLAP; i++)
    {
        std::vector<DescriptorAllocatorGrowable::PoolSizeRatio> frame_sizes =
        {
            { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 3 },
            { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 3 },
            { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 3 },
            { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 4 },
        };

        _frames[i]._frameDescriptors = DescriptorAllocatorGrowable{};
        _frames[i]._frameDescriptors.init(_device, 1000, frame_sizes);

        _mainDeletionQueue.push_function(
        [&, i]() 
        {
            _frames[i]._frameDescriptors.destroy_pools(_device);
        });
    }

    _mainDeletionQueue.push_function(
    [=]()
    {
        vkDestroyDescriptorSetLayout(_device, _drawImageDescriptorLayout, nullptr);
        vkDestroyDescriptorSetLayout(_device, _gpuSceneDataDescriptorLayout, nullptr);
        vkDestroyDescriptorSetLayout(_device, _singleImageDescriptorLayout, nullptr);
        gDescriptorAllocator.destroy_pools(_device);
    });
}

void VulkanEngine::init_pipelines()
{
    // compute pipelines
    init_background_pipelines();


    metalRoughMaterial.build_pipelines(this);
}

void VulkanEngine::init_background_pipelines()
{
    VkPipelineLayoutCreateInfo computeLayout{};
    computeLayout.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    computeLayout.pSetLayouts = &_drawImageDescriptorLayout;
    computeLayout.setLayoutCount = 1;

    VkPushConstantRange pushConstant{};
    pushConstant.offset = 0;
    pushConstant.size = sizeof(ComputePushConstants);
    pushConstant.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    computeLayout.pPushConstantRanges = &pushConstant;
    computeLayout.pushConstantRangeCount = 1;

    VK_CHECK(vkCreatePipelineLayout(_device, &computeLayout, nullptr, &_gradientPipelineLayout));

    VkShaderModule gradientShader;
    if (!vkutil::load_shader_module("../../shaders/gradient_color.comp.spv", _device, &gradientShader))
    {
        fmt::print("Error when building gradient compute shader\n");
    }
    VkShaderModule skyShader;
    if (!vkutil::load_shader_module("../../shaders/sky.comp.spv", _device, &skyShader))
    {
        fmt::print("Error when building sky compute shader\n");
    }


    VkPipelineShaderStageCreateInfo stageInfo{};
    stageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stageInfo.pNext = nullptr;
    stageInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    stageInfo.module = gradientShader;
    stageInfo.pName = "main";

    VkComputePipelineCreateInfo computePipelineCreateInfo{};
    computePipelineCreateInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    computePipelineCreateInfo.layout = _gradientPipelineLayout;
    computePipelineCreateInfo.stage = stageInfo;

    ComputeEffect gradient{};
    gradient.layout = _gradientPipelineLayout;
    gradient.name = "gradient";
    gradient.data = {};
    gradient.data.data1 = glm::vec4(1.f, 0.f, 0.f, 1.f);
    gradient.data.data2 = glm::vec4(0.f, 0.f, 1.f, 1.f);

    VK_CHECK(vkCreateComputePipelines(_device, VK_NULL_HANDLE, 1, &computePipelineCreateInfo, nullptr, &gradient.pipeline));

    computePipelineCreateInfo.stage.module = skyShader;

    ComputeEffect sky{};
    sky.layout = _gradientPipelineLayout;
    sky.name = "sky";
    sky.data = {};
    sky.data.data1 = glm::vec4(0.1f, 0.2f, 0.4f, 0.97f);

    VK_CHECK(vkCreateComputePipelines(_device, VK_NULL_HANDLE, 1, &computePipelineCreateInfo, nullptr, &sky.pipeline));

    backgroundEffects.push_back(gradient);
    backgroundEffects.push_back(sky);

    vkDestroyShaderModule(_device, gradientShader, nullptr);
    vkDestroyShaderModule(_device, skyShader, nullptr);

    _mainDeletionQueue.push_function([&]()
        {
            vkDestroyPipelineLayout(_device, _gradientPipelineLayout, nullptr);
            vkDestroyPipeline(_device, sky.pipeline, nullptr);
            vkDestroyPipeline(_device, gradient.pipeline, nullptr);
        });
}

void VulkanEngine::init_imgui()
{
    // create descriptor pool for imgui
    // note: is oversized based on imgui demo
    VkDescriptorPoolSize poolSizes[] =
    {
        { VK_DESCRIPTOR_TYPE_SAMPLER, 1000 },
        { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1000 },
        { VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1000 },
        { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1000 },
        { VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER, 1000 },
        { VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER, 1000 },
        { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1000 },
        { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1000 },
        { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 1000 },
        { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC, 1000 },
        { VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT, 1000 }
    };

    VkDescriptorPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    poolInfo.maxSets = 1000;
    poolInfo.poolSizeCount = static_cast<uint32_t>(std::size(poolSizes));
    poolInfo.pPoolSizes = poolSizes;

    VkDescriptorPool imguiPool;
    VK_CHECK(vkCreateDescriptorPool(_device, &poolInfo, nullptr, &imguiPool));

    // initialize imgui library
    
    // init core structures
    ImGui::CreateContext();

    // init imgui for sdl
    ImGui_ImplSDL2_InitForVulkan(_window);

    // init imgui for Vulkan
    ImGui_ImplVulkan_InitInfo initInfo{};
    initInfo.Instance = _instance;
    initInfo.PhysicalDevice = _chosenGPU;
    initInfo.Device = _device;
    initInfo.Queue = _graphicsQueue;
    initInfo.DescriptorPool = imguiPool;
    initInfo.MinImageCount = 3;
    initInfo.ImageCount = 3;
    initInfo.UseDynamicRendering = true;
    initInfo.ColorAttachmentFormat = _swapchainImageFormat;

    initInfo.MSAASamples = VK_SAMPLE_COUNT_1_BIT;

    ImGui_ImplVulkan_Init(&initInfo, VK_NULL_HANDLE);

    // upload imgui font textures
    immediate_submit([&](VkCommandBuffer cmd) { ImGui_ImplVulkan_CreateFontsTexture(cmd); });

    // clear font textures from cpu
    ImGui_ImplVulkan_DestroyFontUploadObjects();

    // add to deletion queue imgui created structures
    _mainDeletionQueue.push_function([=]() 
        {
            vkDestroyDescriptorPool(_device, imguiPool, nullptr);
            ImGui_ImplVulkan_Shutdown();
        });
}

void VulkanEngine::init_triangle_pipeline()
{
    VkShaderModule triangleFragShader;
    if (!vkutil::load_shader_module("../../shaders/colored_triangle.frag.spv", _device, &triangleFragShader)) {
        fmt::print("Error when building the triangle fragment shader module\n");
    }
    else {
        fmt::print("Triangle fragment shader succesfully loaded\n");
    }

    VkShaderModule triangleVertexShader;
    if (!vkutil::load_shader_module("../../shaders/colored_triangle.vert.spv", _device, &triangleVertexShader)) {
        fmt::print("Error when building the triangle vertex shader module\n");
    }
    else {
        fmt::print("Triangle vertex shader succesfully loaded\n");
    }

    //build the pipeline layout that controls the inputs/outputs of the shader
    //we are not using descriptor sets or other systems yet, so no need to use anything other than empty default
    VkPipelineLayoutCreateInfo pipeline_layout_info = vkinit::pipeline_layout_create_info();
    VK_CHECK(vkCreatePipelineLayout(_device, &pipeline_layout_info, nullptr, &_trianglePipelineLayout));

    PipelineBuilder pipelineBuilder;
    // use triangle layout
    pipelineBuilder._pipelineLayout = _trianglePipelineLayout;
    // conect vertex and fragment shaders to pipeline
    pipelineBuilder.set_shaders(triangleVertexShader, triangleFragShader);

    pipelineBuilder.set_input_topology(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST);
    pipelineBuilder.set_polygon_mode(VK_POLYGON_MODE_FILL);
    pipelineBuilder.set_cull_mode(VK_CULL_MODE_NONE, VK_FRONT_FACE_CLOCKWISE);
    pipelineBuilder.set_multisampling_none();
    pipelineBuilder.disable_blending();
    pipelineBuilder.disable_depthtest();

    // connect image format we will draw into
    pipelineBuilder.set_color_attachment_format(_drawImage.imageFormat);
    pipelineBuilder.set_depth_format(_depthImage.imageFormat);

    // build pipeline
    _trianglePipeline = pipelineBuilder.build_pipeline(_device, "trianglePipeline");

    // clean structures
    vkDestroyShaderModule(_device, triangleFragShader, nullptr);
    vkDestroyShaderModule(_device, triangleVertexShader, nullptr);

    _mainDeletionQueue.push_function([&]() {
        vkDestroyPipelineLayout(_device, _trianglePipelineLayout, nullptr);
        vkDestroyPipeline(_device, _trianglePipeline, nullptr);
        });
}

void VulkanEngine::init_mesh_pipeline()
{
    VkShaderModule triangleFragShader;
    if (!vkutil::load_shader_module("../../shaders/mesh.frag.spv", _device, &triangleFragShader)) {
        fmt::print("Error when building the triangle fragment (mesh) shader module\n");
    }
    else {
        fmt::print("Triangle fragment shader (mesh) succesfully loaded\n");
    }

    VkShaderModule triangleVertexShader;
    if (!vkutil::load_shader_module("../../shaders/mesh.vert.spv", _device, &triangleVertexShader)) {
        fmt::print("Error when building the triangle vertex shader (mesh) module\n");
    }
    else {
        fmt::print("Triangle vertex shader (mesh) succesfully loaded\n");
    }

    VkPushConstantRange bufferRange{};
    bufferRange.size = sizeof(GPUDrawPushConstants);
    bufferRange.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;

    VkPipelineLayoutCreateInfo pipeline_layout_info = vkinit::pipeline_layout_create_info();
    pipeline_layout_info.pPushConstantRanges = &bufferRange;
    pipeline_layout_info.pushConstantRangeCount = 1;
    pipeline_layout_info.pSetLayouts = &_singleImageDescriptorLayout;
    pipeline_layout_info.setLayoutCount = 1;

    VK_CHECK(vkCreatePipelineLayout(_device, &pipeline_layout_info, nullptr, &_meshPipelineLayout));

    PipelineBuilder pipelineBuilder;

    //use the triangle layout we created
    pipelineBuilder._pipelineLayout = _meshPipelineLayout;
    //connecting the vertex and pixel shaders to the pipeline
    pipelineBuilder.set_shaders(triangleVertexShader, triangleFragShader);
    //it will draw triangles
    pipelineBuilder.set_input_topology(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST);
    //filled triangles
    pipelineBuilder.set_polygon_mode(VK_POLYGON_MODE_FILL);
    //no backface culling
    pipelineBuilder.set_cull_mode(VK_CULL_MODE_NONE, VK_FRONT_FACE_CLOCKWISE);
    //no multisampling
    pipelineBuilder.set_multisampling_none();
    //no blending
#if 1
    pipelineBuilder.disable_blending();
#else
    pipelineBuilder.enable_blending_additive();
#endif

    pipelineBuilder.enable_depthtest(true, VK_COMPARE_OP_GREATER_OR_EQUAL);

    //connect the image format we will draw into, from draw image
    pipelineBuilder.set_color_attachment_format(_drawImage.imageFormat);
    pipelineBuilder.set_depth_format(_depthImage.imageFormat);

    //finally build the pipeline
    _meshPipeline = pipelineBuilder.build_pipeline(_device, "meshPipeline");

    //clean structures
    vkDestroyShaderModule(_device, triangleFragShader, nullptr);
    vkDestroyShaderModule(_device, triangleVertexShader, nullptr);

    _mainDeletionQueue.push_function([&]() {
        vkDestroyPipelineLayout(_device, _meshPipelineLayout, nullptr);
        vkDestroyPipeline(_device, _meshPipeline, nullptr);
        });

}

AllocatedBuffer VulkanEngine::create_buffer(size_t allocSize, VkBufferUsageFlags usage, VmaMemoryUsage memoryUsage)
{
    // allocate buffer
    VkBufferCreateInfo bufferInfo = { .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
    bufferInfo.size = allocSize;
    bufferInfo.usage = usage;

    VmaAllocationCreateInfo vmaAllocInfo{};
    vmaAllocInfo.usage = memoryUsage;
    vmaAllocInfo.flags = VMA_ALLOCATION_CREATE_MAPPED_BIT;
    AllocatedBuffer newBuffer{};

    VK_CHECK(vmaCreateBuffer(_allocator, &bufferInfo, &vmaAllocInfo, &newBuffer.buffer, &newBuffer.allocation, &newBuffer.info));

    return newBuffer;
}

void VulkanEngine::destroy_buffer(const AllocatedBuffer &buffer)
{
    vmaDestroyBuffer(_allocator, buffer.buffer, buffer.allocation);
}

void VulkanEngine::resize_swapchain()
{
    vkDeviceWaitIdle(_device);

    destroy_swapchain();
    int w, h;
    SDL_GetWindowSize(_window, &w, &h);
    _windowExtent.width = static_cast<uint32_t>(w);
    _windowExtent.height = static_cast<uint32_t>(h);

    create_swapchain(_windowExtent.width, _windowExtent.height);
    resize_requested = false;
}
