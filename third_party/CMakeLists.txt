find_package(Vulkan REQUIRED)

add_library(glm INTERFACE)


add_library(stb_image INTERFACE)

set(TINYGLTF_HEADER_ONLY ON CACHE INTERNAL "" FORCE)
set(TINYGLTF_INSTALL OFF CACHE INTERNAL "" FORCE)
add_subdirectory(tinygltf)

target_include_directories(tinygltf INTERFACE tinygltf)

add_subdirectory(fmt EXCLUDE_FROM_ALL)
add_subdirectory(vma)

find_package(SDL3 REQUIRED CONFIG REQUIRED COMPONENTS SDL3-shared)

add_subdirectory(vk-bootstrap)




target_include_directories(glm INTERFACE glm)

add_library(imgui STATIC)

target_include_directories(imgui PUBLIC imgui)

target_sources(imgui PRIVATE 
  imgui/imgui.h
  imgui/imgui.cpp

  imgui/imgui_demo.cpp
  imgui/imgui_draw.cpp
  imgui/imgui_widgets.cpp
  imgui/imgui_tables.cpp
  imgui/backends/imgui_impl_vulkan.cpp
  imgui/backends/imgui_impl_sdl3.cpp
  )

target_link_libraries(imgui PUBLIC Vulkan::Vulkan SDL3::SDL3)

target_include_directories(stb_image INTERFACE stb_image)
