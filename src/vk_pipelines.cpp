#include <vk_pipelines.h>
#include <fstream>
#include <filesystem>
#include "vk_initializers.h"

bool vkutil::load_shader_module(const char* filePath, VkDevice device, VkShaderModule* outShaderModule)
{
	std::filesystem::path p{ filePath };
	
	if (!std::filesystem::exists(p))
	{
		return false;
	}

	size_t fileSize = static_cast<size_t>(std::filesystem::file_size(p));

	std::ifstream file(p, std::ios::binary);

	std::vector<uint32_t> buffer(fileSize / sizeof(uint32_t));

	file.read(reinterpret_cast<char*>(buffer.data()), fileSize);

	file.close();

	VkShaderModuleCreateInfo createInfo{};
	createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;

	createInfo.codeSize = buffer.size() * sizeof(uint32_t);
	createInfo.pCode = buffer.data();

	VkShaderModule shaderModule{};
	if (vkCreateShaderModule(device, &createInfo, nullptr, &shaderModule) != VK_SUCCESS)
	{
		return false;
	}
	*outShaderModule = shaderModule;
	return true;
}
