﻿#pragma once

#include "vk_types.h"
#include "vk_descriptors.h"
#include <unordered_map>
#include <filesystem>

struct GLTFMaterial
{
	MaterialInstance data;
};
struct Bounds
{
	glm::vec3 origin;
	float sphereRadius;
	glm::vec3 extents;
};
struct GeoSurface
{
	uint32_t startIndex;
	uint32_t count;
	Bounds bounds;
	std::shared_ptr<GLTFMaterial> material;
};

struct MeshAsset
{
	std::string name;
	std::vector<GeoSurface> surfaces;
	GPUMeshBuffers meshBuffers;
};

// forward declaration
class VulkanEngine;


struct LoadedGLTF : public IRenderable
{
	// storage for all data on a given gltf file
	std::unordered_map<std::string, std::shared_ptr<MeshAsset>> meshes;
	std::unordered_map<std::string, std::shared_ptr<Node>> nodes;
	std::unordered_map<std::string, AllocatedImage> images;
	std::unordered_map<std::string, std::shared_ptr<GLTFMaterial>> materials;

	// root nodes
	std::vector<std::shared_ptr<Node>> topNodes;

	std::vector<VkSampler> samplers;

	DescriptorAllocatorGrowable descriptorPool;

	AllocatedBuffer materialDataBuffer;

	VulkanEngine *creator;

	~LoadedGLTF() { clearAll(); };

	virtual void Draw(const glm::mat4 &topMatrix, DrawContext &ctx) override;

private:
	void clearAll();
};


std::optional<std::shared_ptr<LoadedGLTF>> loadGltf(VulkanEngine *engine, std::string_view filePath);