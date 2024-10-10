#include "stb_image.h"
#include <iostream>
#include <vk_loader.h>

#include "vk_engine.h"
#include "vk_initializers.h"
#include "vk_types.h"
#include "glm/gtx/quaternion.hpp"

#include "fastgltf/glm_element_traits.hpp"
#include "fastgltf/parser.hpp"
#include "fastgltf/tools.hpp"
#include "fastgltf/util.hpp"

#include "stb_image.h"

#include <cstdio>

VkFilter extract_filter(fastgltf::Filter filter);
VkSamplerMipmapMode extract_mipmap_mode(fastgltf::Filter filter);
std::optional<AllocatedImage> load_image(VulkanEngine *engine, fastgltf::Asset &asset, fastgltf::Image &image);

std::optional<std::shared_ptr<LoadedGLTF>> loadGltf(VulkanEngine *engine, std::string_view filePath)
{
    fmt::println("Loading GLTF: {}", filePath);

    std::shared_ptr<LoadedGLTF> scene = std::make_shared<LoadedGLTF>();
    scene->creator = engine;
    LoadedGLTF &file = *scene.get();

    fastgltf::Parser parser{};
    constexpr auto gltfOptions = fastgltf::Options::DontRequireValidAssetMember | fastgltf::Options::AllowDouble | fastgltf::Options::LoadGLBBuffers | fastgltf::Options::LoadExternalBuffers;

    fastgltf::GltfDataBuffer data;
    data.loadFromFile(filePath);

    fastgltf::Asset gltf;

    std::filesystem::path path = filePath;

    auto type = fastgltf::determineGltfFileType(&data);
    if (type == fastgltf::GltfType::glTF)
    {
        auto load = parser.loadGLTF(&data, path.parent_path(), gltfOptions);
        if (load)
        {
            gltf = std::move(load.get());
        }
        else
        {
            fmt::println(stderr, "Failed to load glTF: {}", fastgltf::to_underlying(load.error()));
        }
    }
    else if (type == fastgltf::GltfType::GLB)
    {
        auto load = parser.loadBinaryGLTF(&data, path.parent_path(), gltfOptions);
        if (load)
        {
            gltf = std::move(load.get());
        }
        else
        {
            fmt::println(stderr, "Failed to load glTF: {}", fastgltf::to_underlying(load.error()));
            return {};
        }
    }
    else
    {
        fmt::println(stderr, "Failed to determine glTF container.");
        return {};
    }

    std::vector<DescriptorAllocatorGrowable::PoolSizeRatio> sizes =
    {
        { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 3 },
        { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 3 },
        { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1 }
    };

    file.descriptorPool.init(engine->_device, static_cast<uint32_t>(gltf.materials.size()), sizes);

    for (fastgltf::Sampler &sampler : gltf.samplers)
    {
        VkSamplerCreateInfo sample{.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
        sample.maxLod = VK_LOD_CLAMP_NONE;
        sample.minLod = 0;

        sample.magFilter = extract_filter(sampler.magFilter.value_or(fastgltf::Filter::Nearest));
        sample.minFilter = extract_filter(sampler.minFilter.value_or(fastgltf::Filter::Nearest));

        sample.mipmapMode = extract_mipmap_mode(sampler.minFilter.value_or(fastgltf::Filter::Nearest));

        VkSampler newSampler;
        vkCreateSampler(engine->_device, &sample, nullptr, &newSampler);

        file.samplers.push_back(newSampler);
    }
    // temporal arrays while creating GLTF data
    std::vector<std::shared_ptr<MeshAsset>> meshes;
    std::vector<std::shared_ptr<Node>> nodes;
    std::vector<AllocatedImage> images;
    std::vector<std::shared_ptr<GLTFMaterial>> materials;

    for (fastgltf::Image &image : gltf.images)
    {
        std::optional<AllocatedImage> img = load_image(engine, gltf, image);
        if ( img.has_value() )
        {
            images.push_back(*img);
            file.images[image.name.c_str()] = *img;
        }
        else
        {
            images.push_back(engine->_errorCheckerboardImage);
            fmt::println(stderr, "glTF failed to load texture {}", image.name);
        }
    }

    file.materialDataBuffer = engine->create_buffer(sizeof(GLTFMetallic_Roughness::MaterialConstants) * gltf.materials.size(), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);
    int data_index = 0;
    GLTFMetallic_Roughness::MaterialConstants *sceneMaterialConstants = reinterpret_cast<GLTFMetallic_Roughness::MaterialConstants *>(file.materialDataBuffer.info.pMappedData);

    for (fastgltf::Material &mat : gltf.materials)
    {
        std::shared_ptr<GLTFMaterial> newMat = std::make_shared<GLTFMaterial>();
        materials.push_back(newMat);
        file.materials[mat.name.c_str()] = newMat; // does this need to be a c_str?

        GLTFMetallic_Roughness::MaterialConstants constants{};
        constants.colorFactors.x = mat.pbrData.baseColorFactor[0];
        constants.colorFactors.y = mat.pbrData.baseColorFactor[1];
        constants.colorFactors.z = mat.pbrData.baseColorFactor[2];
        constants.colorFactors.w = mat.pbrData.baseColorFactor[3];

        constants.metalRoughFactors.x = mat.pbrData.metallicFactor;
        constants.metalRoughFactors.y = mat.pbrData.roughnessFactor;

        // write material params to buffer
        sceneMaterialConstants[data_index] = constants;

        MaterialPass passType = MaterialPass::MainColor;
        if (mat.alphaMode == fastgltf::AlphaMode::Blend)
        {
            passType = MaterialPass::Transparent;
        }

        // default mat textures
        GLTFMetallic_Roughness::MaterialResources materialResources;
        materialResources.colorImage = engine->_whiteImage;
        materialResources.colorSampler = engine->_defaultSamplerLinear;
        materialResources.metalRoughImage = engine->_whiteImage;
        materialResources.metalRoughSampler = engine->_defaultSamplerLinear;

        // set uniform buffer for mat data
        materialResources.dataBuffer = file.materialDataBuffer.buffer;
        materialResources.dataBufferOffset = data_index * sizeof(GLTFMetallic_Roughness::MaterialConstants);

        // grab textures from gltf file
        if (mat.pbrData.baseColorTexture.has_value())
        {
            size_t img = gltf.textures[mat.pbrData.baseColorTexture.value().textureIndex].imageIndex.value();
            size_t sampler = gltf.textures[mat.pbrData.baseColorTexture.value().textureIndex].samplerIndex.value();

            materialResources.colorImage = images[img];
            materialResources.colorSampler = file.samplers[sampler];
        }

        // build material
        newMat->data = engine->metalRoughMaterial.write_material(engine->_device, passType, materialResources, file.descriptorPool);

        data_index++;
    }

    // use same vectors for all meshes
    std::vector<uint32_t> indices;
    std::vector<Vertex> vertices;

    for (fastgltf::Mesh &mesh : gltf.meshes)
    {
        std::shared_ptr<MeshAsset> newmesh = std::make_shared<MeshAsset>();
        meshes.push_back(newmesh);
        file.meshes[mesh.name.c_str()] = newmesh;
        newmesh->name = mesh.name;

        // clear the mesh arrays each mesh, so not merged by error
        indices.clear();
        vertices.clear();

        for (auto &&p : mesh.primitives)
        {
            GeoSurface newSurface;
            newSurface.startIndex = static_cast<uint32_t>(indices.size());
            newSurface.count = static_cast<uint32_t>(gltf.accessors[p.indicesAccessor.value()].count);

            size_t initial_vtx = vertices.size();

            // load indices
            {
                fastgltf::Accessor &indexAccessor = gltf.accessors[p.indicesAccessor.value()];
                indices.reserve(indices.size() + indexAccessor.count);

                fastgltf::iterateAccessor<uint32_t>(gltf, indexAccessor, [&](uint32_t idx) {
                    indices.push_back(idx + static_cast<uint32_t>(initial_vtx));
                    });
            }

            // load vertex positions
            {
                fastgltf::Accessor &posAccessor = gltf.accessors[p.findAttribute("POSITION")->second];
                vertices.resize(vertices.size() + posAccessor.count);

                fastgltf::iterateAccessorWithIndex<glm::vec3>(gltf, posAccessor, 
                    [&](glm::vec3 v, size_t index)
                    {
                        vertices[initial_vtx + index] =  
                        { 
                            .position = v, 
                            .uv_x = 0, 
                            .normal = {1, 0, 0}, 
                            .uv_y = 0, 
                            .color = glm::vec4{1.f}
                        };
                    });
            }

            // load vertex normals
            auto normals = p.findAttribute("NORMAL");
            if (normals != p.attributes.end())
            {
                fastgltf::iterateAccessorWithIndex<glm::vec3>(gltf, gltf.accessors[(*normals).second],
                    [&](glm::vec3 v, size_t index) {
                        vertices[initial_vtx + index].normal = v;
                    });
            }

            // load UVs
            auto uv = p.findAttribute("TEXCOORD_0");
            if (uv != p.attributes.end()) 
            {
                fastgltf::iterateAccessorWithIndex<glm::vec2>(gltf, gltf.accessors[(*uv).second],
                    [&](glm::vec2 v, size_t index)
                    {
                        vertices[initial_vtx + index].uv_x = v.x;
                        vertices[initial_vtx + index].uv_y = v.y;
                    });
            }

            // load vertex colors
            auto colors = p.findAttribute("COLOR_0");
            if (colors != p.attributes.end())
            {
                fastgltf::iterateAccessorWithIndex<glm::vec4>(gltf, gltf.accessors[(*colors).second],
                    [&](glm::vec4 v, size_t index)
                    {
                        vertices[initial_vtx + index].color = v;
                    });
            }

            // find min/max bounds of vertices
            glm::vec3 min_pos = vertices[initial_vtx].position;
            glm::vec3 max_pos = vertices[initial_vtx].position;
            for ( size_t i = initial_vtx; i < vertices.size(); ++i )
            {
                min_pos = glm::min(min_pos, vertices[i].position);
                max_pos = glm::max(max_pos, vertices[i].position);
            }

            // calculate origin and extends from min/max, use extent length for radius
            newSurface.bounds.origin = (max_pos + min_pos) / 2.f;
            newSurface.bounds.extents = (max_pos - min_pos) / 2.f;
            newSurface.bounds.sphereRadius = glm::length(newSurface.bounds.extents);

            if (p.materialIndex.has_value())
            {
                newSurface.material = materials[p.materialIndex.value()];
            }
            else
            {
                newSurface.material = materials[0];
            }
            newmesh->surfaces.push_back(newSurface);
        }
        newmesh->meshBuffers = engine->uploadMesh(indices, vertices);
    }
    // load all nodes
    for (fastgltf::Node &node : gltf.nodes)
    {
        std::shared_ptr<Node> newNode;

        // find if node has a mesh, if so, hook to mesh pointer and allocate with meshnode class
        if (node.meshIndex.has_value())
        {
            newNode = std::make_shared<MeshNode>();
            static_cast<MeshNode *>(newNode.get())->mesh = meshes[*node.meshIndex];
        }
        else
        {
            newNode = std::make_shared<Node>();
        }

        nodes.push_back(newNode);
        file.nodes[node.name.c_str()];

        std::visit(fastgltf::visitor{ [&](fastgltf::Node::TransformMatrix matrix) {
                std::memcpy(&newNode->localTransform, matrix.data(), sizeof(matrix));
            },
            [&](fastgltf::Node::TRS transform) {
                glm::vec3 tl(transform.translation[0], transform.translation[1], transform.translation[2]);
                glm::quat rot(transform.rotation[3], transform.rotation[0], transform.rotation[1], transform.rotation[2]); // w, x, y, z
                glm::vec3 sc(transform.scale[0], transform.scale[1], transform.scale[2]);

                glm::mat4 tm = glm::translate(glm::mat4(1.f), tl);
                glm::mat4 rm = glm::toMat4(rot);
                glm::mat4 sm = glm::scale(glm::mat4(1.f), sc);

                newNode->localTransform = tm * rm * sm;
            } },
            node.transform);
    }
    // run loop again to setup transform hierarchy
    for (size_t i = 0; i < gltf.nodes.size(); i++)
    {
        fastgltf::Node &node = gltf.nodes[i];
        std::shared_ptr<Node> &sceneNode = nodes[i];

        for (auto &c : node.children)
        {
            sceneNode->children.push_back(nodes[c]);
            nodes[c]->parent = sceneNode;
        }
    }

    // find the top nodes with no parents
    for (auto &node : nodes)
    {
        if (node->parent.lock() == nullptr)
        {
            file.topNodes.push_back(node);
            node->refreshTransform(glm::mat4{ 1.f });
        }
    }
    return scene;
}

void LoadedGLTF::Draw(const glm::mat4 &topMatrix, DrawContext &ctx)
{
    // create renderables from the scene nodes
    for (auto &n : topNodes)
    {
        n->Draw(topMatrix, ctx);
    }
}

void LoadedGLTF::clearAll()
{
    VkDevice dv = creator->_device;

    descriptorPool.destroy_pools(dv);
    creator->destroy_buffer(materialDataBuffer);

    for ( auto &[meshName, meshAsset] : meshes )
    {
        creator->destroy_buffer(meshAsset->meshBuffers.indexBuffer);
        creator->destroy_buffer(meshAsset->meshBuffers.vertexBuffer);
    }

    for ( auto &[imageName, allocImage] : images )
    {
        if ( allocImage.image != creator->_errorCheckerboardImage.image )
        {
            creator->destroy_image(allocImage);
        }
    }

    for ( auto &sampler : samplers )
    {
        vkDestroySampler(dv, sampler, nullptr);
    }
}

VkFilter extract_filter(fastgltf::Filter filter)
{
    switch (filter)
    {
    case fastgltf::Filter::Nearest:
        [[fallthrough]];
    case fastgltf::Filter::NearestMipMapNearest:
        [[fallthrough]];
    case fastgltf::Filter::NearestMipMapLinear:
        return VK_FILTER_NEAREST;
    case fastgltf::Filter::Linear:
        [[fallthrough]];
    case fastgltf::Filter::LinearMipMapNearest:
        [[fallthrough]];
    case fastgltf::Filter::LinearMipMapLinear:
        [[fallthrough]];
    default:
        return VK_FILTER_LINEAR;
    }
}

VkSamplerMipmapMode extract_mipmap_mode(fastgltf::Filter filter)
{
    switch (filter) 
    {
    case fastgltf::Filter::NearestMipMapNearest:
        [[fallthrough]];
    case fastgltf::Filter::LinearMipMapNearest:
        return VK_SAMPLER_MIPMAP_MODE_NEAREST;

    case fastgltf::Filter::NearestMipMapLinear:
        [[fallthrough]];
    case fastgltf::Filter::LinearMipMapLinear:
        [[fallthrough]];
    default:
        return VK_SAMPLER_MIPMAP_MODE_LINEAR;
    }
}

std::optional<AllocatedImage> load_image(VulkanEngine *engine, fastgltf::Asset &asset, fastgltf::Image &image)
{
    AllocatedImage newImage{};
    int width, height, nrChannels;

    std::visit(
        fastgltf::visitor
        {
            [](auto &arg) {},
            [&](fastgltf::sources::URI &filePath)
            {
                assert(filePath.fileByteOffset == 0);
                assert(filePath.uri.isLocalPath());


                auto *data = stbi_load(filePath.uri.path().data(), &width, &height, &nrChannels, 4);
                if ( data )
                {
                    VkExtent3D imageSize =
                    {
                        .width = static_cast<uint32_t>(width),
                        .height = static_cast<uint32_t>(height),
                        .depth = 1
                    };
                    newImage = engine->create_image(data, imageSize, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_USAGE_SAMPLED_BIT, false);

                    stbi_image_free(data);
                }
            },
            [&](fastgltf::sources::Vector &vector)
            {
                auto *data = stbi_load_from_memory(vector.bytes.data(), static_cast<int>(vector.bytes.size()),
                    &width, &height, &nrChannels, 4);
                if ( data )
                {
                    VkExtent3D imageSize =
                    {
                        .width = static_cast<uint32_t>(width),
                        .height = static_cast<uint32_t>(height),
                        .depth = 1
                    };
                    newImage = engine->create_image(data, imageSize, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_USAGE_SAMPLED_BIT, false);

                    stbi_image_free(data);
                }
            },
            [&](fastgltf::sources::BufferView &view)
            {
                auto &bufferView = asset.bufferViews[view.bufferViewIndex];
                auto &buffer = asset.buffers[bufferView.bufferIndex];

                std::visit(fastgltf::visitor
                    {
                        [](auto &arg) {},
                        [&](fastgltf::sources::Vector &vector)
                        {
                            auto *data = stbi_load_from_memory(vector.bytes.data() + bufferView.byteOffset,
                                static_cast<int>(bufferView.byteLength), &width, &height, &nrChannels, 4);
                            if ( data )
                            {
                                VkExtent3D imageSize =
                                {
                                    .width = static_cast<uint32_t>(width),
                                    .height = static_cast<uint32_t>(height),
                                    .depth = 1
                                };
                                newImage = engine->create_image(data, imageSize, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_USAGE_SAMPLED_BIT, true);
                                stbi_image_free(data);
                            }
                        }
                    },
                    buffer.data);
            }
        },
        image.data);
    // if any attempts to load the data failed, handle is null
    if ( newImage.image == VK_NULL_HANDLE )
    {
        return {};
    }
    else
    {
        return newImage;
    }
}