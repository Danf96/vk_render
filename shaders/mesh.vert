#version 450

#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_buffer_reference : require

#include "input_structures.glsl"

layout (location = 0) out vec3 f_normal;
layout (location = 1) out vec3 f_color;
layout (location = 2) out vec2 f_uv;

struct Vertex {

	vec3 position;
	float uv_x;
	vec3 normal;
	float uv_y;
	vec4 color;
}; 

layout(buffer_reference, std430) readonly buffer VertexBuffer{ 
	Vertex vertices[];
};

//push constants block
layout( push_constant ) uniform constants
{
	mat4 render_matrix;
	VertexBuffer vertexBuffer;
} PushConstants;

void main() 
{
	Vertex v = PushConstants.vertexBuffer.vertices[gl_VertexIndex];
	
	vec4 position = vec4(v.position, 1.0f);

	gl_Position =  sceneData.viewproj * PushConstants.render_matrix * position;

	f_normal = (PushConstants.render_matrix * vec4(v.normal, 0.f)).xyz;
	f_color = v.color.xyz * materialData.colorFactors.xyz;	
	f_uv.x = v.uv_x;
	f_uv.y = v.uv_y;
}
