#version 450

#extension GL_GOOGLE_include_directive : require
#include "input_structures.glsl"

layout (location = 0) in vec3 f_normal;
layout (location = 1) in vec3 f_color;
layout (location = 2) in vec2 f_uv;

layout (location = 0) out vec4 out_color;

void main() 
{
	float lightValue = max(dot(f_normal, sceneData.sunlightDirection.xyz), 0.1f);

	vec3 color = f_color * texture(color_tex, f_uv).xyz;
	vec3 ambient = color *  sceneData.ambientColor.xyz;

	out_color = vec4(color * lightValue *  sceneData.sunlightColor.w + ambient ,1.0f);
}
