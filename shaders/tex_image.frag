#version 450

layout (location = 0) in vec3 f_color;
layout (location = 1) in vec2 f_uv;

layout (location = 0) out vec4 final_color;

layout (set = 0, binding = 0) uniform sampler2D tex0;

void main()
{
    final_color = texture(tex0, f_uv) * vec4((f_color) * 0.5, 1.0);
}