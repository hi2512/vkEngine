#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(location = 0) in vec3 fragColor;
layout(location = 1) in vec2 texCoords;

layout(location = 0) out vec4 outColor;

layout(binding = 1) uniform sampler2D texSampler;

void main() {

	vec4 texCol = texture(texSampler, texCoords);
    outColor = texCol + vec4(fragColor, 1.0 ) * 0.25f;
}