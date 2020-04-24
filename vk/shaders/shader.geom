#version 450
//#extension GK_KHR_vulkan_glsl : enable
#extension GL_ARB_separate_shader_objects : enable

layout (triangles) in;

layout (triangle_strip, max_vertices = 6) out;

//layout (location = 0) in vec

void main() {
    gl_Position = gl_in[0].gl_Position;
    EmitVertex();
    gl_Position = gl_in[1].gl_Position;
    EmitVertex();
    gl_Position = gl_in[2].gl_Position;
    EmitVertex();
    
    gl_Position = gl_in[0].gl_Position + vec4(0.5, 0.0, 0.0, 0.0);
    EmitVertex();
    gl_Position = gl_in[1].gl_Position + vec4(0.5, 0.0, 0.0, 0.0);
    EmitVertex();
    gl_Position = gl_in[2].gl_Position + vec4(0.5, 0.0, 0.0, 0.0);
    EmitVertex();
    
}
