#version 460

layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0) readonly buffer input_buf {
    mat4 a;
    mat4 b;
};

layout(set = 0, binding = 1) buffer output_buf {
    mat4 c;
};


void main() {
	c = a * b;
	c = inverse(c);
}
