#include <stdlib.h>
#include <stdio.h>
#include <CL/cl.h>
#include "image_utils.c"


#define DEVICE CL_DEVICE_TYPE_GPU//CL_DEVICE_TYPE_DEFAULT



 // opencl values
cl_int err;                       // error code returned from OpenCL calls
size_t global;                    // global domain size

cl_device_id     device_id;       // compute device id
cl_context       context;         // compute context
cl_command_queue commands;        // compute command queue
cl_program       program;         // compute program
cl_kernel        ko_vmul;         // compute kernel for multiplying
cl_kernel        ko_vadd;         // compute kernel for adding
cl_kernel        ko_mat_conv_gray;// compute kernel for convolving matrices
cl_kernel        ko_gray;         // compute kernel for turning rgb images to gray
cl_kernel        ko_resize;       // compute kernel for resizing images
cl_kernel        ko_flip_y;       // compute kernel for fliping y of images


char * open_raw_source(char * file_name){
    FILE * f = fopen(file_name, "r");
    fseek(f, 0L, SEEK_END);
    unsigned long size = ftell(f);
    fseek(f, 0, SEEK_SET);
    char * cur_char = malloc((size+1)*sizeof(char));
    char * raw_source = cur_char;
    if (f != NULL) {
        size_t newLen = fread(raw_source, sizeof(char), size, f);
        if ( ferror( f ) != 0 ) {
            fputs("Error reading file", stderr);
        } else {
            raw_source[newLen++] = '\0'; /* Just to be safe. */
        }
    }
    fclose(f);
    return raw_source;
}


void init_cl(){
    char * raw_source = open_raw_source("kernel.cl");
    // Set up platform and GPU device
    cl_uint numPlatforms;
     // Find number of platforms
    err = clGetPlatformIDs(0, NULL, &numPlatforms);
    if (numPlatforms == 0)
    {
        printf("Found 0 platforms!\n");
        return EXIT_FAILURE;
    }

    cl_platform_id * Platform = malloc(numPlatforms*sizeof(cl_platform_id));
    err = clGetPlatformIDs(numPlatforms, Platform, NULL);
    // Secure a GPU
    for (int i = 0; i < numPlatforms; i++)
    {
        err = clGetDeviceIDs(Platform[i], DEVICE, 1, &device_id, NULL);
        if (err == CL_SUCCESS)
        {
            break;
        }
    }
    // Create a compute context 
    context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
    // Create a command queue
    commands = clCreateCommandQueueWithProperties(context, device_id, 0, &err);
    // Create the compute program from the source buffer
    program = clCreateProgramWithSource(context, 1, (const char **) & raw_source, NULL, &err);
    
    // Build the program  
    err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        size_t len;
        char buffer[2048];

        printf("Error: Failed to build program executable!\n%i\n", err);
        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
        printf("%s\n", buffer);
        // return EXIT_FAILURE;
    }
    ko_vmul = clCreateKernel(program, "multiply", &err);
    ko_vadd = clCreateKernel(program, "add", &err);
    ko_mat_conv_gray = clCreateKernel(program, "convolve_gray", &err);
    ko_gray = clCreateKernel(program, "rgb_to_gray", &err);
    ko_resize = clCreateKernel(program, "resize_img", &err);
    ko_flip_y = clCreateKernel(program, "flip_mat_y", &err);
    free(Platform);
    free(raw_source);
}

void finish_cl(){
    clReleaseProgram(program);
    clReleaseKernel(ko_vmul);
    clReleaseCommandQueue(commands);
    clReleaseContext(context);
}

void vec_op_gpu(const float * a, const float * b, float * c, const unsigned int buffer_size, cl_kernel kernel){
    cl_mem d_a;                     // device memory used for the input  a vector
    cl_mem d_b;                     // device memory used for the input  b vector
    cl_mem d_c;                     // device memory used for the output c vector


    d_a  = clCreateBuffer(context,  CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,  buffer_size, a, &err);
    d_b  = clCreateBuffer(context,  CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,  buffer_size, b, &err);

    // Create the output arrays in device memory
    d_c  = clCreateBuffer(context,  CL_MEM_READ_WRITE, buffer_size, NULL, &err);
    
    const int count = (buffer_size/sizeof(float));

    // Enqueue kernel - first time
    // Set the arguments to our compute kernel
    
    err  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_a);
    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_b);
    err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_c);
    err |= clSetKernelArg(kernel, 3, sizeof(unsigned int), &count);
    global = count;
    err = clEnqueueNDRangeKernel(commands, kernel, 1, NULL, &global, NULL, 0, NULL, NULL);

    err = clEnqueueReadBuffer( commands, d_c, CL_TRUE, 0, sizeof(float) * count, c, 0, NULL, NULL );  
    clReleaseMemObject(d_a);
    clReleaseMemObject(d_b);
    clReleaseMemObject(d_c);
}

void convolve(const Matrix mat, const Matrix kern, Matrix * out_m, const cl_kernel comp_kernel){ 
    cl_mem d_mat;                     
    cl_mem d_kern; 
    cl_mem d_mat_data;
    cl_mem d_kern_data;  
    cl_mem d_out;                  

    unsigned int size_of_mat_data = (mat.member_size)*(mat.width)*(mat.height);
    void * p_out = out_m->data;

    d_mat  = clCreateBuffer(context,  CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
      sizeof(Matrix) - sizeof(void *), &mat, &err);
    d_kern  = clCreateBuffer(context,  CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
      sizeof(Matrix) - sizeof(void *), &kern, &err);
    d_kern_data  = clCreateBuffer(context,  CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
      kern.member_size * kern.height * kern.width, kern.data, &err);
    d_mat_data  = clCreateBuffer(context,  CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
      size_of_mat_data, mat.data, &err);
      
    d_out  = clCreateBuffer(context,  CL_MEM_READ_WRITE, 
     size_of_mat_data, NULL, &err);
    
    const int count = (mat.height*mat.width);

    // Enqueue kernel - first time
    // Set the arguments to our compute kernel
    
    err  = clSetKernelArg(comp_kernel, 0, sizeof(cl_mem), &d_mat);
    err |= clSetKernelArg(comp_kernel, 1, sizeof(cl_mem), &d_kern);
    err |= clSetKernelArg(comp_kernel, 2, sizeof(cl_mem), &d_mat_data);
    err |= clSetKernelArg(comp_kernel, 3, sizeof(cl_mem), &d_kern_data);
    err |= clSetKernelArg(comp_kernel, 4, sizeof(cl_mem), &d_out);
    err |= clSetKernelArg(comp_kernel, 5, sizeof(unsigned int), &count);
    global = count;
    err = clEnqueueNDRangeKernel(commands, comp_kernel, 1, NULL, &global, NULL, 0, NULL, NULL);

    err = clEnqueueReadBuffer( commands, d_out, CL_TRUE, 0, size_of_mat_data, p_out, 0, NULL, NULL );  
    clReleaseMemObject(d_mat);
    clReleaseMemObject(d_kern);
    clReleaseMemObject(d_mat_data); 
    clReleaseMemObject(d_kern_data); 
    clReleaseMemObject(d_out);
    out_m->height =  mat.height;
    out_m->width =  mat.width;
    out_m->member_size = mat.member_size;
}
void action_on_image(const Matrix mat, Matrix * out_m, const cl_kernel comp_kernel){                  
    cl_mem d_mat;                     
    cl_mem d_mat_data;
    cl_mem d_out;  
    cl_mem d_out_data;                  

    unsigned int size_of_out_data = (out_m->member_size)*(out_m->width)*(out_m->height);
    void * p_out = out_m->data;

    d_mat  = clCreateBuffer(context,  CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
      sizeof(Matrix) - sizeof(void*), &mat, &err);
    d_out  = clCreateBuffer(context,  CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
      sizeof(Matrix) - sizeof(void*), out_m, &err);
    d_mat_data  = clCreateBuffer(context,  CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
      (mat.member_size)*(mat.width)*(mat.height), mat.data, &err);
      
    d_out_data  = clCreateBuffer(context,  CL_MEM_READ_WRITE, 
     size_of_out_data, NULL, &err);
    
    const int count = (out_m->height) * (out_m->width);
    
    // Enqueue kernel - first time
    // Set the arguments to our compute kernel
    
    err  = clSetKernelArg(comp_kernel, 0, sizeof(cl_mem), &d_mat);
    err |= clSetKernelArg(comp_kernel, 1, sizeof(cl_mem), &d_out);
    err |= clSetKernelArg(comp_kernel, 2, sizeof(cl_mem), &d_mat_data);
    err |= clSetKernelArg(comp_kernel, 3, sizeof(cl_mem), &d_out_data);
    err |= clSetKernelArg(comp_kernel, 4, sizeof(unsigned int), &count);
    global = count;
    err = clEnqueueNDRangeKernel(commands, comp_kernel, 1, NULL, &global, NULL, 0, NULL, NULL);

    err = clEnqueueReadBuffer( commands, d_out_data, CL_TRUE, 0, size_of_out_data, p_out, 0, NULL, NULL );  
    clReleaseMemObject(d_mat);
    clReleaseMemObject(d_mat_data); 
    clReleaseMemObject(d_out_data); 
    clReleaseMemObject(d_out);
}
void rgb_to_gray(const Matrix img, Matrix * out_m){
    out_m->height =  img.height;
    out_m->width =  img.width;
    out_m->member_size = sizeof(char);
    action_on_image(img, out_m, ko_gray);
}
void flip_y(const Matrix img, Matrix * out_m){
    out_m->height =  img.height;
    out_m->width =  img.width;
    out_m->member_size = img.member_size;
    action_on_image(img, out_m, ko_flip_y);
}
void resize_image(const Matrix img, Matrix * out_m){
    action_on_image(img, out_m, ko_resize);
}
void convolve_gray(const Matrix mat, const Matrix kern, Matrix * out_m){
    convolve(mat, kern, out_m, ko_mat_conv_gray);
}
void add_on_gpu(const float * a, const float * b, float * c, const unsigned int buffer_size) {
    vec_op_gpu(a, b, c, buffer_size, ko_vadd);
}
void mull_on_gpu(const float * a, const float * b, float * c, const unsigned int buffer_size) {
    vec_op_gpu(a, b, c, buffer_size, ko_vmul);
}

