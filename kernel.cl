
typedef struct matrix{
    unsigned int width;
    unsigned int height;
    unsigned int member_size;

}__attribute__((packed)) Matrix;


__kernel void add(global const float * a, global const float * b, global float * c) {
    int id = get_global_id(0);
    c[id] = a[id] + b[id];
    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
}


__kernel void multiply(global const float * a, global const float * b, global float * c) {
    int id = get_global_id(0);
    c[id] = (a[id] * b[id]);
    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
}

__kernel void convolve_gray(global const Matrix * mat, global const Matrix * kern,
                            const global char * mat_data, global const char * kern_data,
                            global char * out) { 
    int id = get_global_id(0);
    char conv = 0;
    int x_start = (id % mat->width)-((kern->width)/2);
    int y_start = (id / mat->width)-((kern->height)/2);
    unsigned int mat_size = (mat->height*mat->width*sizeof(char));
    for (unsigned int i = 0; i < kern->width; i++){
        for (unsigned int j = 0; j < kern->height; j++){
            int cur_index = ((x_start + i) + ((y_start + j)*mat->width));
            if ((cur_index >= 0) && (cur_index < mat_size)) { // makes sure value is 0 if out of matrix bounds)
                conv +=((mat_data[cur_index]) * // current place in mat
                (kern_data[i + (j*kern->width)])); // current place in kern
            }
        }
    }
    // printf("%d", conv);
    out[id] = conv;
    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
}
