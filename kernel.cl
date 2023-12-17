
typedef struct matrix{
    unsigned int width;
    unsigned int height;
    unsigned int member_size;

} Matrix;


__kernel void add(global const float * a, global const float * b, global float * c) {
    int id = get_global_id(0);
    c[id] = a[id] + b[id];
}


__kernel void multiply(global const float * a, global const float * b, global float * c) {
    int id = get_global_id(0);
    c[id] = (a[id] * b[id]);
}

__kernel void convolve_gray(global const Matrix * mat, global const Matrix * kern,
                             global void * mat_data, global const void * kern_data) { 
    // TODO: fix to actualy convolve as matrix and not 1d list
    // TODO: fix to use different pointers for input and output
    int id = get_global_id(0);
    global char * p_cur = (global char *)(mat_data + id);
    char conv = 0;
    long mat_start = mat_data;
    long mat_end = mat_data + (mat->height*mat->width);
    long i_start = p_cur - ((kern->width * kern->height)/2);
    for (long i = i_start;
             i < (long)(p_cur +(kern->width * kern->height)/2 + 1);
             i++) {
        conv += ((i >= mat_start) && (i <= mat_end)) * // makes sure value is 0 if outside bounds
                ((*(char *)i) * (*(global char*)(kern_data+(i - i_start))));
    }
    *(p_cur) = conv;
}
