
typedef struct matrix{
    unsigned int width;
    unsigned int height;
    unsigned int member_size;

}__attribute__((packed)) Matrix;
typedef struct rgb_pixel{
    unsigned char red;
    unsigned char green;
    unsigned char blue;
}__attribute__((packed)) RGB_pixel;

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

__kernel void rgb_to_gray(global const Matrix * img, global const Matrix * gray, global const RGB_pixel * rgb_data, global signed char * out_gray){
    // 0.299 ∙ Red + 0.587 ∙ Green + 0.114 ∙ Blue
    unsigned int id = get_global_id(0);
    RGB_pixel pixel = rgb_data[id];
    out_gray[id] = (char)((0.299*(float)pixel.red) +
                          (0.587*(float)pixel.green)+
                          (0.114*(float)pixel.blue) - 128); // -128 to make it so 128 is white and -128 is black
    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
}

__kernel void resize_RGB_img(global const Matrix * img, global const Matrix * out, global const RGB_pixel * rgb_data, global RGB_pixel * out_data){
    // whenever we multiply by 2 and than divide by 2 its so 0.5 or higher will get rounded to 1
    
    unsigned int id = get_global_id(0);
    float scale_w = ((float)(img->width)) / ((float)(out->width));
    float scale_h = ((float)(img->height)) / ((float)(out->height));
    unsigned int x = ((unsigned int)((float)((id % out->width)) * 2 * scale_w + FLT_EPSILON))/2;
    unsigned int y = ((unsigned int)((float)((id / out->width)) * 2 * scale_h + FLT_EPSILON))/2;
    unsigned int x_radius = ((unsigned int)max(2*scale_w+FLT_EPSILON, (float)2))/2;
    unsigned int y_radius = ((unsigned int)max(2*scale_h+FLT_EPSILON, (float)2))/2;
    float average_s = 1.0 / (float)(x_radius*y_radius);
    float average_red = 0;
    float average_green = 0;
    float average_blue = 0;
    
    for (int i = x; i < x+x_radius; i++){
        for (int j = y; j < y+y_radius; j++){
            RGB_pixel pixel = rgb_data[(i + j*img->width)];
            
            average_red += (float)(pixel.red);
            average_green += (float)(pixel.green);
            average_blue += (float)(pixel.blue);
            // }
        }
    }
    out_data[id].red  =  (unsigned char)(((int)(FLT_EPSILON+2*average_red*average_s)) /2);
    out_data[id].green =(unsigned char)(((int)(FLT_EPSILON+2*average_green*average_s))/2);
    out_data[id].blue = (unsigned char)(((int)(FLT_EPSILON+2*average_blue*average_s))/2);
    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
}
