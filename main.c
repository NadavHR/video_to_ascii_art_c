#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include "cl_utils.c"
#include <CL/cl.h>


void test_convolution(){
    init_cl();
    char dm[] = {0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 1, 1, 1, 1, 1, 0,
                 0, 0, 1, 1, 1, 1, 1, 0,
                 0, 0, 1, 1, 1, 1, 1, 0,
                 0, 0, 1, 1, 1, 1, 1, 0,
                 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0, 0, 0,};
    Matrix mat = {
        .data = dm,
        .height = 10,
        .width = 8,
        .member_size = sizeof(char)
    };
    char dout[sizeof(dm)];
    char dk[] = {1, 0, -1,
                 2, 0, -2,
                 1, 0, -1};
   
    Matrix kern = {
        .data = dk,
        .height = 3,
        .width = 3,
        .member_size = sizeof(char)
    };
    // print_mat_gray(&mat);
    Matrix output = {
        .data = dout,
        .height = 5,
        .width = 5,
        .member_size = sizeof(char)
    };
    convolve_gray(mat, kern, &output);
    print_mat_gray(output);
    finish_cl();
 
}
void test_img(){
    init_cl();
    Matrix img = openBMPFile("img.bmp");
    char resized_data[ 100*200*img.member_size];
    Matrix resized = {
        .data = resized_data,
        .height = 200,
        .width = 100,
        .member_size = img.member_size
    };
    resize_image(img, &resized);
    flip_y(resized, &resized);

    char gray_buffer[resized.width*resized.height];
    Matrix gray = {.data = gray_buffer,};
    rgb_to_gray(resized, &gray);
    // flip_y(gray, &gray);
    print_mat_gray(gray);

    finish_cl();
    free(img.data);
}
void test_resize(){
    init_cl();
    char dm[] = { 0, 0, 0,  0, 0, 0,  0, 0, 0,
                  0, 0, 100,  100, 100, 100,  100, 0, 0,
                  0, 100, 0,  0, 0, 0,  0, 100, 0,
                  0, 100, 0,  0, 0, 0,  0, 100, 0,
                  0, 100, 0,  0, 0, 0,  0, 100, 0,
                  0, 100, 0,  0, 0, 0,  0, 100, 0,
                  0, 100, 0,  0, 0, 0,  0, 100, 0,
                  0, 0, 100,  100, 100, 100,  100, 0, 0,
                  0, 0, 0,  0, 0, 0,  0, 0, 0,
                  0, 0, 0,  0, 0, 0,  0, 0, 0};
    
    Matrix mat = {
        .data = dm,
        .height = 9,
        .width = 3,
        .member_size = sizeof(char) * 3
    };
    char dout[3*5*4];
    Matrix output = {
        .data = dout,
        .height = 4,
        .width = 5,
        .member_size = sizeof(char) * 3
    };
    resize_image(mat, &output);
    print_img(output);
    finish_cl();
}

void test_sobel(){
    init_cl();
    Matrix img = openBMPFile("img.bmp");
    char resized_data[ 200*100*img.member_size];
    Matrix resized = {
        .data = resized_data,
        .height = 200,
        .width = 100,
        .member_size = img.member_size
    };
    resize_image(img, &resized);
    flip_y(resized, &resized);

    char gray_buffer[resized.width*resized.height];
    Matrix gray = {.data = gray_buffer,};
    rgb_to_gray(resized, &gray);


    char d_k_sobel_x[] = {-1, 0, 1,
                        -2, 0, 2,
                        -1, 0, 1};
   
    Matrix kern = {
        .data = d_k_sobel_x,
        .height = 3,
        .width = 3,
        .member_size = sizeof(char)
    };
    char buffer_out_x[resized.width*resized.height];
    Matrix sobel_x = {
        .data = buffer_out_x,
        .height = gray.height,
        .width = gray.width,
        .member_size = 1
    };
    convolve_gray(gray, kern, &sobel_x);
    char d_k_sobel_y[] = { 1,  2,  1,
                            0,  0,  0,
                            -1, -2, -1};

    kern.data = d_k_sobel_y;
    char buffer_out_y[resized.width*resized.height];
    Matrix sobel_y = {
        .data = buffer_out_y,
        .height = gray.height,
        .width = gray.width,
        .member_size = 1
    };
    convolve_gray(gray, kern, &sobel_y);
    char interlaced_buffer[2*resized.width*resized.height];
    Matrix interlaced = {.data = interlaced_buffer,
                         .member_size = 2};
    interlace_bytes(sobel_x, sobel_y, &interlaced);
    print_img(sobel_y); 
    finish_cl();
    free(img.data);
}

int main(void){
    // test_resize();
    // cl_test();
    // test_img();
    test_sobel();
    return 0;
}