#include <stdio.h>
#include <stdlib.h>
 
#define BMP_HEADER_SIZE 54
#define BMP_WIDTH_INDEX 18
#define BMP_HEIGHT_INDEX 22
#define OUT_FRAME_RATE 24


typedef struct matrix{
    void * data;
    unsigned int width;
    unsigned int height;
    unsigned int member_size;

} Matrix;

Matrix readBMP(char* file_bin)
{
    int i;
    
    // unsigned char info[BMP_HEADER_SIZE];
    // // read the 54-byte header
    // fread(info, sizeof(unsigned char), BMP_HEADER_SIZE, file_bin); 
    // extract image height and width from header
    int width = *(int*)&file_bin[BMP_WIDTH_INDEX];
    int height = *(int*)&file_bin[BMP_HEIGHT_INDEX];
    // allocate 3 bytes per pixel
    int size = 3 * width * height;
    unsigned char* data = malloc(sizeof(unsigned char) * size);
    // read the rest of the data at once
    // fread(data, sizeof(unsigned char), size, f); 
    // fclose(f);
    memcpy(data, file_bin+BMP_HEADER_SIZE, size);
    // for(i = 0; i < size; i += 3)
    // {
    //         // flip the order of every 3 bytes
    //         unsigned char tmp = data[i];
    //         data[i] = data[i+2];
    //         data[i+2] = tmp;
    // }
    Matrix mat = {
        .data = data,
        .height = height,
        .width = width
    };
    return mat;
}

Matrix readBMPFile(unsigned char* filename){

    FILE* f = fopen(filename, "rb");
    fseek(f, 0, SEEK_END);
    long fsize = ftell(f);
    fseek(f, 0, SEEK_SET); 
    char *data = malloc(fsize);
    fread(data, fsize, 1, f);
    fclose(f);
    Matrix mat = readBMP(data);
    free(data);
    return mat;

}

void video_to_bmp_folder() {
    system("ffmpeg -i vid.mp4 -vf fps=24 frames/%04d.bmp"); // NOTE: make sure the folder exists
}

void clean_bmp_folder() {
    system("del /q frames\\*"); // windows
}