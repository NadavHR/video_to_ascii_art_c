CC = gcc

COMPILER_OPTIONS = -g 

SOURCE = main.c
TARGET = main.exe

INCLUDE = "-I${OCL_ROOT}\include" "-I${DEV_STUFF}"
LINK = "-L${OCL_ROOT}\lib\x86_64" -lopenCL


all:
	${CC} ${COMPILER_OPTIONS} ${SOURCE} -o ${TARGET} ${INCLUDE} ${LINK} 