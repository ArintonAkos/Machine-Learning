# Machine Learning Projects

## Project Setup Guide

1. Create a new directory for the project
2. Create a `/build` directory
3. Create a `CMakeLists.txt` file
4. Create a `main.cpp` file
5. Run `cmake -DCMAKE_EXPORT_COMPILE_COMMANDS=1 ..`
6. Run `cp compile_commands.json ..`
7. Run `cd ..`
8. Run `make`
9. Run `./project_name`
