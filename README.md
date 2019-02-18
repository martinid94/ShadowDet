# ShadowDet

ShadowDet is a project for Computer Vision class, University of Padua, Master's degree in Computer Science Engineering A.Y. 2017/2018.
This assignment aims at implementing an effective strategy that segments the input image in shadow and non-shadow areas. ShadowDet is developed in C++ with OpenCV 3.4.3 library and CMake building environment in Ubuntu 18.04.

### Folder structure
Folder structure of ShadowDet:
```
├── build                   # contains building files
├── data                    # contains some images used for testing
├── include 		    # contains header .h file
├── results		    # contains the output masks
├── src                     # contains source .cpp file
├── CMakeLists.txt
├── LICENSE
└── README.md
```

### Compile ShadowDet

At first, OpenCV and CMake are required. Then clone or download this repository. 
```sh
    $ cd ShadowDet/build
    $ rm -r *
    $ cmake ..
    $ make
```

### Acknowledgements

If you use this work, please cite this repository as a reference. 
