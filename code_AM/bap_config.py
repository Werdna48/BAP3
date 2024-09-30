def setup_bap3_config():
    import pytensor
    
    pytensor.config.gcc__cxxflags = '-L/opt/miniconda3/envs/bap3/lib -O3 -march=native'
    pytensor.config.cxx = '/usr/bin/clang++'
    pytensor.config.blas__ldflags = '-framework Accelerate'