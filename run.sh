if [[ "$1" == *.cu ]]; then
  nvcc -arch=sm_75 "$1" -o cuda_run.bin && ./cuda_run.bin
elif [[ "$1" == *.cpp ]]; then
  g++ "$1" -o cpp_run.bin && ./cpp_run.bin
fi


# todo:
# abstractnvprof ./cuda_run.bin
# abstract nvprof  --print-gpu-trace ./cuda_run.bin
