## docker build
docker build -t my-ubuntu-image .

## docker run
docker run --gpus all -it --rm -v $(pwd):/workspace -w /workspace my-ubuntu-image bash


docker build -t crowd-nav-dev .

docker run -it --rm \
  -v /home/yufei/research/crowd-follow-nav:/workdir/crowd-follow-nav \
  crowd-nav-dev