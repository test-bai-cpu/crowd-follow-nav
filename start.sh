## docker build
docker build -t my-ubuntu-image .

## docker run
docker run --gpus all -it --rm -v $(pwd):/workspace -w /workspace my-ubuntu-image bash


docker build -t crowd-nav-dev .

docker run -it --rm \
  -v /home/yufei/research/crowd-follow-nav:/workdir/crowd-follow-nav \
  crowd-nav-dev



# For deepgear
docker run -it --rm \
  --gpus all \
  --name crowd-follower \
  -v /home/yzu/crowd-follow-docker/crowd-follow-nav:/workdir/crowd-follow-nav \
  dockerhuaniden/crowd-nav-dev




  # -p 6007:6007 \

tensorboard --logdir results/logs --port 6007
tensorboard --logdir /workdir/crowd-follow-nav/results --port 6007 --bind_all



ssh -L 6007:localhost:6007 yzu@130.243.124.57