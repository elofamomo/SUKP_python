sudo docker run -d -p 8080:8080 --name wandb-local -v wandb:/vol wandb/local
wandb login --host http://localhost:8080