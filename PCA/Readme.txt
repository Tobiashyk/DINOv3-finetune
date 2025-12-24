1. 图像像素得足够大，确保每个patch不会包含很多原子
2. atomsegnet效果似乎更好，消除背景光照等

ori后缀：低分辨率图（256x256）
huge后缀：高分辨率图（1024x1024）
pos后缀：经过atomsegnet处理之后的图片

dinov3_finetune_ref 是 github 克隆下来的微调项目，用作参考
dinov3_finetune 是自己的微调项目
student_weights 存了用不同训练集训练的 loRA 权重，combined 指的是同时用了没pos和pos的总计（2H+1T）四个文件夹的图片


