1. 图像像素得足够大，确保每个patch不会包含很多原子
2. atomsegnet效果似乎更好，消除背景光照等

ori后缀：低分辨率图（256x256）
huge后缀：高分辨率图（1024x1024）
pos后缀：经过atomsegnet处理之后的图片

dinov3_finetune_ref 是 github 克隆下来的微调项目，用作参考
dinov3_finetune 是自己的微调项目
student_weights 存了用不同训练集训练的 loRA 权重，combined 指的是同时用了没pos和pos的总计（2H+1T）四个文件夹的图片


关键洞察：在医学影像等领域，由于图像内容的同质性较高（例如都是肺部CT），过高的DINO Loss权重可能导致模型过度关注背景的一致性。
此时，适当提高iBOT Loss的权重，强迫模型关注病灶区域的局部纹理差异，往往能带来更好的效果 。

Budget Tuning：除了LoRA（秩32），还解冻了最后1个Transformer Block。