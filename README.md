<div align="center">
<h1>StreamPETR_moe</h1>
<h3>[ICCV2023] Exploring Object-Centric Temporal Modeling for Efficient Multi-View 3D Object Detection</h3>
</div>

## Introduction

TODO.


config:修改部分  backbone 输出改成了out_indices=(1,2,3)，随之 neck部分输入改成了in_channels=[512, 1024, 2048]  neck输出改成了 num_outs=4   → 实现了deformable的输入
projects/mmdet3d_plugin/models/dense_heads/stream_petr_head.py：class StreamPETRHead(AnchorFreeHead): def __init__(self,   加入两个num_levels = 4,
StreamPETRHead 的 forward 函数，使其支持多层特征图的遍历处理:把原来只处理单层特征的逻辑，改成遍历所有层级（Stride 8, 16, 32, 64），为每一层分别生成 Memory 和 3D PE
主要改动点：

输入处理：不再假设 data['img_feats'] 是单个 Tensor，而是把它当作列表处理。

循环遍历：增加了 for lvl, x in enumerate(mlvl_feats): 循环。

动态生成 Grid：不再使用外部传入的固定 memory_center，而是在循环内部根据当前层级的 H, W 实时生成 current_center，确保 PE 对应正确的分辨率。

打包列表：将生成的 mlvl_memories 和 mlvl_pos_embeds 打包成列表传给 Transformer。