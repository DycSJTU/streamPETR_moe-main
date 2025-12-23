<div align="center">
<h1>StreamPETR_moe</h1>
<h3>[ICCV2023] Exploring Object-Centric Temporal Modeling for Efficient Multi-View 3D Object Detection</h3>
</div>

## Introduction

TODO.


config:修改部分  backbone 输出改成了out_indices=(1,2,3)，随之 neck部分输入改成了in_channels=[512, 1024, 2048]  neck输出改成了 num_outs=4   → 实现了deformable的输入
projects/mmdet3d_plugin/models/dense_heads/stream_petr_head.py：class StreamPETRHead(AnchorFreeHead): def __init__(self,   加入两个num_levels = 4,