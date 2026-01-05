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



1. 新增文件：projects/mmdet3d_plugin/models/utils/moe_router.py
修改内容：新增了一个 StreamMoERouter 类。

作用：作为路由模块，它接收 (B, Nq, C) 的 Query Embeddings，经过 MLP 映射后输出 (B, Nq, Num_Levels) 的路由权重。在训练时使用 Gumbel Softmax 进行可微采样，推理时使用 Top-K 选择，决定每个 Query 应该关注哪些特征层。

2. 修改文件：projects/mmdet3d_plugin/models/dense_heads/streampetr_head.py
__init__ 部分：在 StreamPETRHead 初始化中实例化了 self.router = StreamMoERouter(...)。

forward 部分：

特征列表化：将输入的 mlvl_feats 处理逻辑改为循环遍历，为每一层特征（P3-P6等）动态生成对应的 Grid Center 和 3D PE，并将结果存入 列表 (mlvl_memories, mlvl_pos_embeds)。

计算权重：在 temporal_alignment 获取到融合历史信息的 query_pos 后，调用 self.router(query_pos) 计算出 routing_weights。

传参变更：在调用 self.transformer 时，传入特征列表、PE 列表以及计算好的路由权重 routing_weights。

3. 修改文件：projects/mmdet3d_plugin/models/utils/petr_transformer.py
PETRTemporalTransformer.forward 部分：

接口更新：参数改为接收列表形式的 mlvl_memories 和 mlvl_pos_embeds，以及 routing_weights。
数据预处理：循环遍历特征列表，对每一层的 Tensor 进行 transpose 和展平操作，打包成列表传给 Decoder。

PETRTemporalDecoderLayer._forward 部分 (核心 MoE 逻辑)：

Cross Attention 改造：在 cross_attn 分支中增加判断——如果 key 是列表且有 routing_weights，则遍历每一层特征分别计算 Attention，最后利用 routing_weights 对各层结果进行加权求和，实现多尺度特征的动态融合。

PETRTemporalDecoderLayer.forward 部分：

透传参数：更新了入口函数签名，增加了 routing_weights 参数，确保它能穿过 Checkpoint 机制传递给内部的 _forward 函数。