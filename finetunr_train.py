import os

import json
import shutil
from PIL import Image
from PIL import ImageEnhance
import random


from modelscope.utils.constant import Tasks
from modelscope.pipelines import pipeline
from modelscope.metainfo import Metrics, Trainers
from modelscope.msdatasets import MsDataset
from modelscope.trainers import build_trainer
from modelscope.utils.constant import ModelFile
from modelscope.outputs import OutputKeys
from modelscope.hub.snapshot_download import snapshot_download


train_batch = 2
evl_batch = 2
epoch = 5


finetune_cfg = \
            {'framework': 'pytorch',  # 运行框架
             'task': 'ocr-recognition',  # 运行任务
             'model': {'type': 'ofa',   # 模型的类型（一般是模型backbone/骨架）这个key下面主要放了如何构建推断的ofa任务
                       'beam_search': {# OFA是基于seq2seq架构，解码时是自回归的方法，这里配置为beam seach的解码策略
                                       'beam_size': 5,   #  解码策略beamseach的参数
                                       'max_len_b': 64,  # # 基于fairseq的beamsearch解码器，解码长度是ax + b，一般a是0，b是16相当于最长生成16个token
                                       'min_len': 1,  # 生成的最小长度
                                       'no_repeat_ngram_size': 0}, # "no_repeat_ngram_size": 0 # 禁止重复生成长度为本参数值的ngram，0代表不禁止重复生成。注：如果用beamsearch解分类任务，请务必将此参数设置为0
                       'seed': 7,  # 模型随机种子
                       'max_src_length': 128,  # 输入最大token数
                       'language': 'zh',  # 输入输出语言
                       'gen_type': 'generation',  # 推断的方式：生成式，也可以是遍历式，注：这对kv和任务绑定。
                       'patch_image_size': 480,  # 将image resize到 (patch_image_size, patch_image_size)
                       'is_document': False,  # 任务特有，ocr针对不同场景预处理不一样
                       'max_image_size': 480,  # 图片最大大小
                       'imagenet_default_mean_and_std': False},  # 是否使用imagenet的均值和方差处理图像。
             'pipeline': {'type': 'ofa-ocr-recognition'},  # pipeline的类型
             'dataset': {'column_map': {'text': 'label', 'image': 'image'}},  # 针对数据集合模型预处理预定义的字段不同，这里做一个映射,key是数据集字段名，value是预处理采用的字段名
             'train': {  # finetune相关配置
                      'work_dir': 'work/ckpts/recognition',  # finetune的log,ckpt等存储目录
                       # 'launcher': 'pytorch',  # 多机多卡的启动方式
                       'max_epochs': epoch,  # 训练轮数
                       'use_fp16': True,  # 是否使用fp16加速
                       'dataloader': {'batch_size_per_gpu': train_batch, 'workers_per_gpu': 0}, # 数据下载器的配置
                       'lr_scheduler': {'name': 'polynomial_decay',  # 学习率配置，不同学习器参数不同。
                                        'warmup_proportion': 0.01,
                                        'lr_end': 1e-07},
                       'lr_scheduler_hook': {'type': 'LrSchedulerHook', 'by_epoch': False},  # ms使用hook进行finetune时各种行为管理，具体来说根据hook是根据step还是epoch以及具体步数进行相应行为的调用
                       'optimizer': {'type': 'AdamW', 'lr': 5e-04, 'weight_decay': 0.01},  # optimizer的配置
                       'optimizer_hook': {'type': 'TorchAMPOptimizerHook',  # optimizer的hook，这里用了fp16所以也采用了torch的amp技术
                                          'cumulative_iters': 2,  # 梯度累积，一般当显存不够大的时候，可以多次forward和backward，最后再进行优化器更新，从而达到放大batch的作用（对于batch_norm会有影响，不等同于同样batch size）
                                          'grad_clip': {'max_norm': 1.0, 'norm_type': 2},  # 梯度裁剪（为了防止梯度过大）
                                          'loss_keys': 'loss'},  # 模型在每个step后产出的loss记录在key-value对立面，ofa loss的key是loss
                       'criterion': {'name': 'AdjustLabelSmoothedCrossEntropyCriterion',  # 这里criterion相当于是计算loss的全部逻辑，仿照了fairseq的写法
                                     'constraint_range': None,  # ofa特定的参数，因为生成时不同任务的token在词表里面的位置不一样，所以会根据任务给出范围，num,num形式（一般是不同token的起始和结束位置，如bin,image code等）
                                     'drop_worst_after': 0,  # 训练技巧的一种，多少步之后会将loss最大的token去除掉，不计算loss
                                     'drop_worst_ratio': 0.0,  # 同上，去除的比例
                                     'ignore_eos': False,  # 计算loss时EOS是不是计算忽略掉
                                     'ignore_prefix_size': 0,  # 计算loss时忽略前面多少个token
                                     'label_smoothing': 0.1,  # 交叉熵在做分类任务时的标签平滑技术，一般能有效提升效果
                                     'reg_alpha': 1.0,
                                     'report_accuracy': False,
                                     'sample_patch_num': 196,
                                     'sentence_avg': False,  # loss是token维度还是样本维度
                                     'use_rdrop': True},  # 是否使用rdrop，一种在有随机因素情况下连续两次forward的结果要变得更近的技巧，会放慢训练速度。
                       'hooks': [{'type': 'BestCkptSaverHook',  # save最优 ckpt的hook
                                  'metric_key': 'accuracy',  # 选取哪个metric作为衡量（因为有可能有多个metric值）
                                  'interval': 100},  # 多久存储一次，默认是按照epoch计算的
                                 {'type': 'TextLoggerHook', 'interval': 1},  # log的hook
                                 {'type': 'IterTimerHook'},  # 即时的hook
                                 {'type': 'EvaluationHook', 'by_epoch': True, 'interval': 1}]},  # 评估的hook
             'evaluation': {'dataloader': {'batch_size_per_gpu': evl_batch, 'workers_per_gpu': 0},  # eval数据下载器的参数
                            'metrics': [{'type': 'accuracy'}]},  # 评估时使用的方法，这里是acc
             'preprocessor': []}  # 预处理配置，这里为空（ofa有统一的预处理方式）

# 预训练模型没有绑定任务，所以想从预训练模型开始训练，可以较为便捷的先从具体任务的model处获取
WORKSPACE = "workspace"
ocr_path = 'damo/ofa_ocr-recognition_scene_base_zh'
pretrained_path = snapshot_download(ocr_path)  # 下载模型至缓存目录，并返回目录


# ofa通用的pretrained模型，未针对OCR场景做过调优
# pretrained_path = 'damo/ofa_pretrain_base_zh'  # 预训练模型的模型id
# pretrained_path = 'damo/ofa_ocr-recognition_scene_base_zh'  # OCR
# pretrained_path = snapshot_download(pretrained_path, revision='v1.0.0')

print('dayin:', pretrained_path)

# shutil.copy(os.path.join(pretrained_path, ModelFile.CONFIGURATION),  # 将任务的配置覆盖预训练模型的配置
#             os.path.join(pretrained_path, ModelFile.CONFIGURATION))

test_image = "train_data/test_5/0.jpg"

ofa_pipeline = pipeline(task=Tasks.ocr_recognition, model=pretrained_path)  # 可以先测试下预训练模型的效果
result = ofa_pipeline(test_image)

# 期待这里打印出的ocr效果不理想
print('蝶恋花', result[OutputKeys.TEXT])  # 输出效果不好，OCR_path,还不错。


os.makedirs(WORKSPACE, exist_ok=True)
# 写一下配置文件
config_file = os.path.join(WORKSPACE, ModelFile.CONFIGURATION)
with open(config_file, 'w') as writer:
    json.dump(finetune_cfg, writer, indent="\t")


def my_img_augument(image):
    '''
        PIL 类型数据的数据增强，增强参数 1，是原始图片，不做任何变化。
    :param image:
    :return: 增强后的PIL型数据。
    '''
    if random.random() < 0.3:
        # 亮度增强，参考范围：[0.8,2]
        enh_bri = ImageEnhance.Brightness(image)
        brightness = random.uniform(0.8, 2.5)
        image = enh_bri.enhance(brightness)

    if random.random() < 0.16:
        # 对比度增强，参考范围：[0.5,2]
        enh_con = ImageEnhance.Contrast(image)
        contrast = random.uniform(0.7, 2)
        image = enh_con.enhance(contrast)

    if random.random() < 0.16:
        # 锐度，参考范围：[0.5,3]
        enh_sha = ImageEnhance.Sharpness(image)
        sharpness = random.uniform(0.5, 3)
        image = enh_sha.enhance(sharpness)

    return image


def map_func(example):
    # do something to example
    image = Image.open(example['img_path'])
    example['image'] = my_img_augument(image)  # 增加了数据增强。
    return example


# train_data = MsDataset.load('ocr_fudanvi_zh', namespace='modelscope', split='train')


# 训练数据
HW_mydata_train = 'train_data/test_5.csv'
HW_mydata_test = 'train_data/test_5.csv'

train_ds = MsDataset.load(HW_mydata_train)
ds_train = train_ds.ds_instance.map(map_func)

test_ds = MsDataset.load(HW_mydata_test)
ds_test = test_ds.ds_instance.map(map_func)


args = dict(
    model=pretrained_path,
    work_dir=WORKSPACE,
    train_dataset=ds_train,
    eval_dataset=ds_test,
    metrics=[Metrics.accuracy],
    cfg_file=config_file,
    launcher='pytorch')


trainer = build_trainer(name=Trainers.ofa, default_args=args)  # 构建训练器
trainer.train()


# 模型训练后模型直接产生在WORKSPACE/ModelFile.TRAIN_OUTPUT_DIR目录，基于新模型再构建一个pipeline
ofa_ocr_pipeline = pipeline(task=Tasks.ocr_recognition,
                            model=os.path.join(WORKSPACE, ModelFile.TRAIN_OUTPUT_DIR)) # 训练的结果会保存在这里
new_ocr_result = ofa_ocr_pipeline(test_image)
# 期待这里打印理想的ocr效果
print(new_ocr_result[OutputKeys.TEXT], 'GT：蝶恋花')
