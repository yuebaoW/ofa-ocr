import os

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.outputs import OutputKeys


ocr_recognize = pipeline(Tasks.ocr_recognition,
                         model='damo/ofa_ocr-recognition_handwriting_base_zh',
                         model_revision='v1.0.1')


def computer_acc_word(infer_sentence, gt_sentence):
    '''
        比较两个文本，两个文本有几处不一样，以及不一样的地方。
    :param infer_sentence:
    :param gt_sentence:
    :return:
    '''
    infer_sentence = infer_sentence.replace(',', '').replace('，', '').replace('.', '')
    gt_sentence = gt_sentence.replace(',', '').replace('，', '')

    m = 0
    words = []
    for i in range(len(infer_sentence)):
        if infer_sentence[i] == gt_sentence[i]:
            continue
        else:
            m += 1
            words.append(infer_sentence[i])

    return m, words


def pre_model(src_img_dir):
    sum_words = 0
    mistake_words = 0

    all_id_sum = 0
    true_id_sum = 0

    for img in os.listdir(src_img_dir):
        try:
            img_path = os.path.join(src_img_dir, img)

            result = ocr_recognize(img_path)
            # print(result)

            infer_sentence = result[OutputKeys.TEXT][0]
            # print('in', infer_sentence)
            gt_sentence = img.split('.')[0]
            # print('gt', gt_sentence)
            m, words = computer_acc_word(infer_sentence, gt_sentence)

            sum_words += len(gt_sentence)
            mistake_words += m
            all_id_sum += 1

            if m > 0:
                print("GT:", img, '  识别结果:', result[OutputKeys.TEXT], '  识别错字：', words, '  错字数：', m)
            else:
                true_id_sum += 1
                print('GT:', img, '  识别结果:', result[OutputKeys.TEXT])
        except:
            print('-----------', img)

    print('单字维度准确率：', (sum_words-mistake_words)/sum_words, '  总字数：', sum_words, '  识别错误字数：', mistake_words)
    print('图像纬度准确率：', true_id_sum / all_id_sum)


src_img_dir = 'train_data/test_5'
pre_model(src_img_dir)
