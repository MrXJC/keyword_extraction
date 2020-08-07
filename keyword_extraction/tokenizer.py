# -*- coding: utf-8 -*-
import sys
sys.path.append('..')
import jieba
import jieba.posseg as pseg
import re


class JiebaTokenizer(object):

    def __init__(self):
        self.tokenizer = jieba
        self.pos_tokenizer = pseg.dt
        self.stop_words = set([' ','  ','   '])
        self.text_clean=u"[^A-Z^a-z^0-9^\u4e00-\u9fa5^ ^-]"

    def set_userdict(self, filepath):
        jieba.load_userdict(filepath)

    def set_stopwords(self, filepath):
        mid = open(filepath, 'r').readlines()
        self.stop_words = self.stop_words | set([x.strip() for x in mid])

    def tokenize(self,
                 text,
                 withFlag=False):
        text = re.sub(self.text_clean, ' ', text)
        text = text.lower()
        return [w for w in self.tokenizer.cut(text) if len(w) >= 2 and w not in self.stop_words and w[0] != ' ']

    def stokenize(self,
                 text,
                 withFlag=False):
        text = re.sub(self.text_clean, ' ', text)
        text = text.lower()
        return [w for w in self.tokenizer.cut(text) if len(w) >= 2 and w[0] != ' ']


    def pos_tokenize(self, text, pos_filter=('n', 'vn', 'v', 'l', 'eng'), withFlag=False):
        text = re.sub(self.text_clean, ' ', text)
        text = text.lower()
        return [ wp.word for wp in self.pos_tokenizer.cut(text) if wp.flag in pos_filter
                and len(wp.word) >= 2 and wp.word not in self.stop_words and wp.word[0] != ' ']



if __name__ == "__main__":
    tokenizer = JiebaTokenizer()
    tokenizer.set_userdict('hr.dict')
    for name in [
            'baidu_stopwords.txt', 'cn_stopwords.txt', 'hit_stopwords.txt',
            'scu_stopwords.txt', 'own_stopwords.txt', 'en_stopwords.txt'
    ]:
        filename = 'stopwords/' + name
        tokenizer.set_stopwords(filename)
    text = '1、负责DR相关产品硬件和固件的开发和调试； 2、在线产品的故障处理和维护； 3、领导安排的其他相关工作；。1、自动化或电子类相关专业，本科以上学历； 2、2年以上机电一体化产品或大型医疗设备的运动控制软件设计及调试经验； 3、精通C语言，具备基于DSP，ARM的软件开发，熟悉嵌入式实时操作系统（uCOS、FreeRTOS等）； 4、熟悉硬件接口：UART，485，IIC,SPI，CAN等接口协议; 5、熟悉步进、伺服及直流无刷电机的使用； 6、具有一定的硬件原理图读图能力； 7、具有较强团队和沟通意识，工作积极主动；'
    for x in tokenizer.tokenize(text):
        print(x)
