# -*- coding: utf-8 -*-

from keyword_extraction.tokenizer import JiebaTokenizer
from keyword_extraction.w2v import FunctionEmbedding
from keyword_extraction.rank import EnsembleRanker, TextRanker, TextRankerW2V, PositionRanker, PositionRankerW2V

if __name__ == '__main__':
    # 初始化分词器，也可以自己实现集成Tokenizer
    tokenizer = JiebaTokenizer()
    # 设置词表
    tokenizer.set_userdict('res/user.dict')
    # 设置停用词表
    tokenizer.set_stopwords('res/stopwords.txt')
    # 初始化词向量
    w2v = FunctionEmbedding('res/embedding/vec.txt', tokenizer= tokenizer)
    # 初始化EnsembleRanker
    eranker = EnsembleRanker(tokenizer=tokenizer,
                             w2v=w2v,
                             keyword_path='res/keyword.dict')
    text = '工作描述：投资分析与建议：根据投资机构与高净值客户需求，研究上市公司、投资项目等所处行业、业务模式、成长性及风险性分析，提供投资建议；方案执行：对量化私募证券与股权投资基金客户进行尽职调查、可行性分析、撰写项目投资分析报告；客户关系维护：对客户进行跟踪维护，及时进行投后管理和持续督导，对投资者举办投资沙龙、策略会等；业务研究：研究超过20家上市公司融资情况与资本运作，评估和发掘公司潜在业务机会；媒体运营：公众号日常运营与维护，连续两个月提高点击量20%以上工作描述：协助研究部门跟踪及定时进行行业及公司基本面研究，包括财务数据收集与整理，财务报表分析，建立盈利预测模型和估值模型包括DCF、相对估值法等；负责撰写及推送英文周报（中国A股、港股、部分大宗商品）共40余篇，熟悉股票及二级市场的运行规律以及掌握宏观经济基本研究方法；定期对在香港股市新发行的IPO招股书进行及时性评论，给客户提供打新建议工作描述：作为项目管理团队成员，参与5个客户项目（保险、奢侈品、商业地产、通讯等行业）的全生命周期包括需求分析、公司内部数据库结构分析及描绘数据模型，设计仪表盘和分析报告，上线测试与用户培训等过程；常驻客户项目并与项目经理、技术团队合作沟通敲定项目需求与实施方法，确保对最终用户的需求给出解决方案；从零开始创建应用项目包括数据分析报告及KPI仪表盘等，在对公司数据库结构的全面了解下处理大容量数据源，通过实施ETL过程搭建数据模型'
    topic = '投资经理金融分析师咨询顾问'
    keywords = eranker.keyword_rank(text,
                                    num_keyphrase=40,
                                    beta_matrix=0.75,
                                    beta_vector=0.75,
                                    matrix_weights=[0.1, 0.9],
                                    vector_weights=[0, 0.2, 0.2, 0.6],
                                    topic=topic,
                                    is_expand=True,
                                    pos_filter=('n', 'vn', 'v', 'l', 'eng'),
                                    with_weights=True)
    print(keywords)
