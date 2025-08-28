# 通过大语言模型对提交者的答案进行评分和评价，并最终形成一份成绩单

from openai import OpenAI
import json
import pandas as pd
from pydantic import BaseModel

class EssayScore(BaseModel):
    score: int

def EssayScoring(question, answer, ref_answer, total_score):
    # 这里调用大语言模型进行评分和评价
    
    # 读取大语言模型的API密钥、base_url和模型名称
    with open("config.json", "r") as f:
        config = json.load(f)
        api_key = config["api_key"]
        base_url = config["base_url"]
        model_name = config["model_name"]

    messages = [{"role":f"system", "content":'''你是一个数据处理专家，现在你需要根据参考答案对学生的论述题进行评分。这个问题的内容是：{}/n/n这个问题的参考答案是：{}/n/n请根据参考答案对学生的回答进行评分，满分{}分。请深思熟虑后给出一个整数分数，不要进行解释。'''.format(question, ref_answer, total_score)},{"role":"user", "content": "学生的答案为：{}".format(answer)}]

    client = OpenAI(api_key=api_key, base_url=base_url)
    response = client.chat.completions.parse(
        model=model_name,
        messages=messages,
        response_format=EssayScore
    )

    response_data = response.choices[0].message.content
    score = EssayScore.model_validate_json(response_data)
    return score.score


# 打开提交结果，导入答题者的答案
result_file = "data/results.xlsx"
df = pd.read_excel(result_file)

# 删除第1列
df = df.drop(df.columns[0], axis=1)

# 使用列表存储所有答题者的成绩单
grade_sheets = []

# 前5题为单选题（每题1分），中间三题为多选（每题2分），后面两题为问答（第一题9分，第二题10分）
personal_grades = []
for index, row in df.iterrows():
    # 获取答题者姓名
    student_name = df.iloc[index, -1]
    personal_grades.append(student_name)

    # 第一题
    if df.iloc[index, 0] == "可读性":
        personal_grades.append(1)
    else:
        personal_grades.append(0)
    
    # 第二题
    if df.iloc[index, 1] == "中文字体":
        personal_grades.append(1)
    else:
        personal_grades.append(0)

    # 第三题
    if df.iloc[index, 2] == "NumPy":
        personal_grades.append(1)
    else:
        personal_grades.append(0)

    # 第四题
    if df.iloc[index, 3] == "最小最大标准化（归一化）":
        personal_grades.append(1)
    else:
        personal_grades.append(0)

    # 第五题
    if df.iloc[index, 4] == "平方误差":
        personal_grades.append(1)
    else:
        personal_grades.append(0)

    # 第六题
    if df.iloc[index, 5] == "数字（int有符号整型、float浮点型、complex复数、布尔型bool）, 字符串（string）, 列表（list）, 元组（tuple）, 字典（dictionary）, 集合（set）":
        personal_grades.append(2)
    else:
        personal_grades.append(0)

    # 第七题
    if df.iloc[index, 6] == "直接使用含有缺失值的特征, 删除含有缺失值的特征（当缺失的太多时）, 缺失值补全（当只有少量数据缺失时）":
        personal_grades.append(2)
    else:
        personal_grades.append(0)

    # 第八题
    if df.iloc[index, 7] == "欧氏距离（Euclidean Distance）, 曼哈顿距离（Manhattan Distance）, 余弦相似度（Cosine Similarity）":
        personal_grades.append(2)
    else:
        personal_grades.append(0)

    # 第九题为问答题，满分9分
    question = "阐述数据分类分析的目标是什么？并列举至少三种典型的分类算法。"
    answer = df.iloc[index, 8]
    ref_answer = '''数据分类分析的目标： 数据分类分析是监督学习领域的核心任务，其主要目标是构建一个模型，根据事物的已知特征，将其分配到预先定义的类别中。具体目标包括：
        ▪ 预测归属：最核心的目标是预测新样本所属的类别。
        ▪ 模式识别：识别不同类别数据所具有的共同特征和决定性规律，例如，识别垃圾邮件的关键特征。
        ▪ 决策支持：基于分类结果为决策提供依据，例如，银行根据客户信用风险分类来决定是否批准贷款。
    ◦ 典型的分类算法（至少列举三种）：
        ▪ 逻辑回归 (Logistic Regression)：虽然名字带有“回归”，但它是一种经典、高效的分类算法，尤其适用于二元分类，输出结果为属于某个类别的概率。
        ▪ 决策树 (Decision Tree)：通过构建一个树状模型，利用一系列“是/否”问题进行决策和分类，模型可解释性强。每个内部节点表示一个属性上的判断，每个分支代表一个判断结果的输出，最后每个叶节点代表一种分类结果。常用的决策树包括ID3、C4.5和CART。
        ▪ K-近邻 (K-Nearest Neighbors, KNN)：这是一种“懒惰学习”算法，不进行模型训练。在预测时，它通过寻找新样本周围K个最近邻居中出现频率最高的类别作为新数据的标签。
        ▪ 支持向量机 (Support Vector Machine, SVM)：在特征空间中寻找一个能将不同类别样本最大程度分开的“决策边界”（超平面）。在处理高维数据和非线性问题时非常有效，但更适合小批量样本的任务。
        ▪ 朴素贝叶斯 (Naive Bayes)：基于贝叶斯定理，假设特征之间相互独立。它简单、快速，在文本分类（如垃圾邮件过滤）中效果显著。
        ▪ 集成学习 (Ensemble Learning)：将多个弱学习器组合起来，形成一个更强大、更稳健的分类器，例如随机森林和梯度提升机。
        '''
    total_score = 9
    score = EssayScoring(question, answer, ref_answer, total_score)
    personal_grades.append(score)

    # 第十题为问答题，满分10分
    question = '''假设你获得了一个新的、包含混合数据类型（如数值、文本、类别等）的原始数据集，需要对其进行处理和分析。请你结合课件中“数据预处理和可视化”及“数据分析”的内容，描述你会如何进行数据处理和分析的整体流程，并举例说明在此过程中可能用到的Python第三方库或分析方法。 (本题为开放性问题，请结合所学知识自由发挥。)'''
    answer = df.iloc[index, 9]
    ref_answer = '''整体流程概述： 在工程实践中，数据预处理没有标准的流程，通常针对不同的任务和数据集属性而异。但一般而言，可以从数据类型识别开始，然后进行数据预处理，最后进入数据分析阶段。
    ◦ 1. 数据类型识别： 首先，我会识别数据集中的数据类型，包括结构化数据（如关系数据库表形式的数据）、非结构化数据（如文本、图片、视频）和半结构化数据（如JSON、XML文档）。这将指导后续的数据处理策略。
    ◦ 2. 数据预处理： 数据预处理是使用数据之前的必要步骤，以处理原始数据中可能存在的缺失值、重复值等问题。
        ▪ 数据清洗：
            • 去除唯一属性：识别并删除那些不能刻画样本自身分布规律的唯一属性，例如ID列。
            • 处理缺失值：根据缺失的程度和业务需求，选择合适的方法。如果缺失极少，可以考虑直接使用含有缺失值的特征；如果某一特征缺失值过多，可以考虑删除含有缺失值的特征；对于少量数据缺失，常用的补全方法有均值插补、同类均值插补、建模预测等。
        ▪ 特征编码：
            • 对于类别型数据（如文本标签），需要进行编码转换为数值型，以便模型处理。常用的方法有特征二元化（将数值属性转换为布尔值，设定阈值划分0和1）和独热编码（One-Hot Encoding）（用N位状态寄存器对N个取值进行编码，保证在任意时刻只有一位有效）。
        ▪ 数据标准化：
            • 对于数值型特征，为了消除不同属性间量纲和量级差异的影响，需要进行标准化。常用的方法包括最小最大标准化（归一化），将数据缩放到0-1范围；以及Z-score标准化（规范化），使数据符合标准正态分布。
        ▪ 可能用到的Python第三方库：在数据预处理阶段，Pandas库（提供DataFrame数据结构，方便进行数据清洗、转换和IO操作）和NumPy库（用于高效的数值计算）是不可或缺的。
    ◦ 3. 数据分析： 数据分析的目标是将海量信息转化为可执行的洞察，优化资源配置，提升效率，并能预测未来趋势。根据具体的业务问题，选择合适的分析方法：
        ▪ 如果目标是发现数据中的潜在分组或结构：我会采用聚类分析。例如，可以使用K-均值算法（K-Means）高斯混合模型（GMM）欧氏距离、曼哈顿距离或余弦相似度。
        ▪ 如果目标是预测一个连续的数值结果：我会采用回归分析。例如，可以使用简单线性回归或多元线性回归来建模自变量与因变量之间的线性关系；如果关系是非线性的，可以考虑多项式回归；为了防止过拟合和处理多重共线性，可以考虑岭回归或Lasso回归。
        ▪ 如果目标是将数据分配到预定义的类别中：我会采用数据分类分析。例如，可以使用**决策树（Decision Tree）构建可解释性强的分类模型；或者使用K-近邻（KNN）**算法基于邻居的类别进行分类；对于二元分类，逻辑回归也是一个高效的选择；对于高维或非线性问题，**支持向量机（SVM）**可能会很有效。
        ▪ 如果数据维度过高，需要进行降维以简化数据集：我会使用主成分分析（PCA），它通过正交变换将可能相关的变量转换为线性不相关的变量（主成分），同时保留数据集方差贡献最大的特征。
        ▪ 数据可视化：在分析过程中，我会使用Matplotlib和Seaborn库绘制各种图表（如散点图、直方图、热力图、箱线图等），以直观地探索数据分布、特征关系和分析结果。
        ▪ 可能用到的Python第三方库：Scikit-learn是机器学习的核心库，提供了丰富的分类、回归、聚类和降维算法。对于复杂的任务，特别是涉及神经网络和深度学习时，我会考虑使用PyTorch或TensorFlow。
    '''
    total_score = 10
    score = EssayScoring(question, answer, ref_answer, total_score)
    personal_grades.append(score)

    sum_score = sum(personal_grades[1:])
    personal_grades.append(sum_score)

    # 将该答题者的成绩单添加到总的成绩单列表中
    grade_sheets.append(personal_grades)

# 按照grade_sheets 中的最后一列（总分）进行降序排序
grade_sheets.sort(key=lambda x: x[-1], reverse=True)

# 输出所有答题者的成绩单
for sheet in grade_sheets:
    print(sheet)

# 将成绩单保存到Excel文件中
grade_df = pd.DataFrame(grade_sheets, columns=["姓名","单选1", "单选2", "单选3", "单选4", "单选5", "多选1", "多选2", "多选3", "问答1", "问答2", "总分"])
grade_df.to_excel("grade_sheets.xlsx", index=False)
