交互日志：

在接下来的对话中，请尽量简化表述，只输出核心内容，并牢记上文提出的要求

请帮我优化这段代码，使其成为一个基于python的功能模块，要求如下：
能够加载PARQUET格式数据，文件名为yellow_tripdata_2026-01，生成数据质量报告（缺失率、异常值统计）；清洗数据并在注释中说明每步策略的理由；从行程时间中提取小时、星期、是否高峰等特征，并自行设计至少 2 个有意义的衍生特征。
PARQUET格式的示例如下：
import pyarrow.parquet as pq 
trips = pq.read_table('trips.parquet') 
trips = trips.to_pandas()
更详细的使用方法，参考https://arrow.apache.org/docs/python/parquet.html

在上一个模块和附带文件的基础上，修改第二个模块的代码，要求：
完成以下 4 项分析，每项对应至少 1 张图表，所有图表自动保存至 outputs/ 目录（与main.py位于同一路径下）：
1.出行需求时间规律（分小时、分工作日/周末展示骑行量）例如：绘制分小时平均订单量折线图；
2.区域热度分析（上下客量最高的 TOP 10 区域及高峰时段分布）例如：柱状或热力图；
3.车费影响因素分析（距离、时段、乘客人数与车费的关系）例如：行程距离-车费散点图；
4.完成订单速度与时段关系分析。

在模块2中，将输出的图标中的汉字替换为英语，并将所有以数据类型作为标记的内容更换为实际涵义（这里出现口口，我觉得是Unicode之类的问题，懒得搞干脆改英文了）

在生成图片的过程中，出现大量类似如下警告：
C:\Users\admin\PycharmProjects\PythonProject0\main.py:73: FutureWarning: 

Passing `palette` without assigning `hue` is deprecated and will be removed in v0.14.0. Assign the `y` variable to `hue` and set `legend=False` for the same effect.

  sns.barplot(x=top_pu.values, y=top_pu.index.astype(str),
  但图片生成成功，需要修改吗？

生成第三个模块，要求：
用 TensorFlow 或 PyTorch 构建神经网络，预测某区域某时段的出行需求量；
划分训练/测试集（8:2），绘制 loss 曲线，测试集报告 MAE 与 RMSE；
与随机森林进行对比实验，分析两种方法在此任务上的优劣。

生成第四个模块，要求：
可通过.py文件实现命令行问答循环：用户输入自然语言问题，系统对自然语言分析匹配，然后结合M1-3中的函数进行调用（提取关键词，输入到参数调用)，返回数字结论+图表路径；
支持不少于 5 种问题类型（如时段查询、区域排名、需求预测、可能费用等）；
接入api:这里github警告secret scan了
M1-3的文件名分别为：data_pipeline,picture_generation,data_prediction
（这里因为用的是我买的，AI建议生产环境改用环境变量，我寻思应该没人会乱用，拒绝了）

请设计一个可视化程序，并接入以上四个模块，要求有基本的图形界面，用户能用光标选择使用不同的模块功能，需要用户文字输入的地方生成一个文本框以供输入，涉及图片生成的功能，默认打开生成的第一个图片，并包含一个退出程序的按钮。
M1-4的文件名分别为：data_pipeline,picture_generation,data_prediction，AI_assistance

请检查一下以上所有的输出代码（M1-4和main），是否都是使用相对路径（这里提了几个加固的建议，但我拒绝了）

对比：
第一个模块我不会清洗，只根据网站里parquet的教程和dictionary写了导入数据和把数据分类的部分：
import pyarrow.parquet as pq
pq.write_table(table, 'yellow_tripdata_2026-01.parquet')
第二个模块我一开始的想法是根据倒入的表格的行和列来把数据分类，然后在分类的基础上生成图像，然后发现我不记得生成对应图标类型和颜色的代码，问AI问到一半太懒了直接让它改了

反思：
我承认我AI用的有点多，但我认为AI在这类涉及大量数据处理的任务上效率远高于人类
我用的是Claude，说实话打开那个图形界面的时候我都被震惊了，本来以为会是BIOS或者PE那种简易界面，我说贵有贵的道理，这个生产效率实在是。。。

