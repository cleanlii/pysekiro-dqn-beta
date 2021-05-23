# test_model.py

# 注1：因为肖智清强化学习教材中实例代码均使用TF2.0，故此项目以TF2.4.1为准
# 注2：实际训练中需配合游戏本体修改器使用（无限复活功能）
# 注3：存在死亡后复活锁定消失的问题

from train_model import Agent
target = 'Genichiro_Ashina' # 苇名弦一郎
# target = 'Genichiro_Way_of_Tomoe' # 巴流 苇名弦一郎
# target = 'True_Monk' # 宫内破戒僧
# target = 'Isshin_the_Sword_Saint' # 剑圣一心

# 保存路径
SAVE_PATH = 'Boss/' + target + '/' + target

'''
# 训练
train = Agent(
    save_memory_path = SAVE_PATH + '_memory.json',    # 保存记忆
    # load_memory_path = SAVE_PATH + '_memory.json',    # 加载记忆
    save_weights_path = SAVE_PATH + '_w.h5',    # 保存模型权重
    # load_weights_path = SAVE_PATH + '_w.h5'     # 加载模型权重
)
train.run()
'''

# 测试
test = Agent(
    load_weights_path = target + '_w.h5'
)
test.run()
