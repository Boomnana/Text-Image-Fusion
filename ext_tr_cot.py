import sys
import random

import zhipuai
import pandas as pd
import time
from tqdm import tqdm


def ASK(prompts):

    response = []
    while type(response)=="NoneType" or ('data' not in response):
        response = zhipuai.model_api.async_invoke(
            model="chatglm_turbo",
            prompt=prompts,
            # prompt = [{"role": "user", "content": "人工智能"}],
            top_p=0.7,
            temperature=0.95,
            # temperature=0.05,
        )
        time.sleep(1)
    # while 'data' not in response:
    #     time.sleep(1)
    data1 = response['data']
    taskId = data1['task_id']
    return taskId


def askZhipu(coldata, prompts, allTaskId, s, result):
    df_list = coldata.values.tolist()
    # df_list = ['bug报告：' + str(ele[0]) for ele in df_list]
    df_list = ['截图文本信息：'+str(ele[2])+'\nbug报告：'+str(ele[1]) for ele in df_list]
    # print(df_list[1])
    # sys.exit()
    # df_list = [ele[1] for ele in df_list]
    sub_evo = ["排版不是很好",
               "界面设计丑陋，过多留白",
               "页面排版残缺，整体视觉效果差，且过于单调。TOP10 字体设置太小。",
               "图片质量太低。",
               "严重影响用户体验",
               "界面单一，色彩暗淡",
               "不会及时出来…",
               "一点提示信息都没有，让人不爽",
               "不能让人一目了然。",
               "十分耗电。性能不是很好",
               "这个用户极为反感，用户体验极差！！！！",
               "提示音太大，出乎意料的就响了，给用户造成极大的不满，甚至吓一跳！！！用户体验极为不好！！",
               "界面布局严重浪费",
               "过程过长，需要耐心等待。",
               "反应时间较长",
               "用户体验上感觉不太好。",
               "页面布局简单，过于单调。",
               "过程比较长，用户体验不佳。",
               "用户体验不好。",
               "页面整体颜色偏白",
               "表情好难看啊！",
               "没有做到快速的回复",
               "界面单一，功能太少了",
               "匹配度不高。不好辨识。",
               ]
    for i in tqdm(range(len(df_list))):
        if i!=0 and i%99==0:
            time.sleep(100)
    # for row in tqdm(df_list):
        if s == "":
            prompt = {"role": "user", "content": #"作为一个数据分析师，你的任务是从提交的bug报告中提取关键信息。请你深入理解下面内容，并回答我接下来的问题。"
                                                 "作为一个数据分析师，你的任务是从截图的文本信息和提交的bug报告中提取关键信息。请你深入理解下面内容，并回答我接下来的问题。"
                                                 # "请看以下示例：需要深入理解内容：bug报告：用户修改原有的密码，若新密码与旧密码相同，竟没有对用户进行提示，可能会导致用户的账号安全得不到保证！例如：原始密码：hhhhhhh   新的密码：hhhhhhh   重复密码：hhhhhhh  这样是没有报错的"
                                                 "请看以下示例：需要深入理解内容：截图文本信息：4G\n13:48\n修改密码\n原始密码\n新的密码\n重复密码\n确定\nbug报告：用户修改原有的密码，若新密码与旧密码相同，竟没有对用户进行提示，可能会导致用户的账号安全得不到保证！例如：原始密码：hhhhhhh   新的密码：hhhhhhh   重复密码：hhhhhhh  这样是没有报错的"
                                                 "user：通过上述信息判断，这可能是在什么APP的什么界面？回答限定在十个字以内。"
                                                 "assistant：这是密码修改界面。"
                                                 "user：根据上述信息判断，出现了什么缺陷？"
                                                 "assistant：修改后的新密码与旧密码相同，没有对用户进行提示。"
                                                 "user：根据上述信息判断，是什么操作导致出现了这个缺陷？"
                                                 "assistant：用户进行密码修改。"
                                                 "user：综合上述信息输出格式为<场景，操作，缺陷>的三元组。"
                                                 "assistant：<密码修改界面，密码修改，新密码与旧密码相同未提示>"
                                                 "以下是截图文本信息和bug报告的内容：" + df_list[i]} #+ random.choice(sub_evo)}
            prompts[i].append(prompt)
            prompt = {"role": "assistant", "content": "好的，我会根据上述内容回答你接下来的问题。"}
            prompts[i].append(prompt)
            prompt = {"role": "user", "content": "通过上述信息判断，这可能是在什么APP的什么界面？回答限定在十个字以内。"}
        else:
            prompts[i].append({"role": "assistant", "content": str(result[i])})
            prompt = {"role": "user", "content": s}
        # print(prompt)
        # x = input("是否继续？y/n")
        # if (x == 'n'):
        #     sys.exit()
        prompts[i].append(prompt)
        Id = ASK(prompts[i])
        # prompts.pop()
        time.sleep(1)
        allTaskId.append(Id)

    time.sleep(70)

def getData(allTaskId):
    result = []
    for row in tqdm(allTaskId):
        response = zhipuai.model_api.query_async_invoke_result(row)
        # print(response['data']['choices'])
        # print(response['data'])
        # x = input("是否继续？y/n")
        if response['success']:
            data = response['data']
            choices = data['choices']
            content = choices[0]['content']
            result.append(content)
        else:
            time.sleep(10)
            response = zhipuai.model_api.query_async_invoke_result(row)
            if response['success']:
                data = response['data']
                choices = data['choices']
                content = choices[0]['content']
                result.append(content)
            else:
                result.append(' ')
        # x = input("是否继续？y/n")
        # if (x == 'n'):
        #     sys.exit()
    return result


if __name__ == "__main__":

    result = []
    # pre zero-shot
    # staticPrompt1 = "作为一个数据分析师，你的任务是从bug报告中提取关键信息。请根据以下报告内容，提取出格式为<场景，操作，缺陷>的三元组，如果信息不足或为空则输出error。以下是bug报告的内容："
    #
    # # pre one-shot
    # staticPrompt1 = "作为一个数据分析师，你的任务是从bug报告中提取关键信息。请根据以下报告内容，提取出格式为<场景，操作，缺陷>的三元组，如果信息不足或为空则输出error。以下是bug报告的内容：我的主页界面,点击查看“学习榜”，加载过于缓慢。"
    # staticPrompt2 = "<主页界面，点击查看“学习榜”，加载过于缓慢>"

    allTaskId = []
    allPrompts = []
    # file_path = './experiment/三元组_res.xlsx'
    file_path = './挑选数据.xlsx'
    data = pd.read_excel(file_path, keep_default_na=False)
    # data = pd.read_excel(file_path, sheet_name="JayMe",keep_default_na=False)
    coldata = data.iloc[:,1 : 5]
    # print(coldata.values.tolist()[0][1])
    # print(coldata["img-sence"][0])
    # sys.exit()
    # print(type(coldata))
    # x = input("是否继续？y/n")
    # if(x == 'n'):
    #     sys.exit()



    zhipuai.api_key = "enter your key"

    # zero-shot
    # prompts = {"role": "user", "content": staticPrompt1}
    # prompts2 = {"role": "assistant", "content": staticPrompt2}

    # one-shot
    # prompts1 = {"role": "user", "content": staticPrompt1}
    # prompts2 = {"role": "assistant", "content": staticPrompt2}
    # prompts = [prompts1, prompts2]

    for i in range(len(coldata)):
        allPrompts.append([])

    askZhipu(coldata,allPrompts,allTaskId, "",result)
    time.sleep(10)
    result = getData(allTaskId)
    # time.sleep(100)
    print(result[0])
    print(allPrompts[0])
    # allTaskId = []
    # askZhipu(coldata, allPrompts, allTaskId, "通过上述信息判断，这可能是在什么APP的什么界面？",result)
    # time.sleep(10)
    # result = getData(allTaskId)
    allTaskId = []
    askZhipu(coldata, allPrompts, allTaskId, "根据上述信息判断，出现了什么缺陷？",result)
    time.sleep(10)
    result = getData(allTaskId)
    # time.sleep(100)
    allTaskId = []
    askZhipu(coldata, allPrompts, allTaskId, "根据上述信息判断，是什么操作导致出现了这个缺陷？",result)
    time.sleep(10)
    result = getData(allTaskId)
    # time.sleep(100)
    allTaskId = []
    askZhipu(coldata, allPrompts, allTaskId, "综合上述信息输出格式为<场景，操作，缺陷>的三元组。",result)
    time.sleep(10)
    result = getData(allTaskId)
    # time.sleep(100)
    df = pd.DataFrame(result)
    df.to_excel("./output/outputcot_nomask.xlsx", index=False)

