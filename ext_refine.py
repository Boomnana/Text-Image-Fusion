import sys

import zhipuai
import pandas as pd
import time
from tqdm import tqdm


def ASK(prompts):

    response = []
    while response==None or type(response)=="NoneType" or ('data' not in response):
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


def askZhipu(coldata, prompts, allTaskId):
    df_list = coldata.values.tolist()
    # df_list = [str(ele[3])+str(ele[0]) for ele in df_list] #['bug类别：' + ele[0] + '；' + 'bug表现：' + ele[1] for ele in df_list]
    # print(df_list[1])
    # sys.exit()
    # df_list = [ele[1] for ele in df_list]
    for row in tqdm(df_list):
        # prompt = {"role": "user", "content": "如下文本中存在一个三元组，即<场景;操作;缺陷>。请将其找出并输出为<场景;操作;缺陷>，中间用”;“隔开:" + row}
        prompt = "如下文本中存在一个三元组，即<场景;操作;缺陷>。请将其找出并输出为<场景;操作;缺陷>，中间用”;“隔开:<场景，操作，缺陷>三元组如下:" + row
        # print(prompt)
        # x = input("是否继续？y/n")
        # if (x == 'n'):
        #     sys.exit()
        temp = {'role': 'user', 'content': prompt}
        prompts.append(temp)
        Id = ASK(prompts)
        prompts.pop()
        time.sleep(1)
        allTaskId.append(Id)

    time.sleep(70)
    # time.sleep(350)

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
    staticPrompt1 = "作为一个数据分析师，你的任务是从bug报告中提取关键信息。请根据以下报告内容，提取出格式为<场景，操作，缺陷>的三元组，如果信息不足或为空则输出error。以下是bug报告的内容："

    # pre one-shot
    staticPrompt1 = "如下文本中存在一个三元组，即<场景;操作;缺陷>。请将其找出并输出为<场景;操作;缺陷>，中间用”;“隔开:“<场景，操作，缺陷>三元组如下:\n\n1.场景:搜索一栏\n 操作:输入内容\n 缺陷:只能取消，不能确认\n\n2.场景:搜索一栏\n 操作:输入内容\n 缺陷:无\n\n3.场景:搜索一栏\n 操作:无\n 缺陷:无\n\n4.场景:搜索一栏\n操作:取消\n 缺陷:无\n\n5.场景:搜索一栏\n 操作:确认\n 缺陷:只能取消，不能确认\n\n6.场景:搜索-栏\n 操作:确认\n 缺陷:无\n\n7.场景:搜索一栏\n 操作:无\n 缺陷:无。”"
    staticPrompt2 = "<搜索一栏;输入内容;只能取消，不能确认>"

    allTaskId = []
    # file_path = './experiment/三元组_res - refine.xlsx'
    file_path = './output/outputcot.xlsx'
    data = pd.read_excel(file_path, keep_default_na=False)
    # data = pd.read_excel(file_path, sheet_name="雅思听力",keep_default_na=False)
    coldata = data.iloc[:]
    # coldata = coldata["cot-zero"]
    # print(coldata.values.tolist()[2][0])
    # print(coldata.values.tolist()[0])
    # print(coldata["role-zero"][0])
    # sys.exit()
    # print(type(coldata))
    # x = input("是否继续？y/n")
    # if(x == 'n'):
    #     sys.exit()

    prompts = []

    zhipuai.api_key = "enter your key"

    # zero-shot
    # prompts = {"role": "user", "content": staticPrompt1}
    # prompts2 = {"role": "assistant", "content": staticPrompt2}

    # one-shot
    prompts1 = {"role": "user", "content": staticPrompt1}
    prompts2 = {"role": "assistant", "content": staticPrompt2}
    prompts = [prompts1, prompts2]

    # rangeitem = ["pre-one","suf-zero","suf-one","cot-zero"]
    rangeitem = [0]
    for item in rangeitem:
        i = 0
        allTaskId = []
        coldata1 = coldata[item]
        while (i < len(coldata1)):
            askZhipu(coldata1[i:i+100], prompts, allTaskId)
            i = i + 100
            time.sleep(10)
            # result = getData(allTaskId)
            # df = pd.DataFrame(result)
            # df.to_excel("./output/output"+ str(i) + ".xlsx", index=False)

        # askZhipu(coldata[1400:], prompts, allTaskId)


        result = getData(allTaskId)
        df = pd.DataFrame(result)
        df.to_excel('./output/outputcot_refine.xlsx', index=False)
