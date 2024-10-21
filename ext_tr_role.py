import sys

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


def askZhipu(coldata, prompts, allTaskId):
    df_list = coldata.values.tolist()
    df_list = [str(ele[3])+str(ele[0]) for ele in df_list] #['bug类别：' + ele[0] + '；' + 'bug表现：' + ele[1] for ele in df_list]
    # print(df_list[1])
    # sys.exit()
    # df_list = [ele[1] for ele in df_list]
    for row in tqdm(df_list):
        prompt = row
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
    # time.sleep(570)

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
    # role zero-shot
    staticPrompt1 = "假如你是一位数据分析师，专门进行依据bug描述提取为一个三元组<场景，操作，缺陷>的工作。接下来请根据我给你的详细描述，输出一个三元组，在回答时，请使用严格的格式：<场景，操作，缺陷>，如果描述内容不能做出判断，则直接输出error。现在，请根据我接下来输入的信息进行提取三元组。"
    staticPrompt2 = "明白了，请提供详细的bug描述，我将据此提取并输出:<场景，操作，缺陷>，或者直接输出:error。"
    # role one-shot
    # staticPrompt1 = "假如你是一位数据分析师，专门进行依据bug描述提取为一个三元组<场景，操作，缺陷>的工作。接下来请根据我给你的详细描述，输出一个三元组，在回答时，请使用严格的格式：<场景，操作，缺陷>，如果描述内容不能做出判断，则直接输出error。请看以下示例：\n示例报告：“个人主页界面，个人信息修改中，对出生日期选择没有限制，可以选择未来的任何时间。”\n示例三元组：“<个人主页，修改出生日期，可以选择未来时间>。”\n现在，请根据我接下来输入的信息进行提取三元组。"
    # staticPrompt2 = "明白了，请提供详细的bug描述，我将据此提取并输出:<场景，操作，缺陷>，或者直接输出:error。"
    # staticPrompt3 = ''
    allTaskId = []
    file_path = './experiment/三元组_res.xlsx'
    data = pd.read_excel(file_path, sheet_name="JayMe",keep_default_na=False)
    coldata = data.iloc[:,1 : 5]
    # print(coldata.values.tolist()[2][0])
    # print(coldata["img-sence"][0])
    # sys.exit()
    # print(type(coldata))
    # x = input("是否继续？y/n")
    # if(x == 'n'):
    #     sys.exit()



    zhipuai.api_key = "enter your key"

    prompts1 = {"role": "user", "content": staticPrompt1}
    prompts2 = {"role": "assistant", "content": staticPrompt2}
    prompts = [prompts1, prompts2]

    i = 0
    while (i < len(coldata)):
        askZhipu(coldata[i:i+100], prompts, allTaskId)
        i = i + 100
        time.sleep(10)
        result = getData(allTaskId)
        df = pd.DataFrame(result)
        df.to_excel("./output/output"+ str(i) + ".xlsx", index=False)

    # askZhipu(coldata[1400:], prompts, allTaskId)


    # result = getData(allTaskId)
    df = pd.DataFrame(result)
    df.to_excel('./output/output.xlsx', index=False)