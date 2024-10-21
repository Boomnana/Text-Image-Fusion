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
    df_list = [str(ele[1])+ele[1] for ele in df_list] #['bug类别：' + ele[0] + '；' + 'bug表现：' + ele[1] for ele in df_list]
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

def getData(allTaskId):
    result = []
    for row in tqdm(allTaskId):
        response = zhipuai.model_api.query_async_invoke_result(row)
        # print(response['data']['choices'])
        # print(response['data'])
        # x = input("是否继续？y/n")
        if response['success']:
            print(response)
            sys.exit()
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
    staticPrompt1 = "接下来我会给你发送型如：测试员在（）场景中，进行了（）操作，出现了（）缺陷。的句式，请你输出一个三元组<场景，操作，缺陷>。如果句子中没有上述内容，则输出error。每一条输入都是独立的，请不要私自融合或切分。"
    staticPrompt2 = "理解了，请您提供具体的句子，我会根据您提供的格式输出三元组。"
    staticPrompt3 = ''
    allTaskId = []
    file_path = './测试数据400_三元组.xlsx'
    data = pd.read_excel(file_path)
    coldata = data.iloc[:,1 : 5]
    # print(coldata.values.tolist()[0][3])
    # print(coldata["ima2text"][0])
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
   