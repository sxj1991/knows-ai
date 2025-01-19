from langchain_core.prompts import PromptTemplate
from ollama import Client


def template_prompt(question: str) -> str:
    """
    定义基础提示词模版
    :return:
    """
    template = """你是一个提供帮助的语言助手.

    根据用户提供的文字:{question}，总结不少于100字摘要，并且仿写不少于100字相似内容。
    回答的格式为："summary":"摘要内容" "similar":"相似内容"
    """

    # 构造 PromptTemplate
    prompt = PromptTemplate(
        input_variables=["question"],
        template=template
    )

    return prompt.format(question=question)


def main():
    client = Client(host='http://localhost:11434')
    prompt = template_prompt(question="""首先，我们对估算技术缺乏有效的研究，更加严肃地说，它反映了一种悄无声息，但
并不真实的假设——一切都将运作良好。
第二，我们采用的估算技术隐含地假设人和月可以互换，错误地将进度与工作量相互
混淆。
第三，由于对自己的估算缺乏信心，软件经理通常不会有耐心持续地进行估算这项工
作。
第四，对进度缺少跟踪和监督。其他工程领域中，经过验证的跟踪技术和常规监督程
序，在软件工程中常常被认为是无谓的举动。
第五，当意识到进度的偏移时，下意识（以及传统）的反应是增加人力。这就像使用
汽油灭火一样，只会使事情更糟。越来越大的火势需要更多的汽油，从而进入了一场注定会
导致灾难的循环。""")

    response = client.chat(model='qwen2.5:3b-instruct-q8_0', format='json', messages=[
        {
            'role': 'user',
            'content': prompt,
        },
    ])

    print(f'resp:{response.message.get("content")}')


if __name__ == '__main__':
    main()
