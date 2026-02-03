import os
from openai import OpenAI

def main():
    base_url = "https://api.siliconflow.cn/v1"
    api_key = "sk-rohlpqmymalourknivvjykdbuxxdgdagkqixmcbnjtghtttc"  # 不要硬编码

    if not api_key:
        raise RuntimeError("请先设置环境变量 ONEAI_API_KEY")

    client = OpenAI(base_url=base_url, api_key=api_key)

    # 1) 尝试列出模型（有些代理不支持 /models，会报错）
    try:
        models = client.models.list()
        model_ids = [m.id for m in models.data]
        print("models.list() 返回的模型：")
        for mid in model_ids:
            print(" -", mid)
    except Exception as e:
        print("models.list() 调用失败（代理可能不支持 /models）：", repr(e))

    # 2) 发一条消息，让它自报版本，同时打印 API 返回的 response.model
    resp = client.chat.completions.create(
        model="deepseek-ai/DeepSeek-V3.2",
        messages=[
            {"role": "system", "content": "你需要简洁回答。"},
            {
                "role": "user",
                "content": (
                    "请告诉我你具体是什么模型版本或引擎标识。"
                    "如果你无法确定，请直接说无法确定。"
                    "另外请原样输出你认为的 model 名称。"
                ),
            },
        ],
        temperature=0,
    )

    print("\nAPI 返回的 response.model：", resp.model)
    print("模型回复：")
    print(resp.choices[0].message.content)

if __name__ == "__main__":
    main()