from urllib.parse import quote
import time
import datetime
import random
import aiohttp
import asyncio
import requests

now_time = time.localtime()
local_times_format = time.strftime("%Y-%m-%d+%H:%M:%S",now_time).replace(":","%3A")

def outTradeNo():
    now = datetime.datetime.now()
    # 格式化日期时间部分
    date_time_str = now.strftime('%Y%m%d%H%M%S')
    # 生成8位随机数
    random_numbers = "".join(str(random.randint(0, 9)) for _ in range(8))
    out_trade_no = date_time_str + random_numbers
    return out_trade_no

proxy_url = "http://218.203.164.245:9980/style/school/pc/proxy.jsp?url="
params = "optype=apply&UserName=18418613417&CodeType=0&LocalTime={0}&UUID={1}".format(local_times_format,outTradeNo())
verifycode_url = "http://127.0.0.1:9880/verifycode"
final_params_encode = quote(verifycode_url + "?" + params,safe="")
final_url = proxy_url + final_params_encode

async def fetch(session, url):
    async with session.get(url) as response:
        return await response.text()

async def main():
    async with aiohttp.ClientSession() as session:
        html = await fetch(session, final_url)
        print(html)

"""去掉注释运行即可""" #短信发送
#loop = asyncio.get_event_loop()
#loop.run_until_complete(main())
#loop.close()

"""
paramStr（未验证）paramStr 是用户的认证参数 | 
将其添加到请求头中再请求生成图像验证码的URL 理论上这个时候我的网络认证图像验证码
就是返回的图像中的
"""
verify_code_create_url = "http://218.203.164.245:9980/createVerifycode"
verify_code_referer = """http://218.203.164.245:9980/style/school/pc/index.jsp?paramStr=h1fVAB6tzagRUTAod6ZSXgc3UILQ1%2BKO%2FC8eUcUpCvwQV8XV9Mzb2yhSuaWiID5zylUckREwcIJX3w28mE6nLKpK3z8%2Fd9MB7Kgwza%2Bw8%2BkbwELAyfN2zOTnhcFVNOrsWOobyDIT06U94MQfjx9pTwOy2kVVzHD2tmlELyTseWDfHCzL%2FdwkW4k030ocIxvHYNDo6ysaKP7w8rd1Jsd7FFOc4YVhIhSJCOYwSnOxYtkHN1CC0NfijseaqGDYGc%2Bgl9OTHVXFTEev9emNDM3Ty1DYT8hMz2vJ8%2Bta2A6KaFl5qk3aMZz3XLspBcCymsYii9JEIMpdaNK07TUrJnWQ0MyZYsxedjdtT1klfuGEMeD6%2Bh0mas3I%2Bhg6jsbUFekS"""
verify_code_headers = {
    "Referer":verify_code_referer
}
with open("./verify.jpg","wb") as verifyImg:
    verifyImg.write(requests.get(verify_code_create_url,headers=verify_code_headers).content)
    verifyImg.close

auth_url = "http://218.203.164.245:9980/page_auth.jsp"
auth_params = ""

